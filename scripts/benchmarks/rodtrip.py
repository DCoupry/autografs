"""Rod round-trip driver: deconstruct -> rebuild on the rod path -> verify.

The finite round-trip driver (``roundtrip.py``) reports any structure
containing a 1-periodic building unit as ``rod`` and stops, because a rod
cannot be placed by the slot-type mappings it enumerates. That leaves the
rod claim resting on library nets and synthetic fixtures. This driver
closes the gap: for every structure whose deconstruction yields at least
one rod, it harvests the rod, chooses lateral SBUs from the same
structure's finite fragments, and attempts ``build_rod`` on each
identified net, gated by exact net verification.

Outcomes, most informative first:

- ``closed``          built on the rod path and passed verify_net
- ``verify_failed``   built, but the realised quotient graph is not the
                      blueprint's - a different net, which is exactly
                      what the gate exists to catch
- ``build_failed``    every net/linker combination was gated (closure,
                      contact or alignment)
- ``no_run``          the identified net carries no axial or helical slot
                      run a rod can occupy
- ``no_net``          rods present but no library net matched, so there
                      is no blueprint to build on. This is the dominant
                      bucket and it is a *identification* limit, not a
                      building one
- ``no_rod``          no 1-periodic unit after all (the input list is a
                      prefilter, so this should be rare)
- ``deconstruction_failed`` / ``driver_error``

Deterministic, resumable, and parallel in the same way as roundtrip.py.

Usage:
    python scripts/benchmarks/rodtrip.py corpus_dir -o rodtrip.json
    python scripts/benchmarks/rodtrip.py corpus_dir --only rods.txt --n-jobs 12
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from _corpus import collect as _collect

from autografs import Autografs
from autografs.exceptions import AutografsError
from autografs.net import axial_runs, helical_runs
from autografs.rods import rod_fragment

# lateral SBU combinations tried per net before giving up
MAX_LINKERS_PER_NET = 4


def _lateral_candidates(result) -> list:
    """Finite fragments that could sit on the blueprint's lateral slots.

    Ordered by descending connectivity: a polytopic unit is a more
    informative first guess than a ditopic one, and the budget is small.
    """
    fragments = [
        fragment
        for fragment in result.fragments.values()
        if len(fragment.atoms.indices_from_symbol("X")) >= 2
    ]
    return sorted(
        fragments,
        key=lambda f: -len(f.atoms.indices_from_symbol("X")),
    )[:MAX_LINKERS_PER_NET]


def rodtrip_one(mofgen: Autografs, source, max_rmsd: float) -> dict:
    """Deconstruct one structure and attempt the verified rod rebuild."""
    record: dict = {"outcome": None, "net": None, "error": None}
    t0 = time.perf_counter()
    try:
        result = mofgen.deconstruct(source)
    except (AutografsError, ValueError, KeyError, IndexError) as exc:
        record["outcome"] = "deconstruction_failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = time.perf_counter() - t0
        return record

    record["n_rods"] = len(result.rod_units)
    record["n_fragments"] = len(result.fragments)
    record["net"] = result.net_candidates or None
    if not result.rod_units:
        record["outcome"] = "no_rod"
        record["seconds"] = time.perf_counter() - t0
        return record
    if not result.net_candidates:
        record["outcome"] = "no_net"
        record["seconds"] = time.perf_counter() - t0
        return record

    try:
        rod = rod_fragment(result.structure, result.rod_units[0])
        record["screw_order"] = rod.repeat.screw_order
        record["repeat"] = round(float(rod.repeat.repeat_length), 4)
        record["screw_angle"] = round(float(rod.repeat.screw_angle), 3)
    except (AutografsError, ValueError, IndexError) as exc:
        record["outcome"] = "build_failed"
        record["error"] = f"rod_fragment: {type(exc).__name__}: {exc}"
        record["seconds"] = time.perf_counter() - t0
        return record

    laterals = _lateral_candidates(result)
    saw_run = False
    saw_build = False
    for net in result.net_candidates:
        topology = mofgen.topologies[net]
        try:
            runs = list(axial_runs(topology)) + list(helical_runs(topology))
        except (AutografsError, ValueError):
            runs = []
        if not runs:
            continue
        saw_run = True
        for linker in laterals or [None]:
            try:
                framework = mofgen.build_rod(
                    topology,
                    rod,
                    linkers=linker,
                    max_rmsd=max_rmsd,
                    verify_net=True,
                )
            except AutografsError as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                continue
            except Exception as exc:  # noqa: BLE001 - a build bug is data
                record["error"] = f"{type(exc).__name__}: {exc}"
                continue
            saw_build = True
            record["outcome"] = "closed"
            record["rebuilt_net"] = net
            record["min_contact"] = framework.min_contact()
            record["formula"] = framework.structure.composition.reduced_formula
            record["experimental_formula"] = (
                result.structure.composition.reduced_formula
            )
            record["seconds"] = time.perf_counter() - t0
            return record

    if saw_build:
        record["outcome"] = "verify_failed"
    elif saw_run:
        record["outcome"] = "build_failed"
    else:
        record["outcome"] = "no_run"
    record["seconds"] = time.perf_counter() - t0
    return record


_WORKER: dict = {}


def _init_worker(topofile: str | None, max_rmsd: float) -> None:
    _WORKER["mofgen"] = Autografs(topofile=topofile)
    _WORKER["max_rmsd"] = max_rmsd


def _run_worker(path: str) -> tuple[str, dict]:
    name = Path(path).name
    try:
        record = rodtrip_one(_WORKER["mofgen"], path, _WORKER["max_rmsd"])
    except Exception as exc:  # noqa: BLE001 - never lose a sweep to one CIF
        record = {"outcome": "driver_error", "error": f"{type(exc).__name__}: {exc}"}
    return name, record


def _load_checkpoint(path: Path) -> dict:
    records: dict = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        records[entry["name"]] = entry["record"]
    return records


def run(
    corpus: list[Path],
    max_rmsd: float = 0.35,
    topofile: str | None = None,
    n_jobs: int = 1,
    checkpoint: Path | None = None,
) -> dict:
    records = _load_checkpoint(checkpoint) if checkpoint is not None else {}
    if records:
        print(f"  resuming: {len(records)} done")
    pending = [p for p in sorted(corpus) if p.name not in records]
    handle = checkpoint.open("a", encoding="utf-8") if checkpoint else None
    try:
        if n_jobs > 1 and pending:
            executor = ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=_init_worker,
                initargs=(topofile, max_rmsd),
            )
            with executor:
                stream = executor.map(
                    _run_worker, [str(p) for p in pending], chunksize=1
                )
                for index, (name, record) in enumerate(stream, 1):
                    records[name] = record
                    _report(index, len(pending), name, record, handle)
        elif pending:
            _init_worker(topofile, max_rmsd)
            for index, path in enumerate(pending, 1):
                name, record = _run_worker(str(path))
                records[name] = record
                _report(index, len(pending), name, record, handle)
    finally:
        if handle is not None:
            handle.close()
    outcomes = Counter(r["outcome"] for r in records.values())
    return {
        "benchmark": "rodtrip",
        "max_rmsd": max_rmsd,
        "n_structures": len(records),
        "outcomes": dict(sorted(outcomes.items())),
        "structures": dict(sorted(records.items())),
    }


def _report(index: int, total: int, name: str, record: dict, handle) -> None:
    print(f"  [{index}/{total}] {name:<44} {record['outcome']}", flush=True)
    if handle is not None:
        handle.write(json.dumps({"name": name, "record": record}) + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", help="directory, glob, or single CIF")
    parser.add_argument("-o", "--output", default="rodtrip.json")
    parser.add_argument(
        "--only",
        default=None,
        help="text file of filenames to restrict to (one per line); "
        "use a prior rod scan so the sweep does not re-deconstruct "
        "structures already known to be rod-free",
    )
    parser.add_argument("--max-rmsd", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--topofile", default=None)
    args = parser.parse_args()

    corpus = _collect(args.corpus)
    if args.only:
        wanted = {
            line.strip()
            for line in Path(args.only).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        corpus = [p for p in corpus if p.name in wanted]
    if args.limit is not None:
        corpus = corpus[: args.limit]
    if not corpus:
        raise SystemExit(f"no structures matched {args.corpus!r}")
    n_jobs = os.cpu_count() or 1 if args.n_jobs == -1 else args.n_jobs
    payload = run(
        corpus,
        max_rmsd=args.max_rmsd,
        topofile=args.topofile,
        n_jobs=n_jobs,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
    )
    Path(args.output).write_text(json.dumps(payload, indent=1, default=str))
    print(f"\n{payload['n_structures']} structures -> {args.output}")
    for outcome, count in payload["outcomes"].items():
        print(f"  {outcome:<24} {count}")


if __name__ == "__main__":
    main()
