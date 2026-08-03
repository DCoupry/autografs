"""Self-templated round trips: rebuild each crystal from its OWN blueprint.

The library round trip (``roundtrip.py``) asks whether a structure is
expressible in the *cataloged* abstraction: its net must match a library
blueprint and its harvested units must fit that blueprint's slots. This
driver removes the library from the question entirely
(coverage plan stage 3). ``topology_from_deconstruction`` erects the
structure's own blueprint - one slot per building unit at its real
position, one shared connection point per cut bond - and the rebuild
maps every slot to its own unit's deduplicated representative fragment.

What remains under test is the rigid-unit abstraction itself: can rigid
representative building blocks, placed on the crystal's own net
embedding with the cell re-optimized from covalent targets, regenerate
the material? Failures are therefore attributable to unit rigidity and
instance-to-representative variation, not to net identification,
library coverage, slot-type capacity, or the idealized embedding.

``verify_net`` runs against the self-blueprint - near-tautological by
construction, kept as an internal consistency check whose failures
expose representation limits (e.g. a linker bonding the same unit pair
through two images, which the simple-graph min-image convention cannot
hold) - and the decisive gates are composition and realized geometry.

Usage:
    python scripts/benchmarks/selftemplate.py CORPUS -o selftemplate.json \
        --checkpoint selftemplate.jsonl --n-jobs 10
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _corpus import collect as _collect  # noqa: E402
from embedding import bond_residuals  # noqa: E402

from autografs import Autografs  # noqa: E402
from autografs.builder import build_framework  # noqa: E402
from autografs.exceptions import (  # noqa: E402
    AutografsError,
    NetMismatchError,
    TopologyExtractionError,
)
from autografs.extract_topology import topology_from_deconstruction  # noqa: E402

_WORKER: dict = {}


def selftemplate_one(mofgen: Autografs, source, max_rmsd: float) -> dict:
    record: dict = {"outcome": None, "error": None, "seconds": 0.0}
    start = time.perf_counter()
    try:
        result = mofgen.deconstruct(source)
    except (AutografsError, ValueError, KeyError, IndexError) as exc:
        record["outcome"] = "deconstruction_failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = time.perf_counter() - start
        return record
    record["n_units"] = len(result.units)
    record["n_fragment_types"] = len(result.fragments)
    record["fold"] = result.n_periodic_components
    if result.rod_units:
        record["outcome"] = "skipped_rod"
        record["seconds"] = time.perf_counter() - start
        return record
    # catenated structures are IN scope: the recipe holds every
    # component's units, so the erected blueprint is a disconnected
    # quotient whose nets share the one real cell at their true
    # relative offset, and the build places all of them together
    try:
        topology, mapping = topology_from_deconstruction(result)
    except TopologyExtractionError as exc:
        record["outcome"] = "no_blueprint"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = time.perf_counter() - start
        return record
    record["n_slots"] = len(topology)
    mappings = {
        index: copy.deepcopy(result.fragments[name]) for index, name in mapping.items()
    }
    try:
        framework = build_framework(
            topology, mappings, max_rmsd=max_rmsd, verify_net=True
        )
    except NetMismatchError as exc:
        record["outcome"] = "verify_failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = time.perf_counter() - start
        return record
    except AutografsError as exc:
        record["outcome"] = "build_failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = time.perf_counter() - start
        return record
    except Exception as exc:  # noqa: BLE001 - a build bug is data
        record["outcome"] = "build_failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = time.perf_counter() - start
        return record
    built = framework.structure
    experimental = result.structure
    record["volume_ratio"] = (built.volume / len(built)) / (
        experimental.volume / len(experimental)
    )
    record["min_contact"] = framework.min_contact()
    record["bond_residual"] = bond_residuals(framework)
    matched = (
        built.composition.reduced_formula == experimental.composition.reduced_formula
    )
    record["formula"] = built.composition.reduced_formula
    record["experimental_formula"] = experimental.composition.reduced_formula
    record["outcome"] = "closed_self" if matched else "composition_mismatch"
    record["seconds"] = time.perf_counter() - start
    return record


def _init_worker(topofile: str | None, max_rmsd: float) -> None:
    _WORKER["mofgen"] = Autografs(topofile=topofile) if topofile else Autografs()
    _WORKER["max_rmsd"] = max_rmsd


def _run_worker(path: str) -> tuple[str, dict]:
    name = Path(path).name
    try:
        return name, selftemplate_one(_WORKER["mofgen"], path, _WORKER["max_rmsd"])
    except Exception as error:  # noqa: BLE001 - one bad CIF must not kill a sweep
        return name, {
            "outcome": "driver_error",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=3),
        }


def run(
    corpus: list[Path],
    *,
    max_rmsd: float = 0.5,
    topofile: str | None = None,
    n_jobs: int = 1,
    checkpoint: Path | None = None,
) -> dict:
    records: dict[str, dict] = {}
    if checkpoint and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                records[entry["name"]] = entry["record"]
        print(f"resuming: {len(records)} already done")
    todo = [p for p in corpus if p.name not in records]
    handle = checkpoint.open("a", encoding="utf-8") if checkpoint else None
    try:
        if n_jobs > 1:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=_init_worker,
                initargs=(topofile, max_rmsd),
            ) as pool:
                for index, (name, record) in enumerate(
                    pool.map(_run_worker, [str(p) for p in todo], chunksize=2), 1
                ):
                    records[name] = record
                    _report(index, len(todo), name, record, handle)
        else:
            _init_worker(topofile, max_rmsd)
            for index, path in enumerate(todo, 1):
                name, record = _run_worker(str(path))
                records[name] = record
                _report(index, len(todo), name, record, handle)
    finally:
        if handle:
            handle.close()
    return {
        "benchmark": "selftemplate",
        "max_rmsd": max_rmsd,
        "n_structures": len(records),
        "outcomes": dict(Counter(r["outcome"] for r in records.values())),
        "structures": records,
    }


def _report(index: int, total: int, name: str, record: dict, handle) -> None:
    print(f"[{index}/{total}] {name}: {record['outcome']}", flush=True)
    if handle:
        handle.write(json.dumps({"name": name, "record": record}, default=str) + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", help="directory, glob, manifest, or single CIF")
    parser.add_argument("-o", "--output", default="selftemplate.json")
    parser.add_argument("--max-rmsd", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--topofile", default=None)
    args = parser.parse_args()

    corpus = _collect(args.corpus)
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
    for outcome, count in sorted(payload["outcomes"].items()):
        print(f"  {outcome:<24} {count}")


if __name__ == "__main__":
    main()
