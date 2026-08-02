"""Calibrate DIRECTION_WEIGHT against the displacement-seeded optimizer.

The shipped weight (0.5) was selected by the #207 sweep, whose optimizer
explored the displacement block through a nearly degenerate initial
simplex (scipy seeds exactly-zero coordinates with zdelt = 0.00025,
four orders of magnitude below xatol), so the calibration never saw the
relaxed objective's true optima. Seeding the main solve reaches them,
and on shape-mismatched nets they are the #197 pathology (pts inflates
its cell 2.9x at guard-passing closure) - so the seeded explorer needs
its own weight, chosen the #207 way: on the population, never on one
material.

Protocol (per structure, first N of the corpus manifest):

1. deconstruct once;
2. lock a (net, mapping): the first candidate mapping whose FIXED-slot
   build passes ``verify_net`` and reproduces the experimental reduced
   formula - the faithful-rebuild criterion of #207, decided by a
   weight-independent build so every arm rebuilds the same thing;
3. rebuild that mapping once per arm: the unseeded shipped config as
   baseline, and each swept weight with the main solve seeded;
4. record packing (|volume_ratio - 1|), closure (worst bond), and which
   candidate the guard kept, per arm.

The pinned stratum (no effective displacement freedom) must come out
byte-identical across arms; it is the measurement's internal check.

Usage:
    python scripts/benchmarks/calibrate_direction_weight.py \
        research/paper/data/corpus-2025-files.txt --limit 600 \
        -o calibration-seeded.json --n-jobs 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _corpus import collect as _collect  # noqa: E402
from embedding import bond_residuals  # noqa: E402
from roundtrip import candidate_mappings  # noqa: E402

import autografs.alignment  # noqa: E402
import autografs.builder  # noqa: E402
from autografs import Autografs  # noqa: E402
from autografs.exceptions import AutografsError  # noqa: E402

# the #207 grid, extended one step DOWN: with a working explorer the
# direction terms bite harder, so the optimum may sit below the old one
WEIGHTS = (0.1, 0.25, 0.5, 1.0, 2.0)

_WORKER: dict = {}


@contextmanager
def _arm(weight: float | None, seed_main: bool):
    """Temporarily set the direction weight and main-solve seeding."""
    saved_weight = autografs.alignment.DIRECTION_WEIGHT
    saved_seed = autografs.builder.SEED_MAIN_SOLVE
    if weight is not None:
        autografs.alignment.DIRECTION_WEIGHT = weight
    autografs.builder.SEED_MAIN_SOLVE = seed_main
    try:
        yield
    finally:
        autografs.alignment.DIRECTION_WEIGHT = saved_weight
        autografs.builder.SEED_MAIN_SOLVE = saved_seed


def _metrics(framework, result) -> dict:
    built = framework.structure
    experimental = result.structure
    fold = result.n_periodic_components
    marker = framework.graph.graph.get("relaxation") or {}
    return {
        "volume_ratio": (built.volume / len(built))
        / (experimental.volume / len(experimental) * fold),
        "worst_bond": (bond_residuals(framework) or {}).get("max"),
        "n_slot_free": marker.get("n_slot_free"),
        "kept": marker.get("kept"),
    }


def calibrate_one(mofgen: Autografs, path: Path, max_rmsd: float) -> dict:
    record: dict = {"outcome": None, "seconds": 0.0}
    start = time.perf_counter()
    try:
        result = mofgen.deconstruct(str(path))
    except (AutografsError, ValueError, KeyError, IndexError) as exc:
        record["outcome"] = "deconstruction_failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = time.perf_counter() - start
        return record
    if result.rod_units or not result.net_candidates:
        record["outcome"] = "rod" if result.rod_units else "unidentified"
        record["seconds"] = time.perf_counter() - start
        return record
    experimental_formula = result.structure.composition.reduced_formula

    # lock (net, mapping) with a weight-independent fixed-slot build so
    # every arm rebuilds the identical assignment
    locked = None
    for net in result.net_candidates:
        topology = mofgen.topologies[net]
        for mappings in candidate_mappings(topology, result.fragments, max_rmsd):
            try:
                framework = mofgen.build(
                    topology, mappings=mappings, max_rmsd=max_rmsd, verify_net=True
                )
            except AutografsError:
                continue
            faithful = (
                framework.structure.composition.reduced_formula == experimental_formula
            )
            if faithful:
                locked = (net, mappings, framework)
                break
        if locked:
            break
    if locked is None:
        record["outcome"] = "no_faithful_rebuild"
        record["seconds"] = time.perf_counter() - start
        return record

    net, mappings, fixed_framework = locked
    record["outcome"] = "calibrated"
    record["net"] = net
    record["arms"] = {"fixed": _metrics(fixed_framework, result)}

    def relaxed_arm(label: str, weight: float, seed_main: bool) -> None:
        with _arm(weight, seed_main):
            try:
                framework = mofgen.build(
                    topology,
                    mappings=mappings,
                    max_rmsd=max_rmsd,
                    verify_net=True,
                    relax_embedding=True,
                )
            except AutografsError as exc:
                record["arms"][label] = {"error": f"{type(exc).__name__}: {exc}"}
                return
        record["arms"][label] = _metrics(framework, result)

    topology = mofgen.topologies[net]
    relaxed_arm("unseeded-0.5", 0.5, seed_main=False)
    for weight in WEIGHTS:
        relaxed_arm(f"seeded-{weight}", weight, seed_main=True)
    record["seconds"] = time.perf_counter() - start
    return record


def _init_worker(topofile: str | None, max_rmsd: float) -> None:
    _WORKER["mofgen"] = Autografs(topofile=topofile) if topofile else Autografs()
    _WORKER["max_rmsd"] = max_rmsd


def _run_worker(path: str) -> tuple[str, dict]:
    name = Path(path).name
    try:
        return name, calibrate_one(_WORKER["mofgen"], Path(path), _WORKER["max_rmsd"])
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
        "benchmark": "direction-weight-calibration-seeded",
        "corpus_note": "first N structures of the CoRE MOF 2025 manifest",
        "weights": list(WEIGHTS),
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
    parser.add_argument("-o", "--output", default="calibration-seeded.json")
    parser.add_argument("--max-rmsd", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=600)
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
