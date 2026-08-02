"""Why identified structures have no compatible fragment-to-slot mapping.

``roundtrip.py`` reports ``no_mapping`` for a structure whose net was
found and whose own harvested units then fit no slot of it. That is the
second-largest outcome in the census and, pooled, it says nothing: it is
consistent both with a *connectivity* mismatch (a unit whose arm count
matches no vertex figure of the net, which is a statement about how the
deconstruction cuts and how the net is decorated) and with a *geometric*
one (right arm count, arm directions too far from the slot's, which is
the same shape error the embedding analysis measures continuously).
Those are different findings and the distinction is not recoverable from
the round-trip output, because the driver records only that no mapping
existed.

This re-runs deconstruction on those structures and asks, per candidate
net and per slot type, which of the two walls was hit first --- reusing
``Fragment.has_compatible_symmetry``'s own two-stage test rather than
reimplementing it, so the answer is about the shipped predicate and not
about a model of it.

A structure is attributed to the *most favourable* candidate net it has,
since a single net that fails on arm count while another fails only on
geometry is a geometry-limited structure: some net of the library got
the connectivity right.

Usage:
    python scripts/benchmarks/mapping_gap.py CORPUS -o mapping-gap.json
    python scripts/benchmarks/mapping_gap.py CORPUS --only no-mapping.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _corpus import collect as _collect  # noqa: E402

from autografs import Autografs  # noqa: E402
from autografs.exceptions import AutografsError  # noqa: E402
from autografs.fragment import COMPATIBILITY_MAX_RMSD  # noqa: E402

_WORKER: dict = {}


def _slot_verdict(slot, fragments, max_rmsd: float) -> dict:
    """How close any harvested unit comes to occupying this slot.

    Two walls, in the order ``has_compatible_symmetry`` applies them:
    an equal dummy count is required outright, and only then are the
    arm directions matched. Reporting which one stopped a slot is the
    whole point of this driver, so both are recorded -- along with the
    best directional RMSD achieved among the right-sized candidates,
    which says whether a geometric failure was near or hopeless.
    """
    n_arms = len(slot.arm_units)
    right_size = [f for f in fragments if len(f.arm_units) == n_arms]
    if not right_size:
        available = sorted({len(f.arm_units) for f in fragments})
        return {
            "wall": "arm_count",
            "slot_arms": n_arms,
            "unit_arms": available,
            "best_rmsd": None,
        }
    # permissive first: does anything fit at the sieve's own threshold?
    if any(slot.has_compatible_symmetry(f, max_rmsd=max_rmsd) for f in right_size):
        return {"wall": None, "slot_arms": n_arms, "best_rmsd": 0.0}
    # nothing fits; find how far off the nearest right-sized unit is by
    # re-testing at widening thresholds. The predicate is a boolean, so
    # bisecting it is the only way to get a distance out of the shipped
    # code path rather than a reimplementation of it.
    low, high = max_rmsd, 4.0
    if not any(slot.has_compatible_symmetry(f, max_rmsd=high) for f in right_size):
        return {"wall": "geometry", "slot_arms": n_arms, "best_rmsd": None}
    for _ in range(12):
        middle = 0.5 * (low + high)
        if any(slot.has_compatible_symmetry(f, max_rmsd=middle) for f in right_size):
            high = middle
        else:
            low = middle
    return {"wall": "geometry", "slot_arms": n_arms, "best_rmsd": round(high, 3)}


def _net_verdict(topology, fragments, max_rmsd: float) -> dict:
    """Attribute one candidate net: the wall met by its hardest slot type."""
    per_slot = [
        _slot_verdict(slot_type, fragments, max_rmsd) for slot_type in topology.mappings
    ]
    walls = [s["wall"] for s in per_slot if s["wall"]]
    if not walls:
        return {"wall": None, "slots": per_slot}
    # geometry is the *softer* wall, so a net reports arm_count only when
    # a slot type has no right-sized unit at all
    wall = "arm_count" if "arm_count" in walls else "geometry"
    near = [s["best_rmsd"] for s in per_slot if s.get("best_rmsd")]
    return {
        "wall": wall,
        "n_slot_types": len(per_slot),
        "n_blocked": len(walls),
        "worst_rmsd": max(near) if near else None,
        "slots": per_slot,
    }


def analyse_one(mofgen: Autografs, path: Path, max_rmsd: float) -> dict:
    record: dict = {"outcome": None, "seconds": 0.0}
    start = time.perf_counter()
    try:
        result = mofgen.deconstruct(str(path))
    except AutografsError as error:
        record["outcome"] = "deconstruction_failed"
        record["error"] = f"{type(error).__name__}: {error}"
        record["seconds"] = time.perf_counter() - start
        return record
    nets = list(result.net_candidates or [])
    record["net"] = nets
    if not nets:
        record["outcome"] = "unidentified"
        record["seconds"] = time.perf_counter() - start
        return record
    fragments = list(result.fragments.values())
    record["n_fragments"] = len(fragments)
    verdicts = {}
    for name in nets:
        topology = mofgen.topologies.get(name)
        if topology is None:
            continue
        verdicts[name] = _net_verdict(topology, fragments, max_rmsd)
    if not verdicts:
        record["outcome"] = "no_blueprint"
        record["seconds"] = time.perf_counter() - start
        return record
    # the most favourable net decides: order is fit < geometry < arm_count
    rank = {None: 0, "geometry": 1, "arm_count": 2}
    best_net = min(verdicts, key=lambda n: rank[verdicts[n]["wall"]])
    best = verdicts[best_net]
    record["outcome"] = "mapped" if best["wall"] is None else best["wall"]
    record["best_net"] = best_net
    record["n_slot_types"] = best.get("n_slot_types")
    record["n_blocked"] = best.get("n_blocked")
    record["worst_rmsd"] = best.get("worst_rmsd")
    record["slot_arms"] = [s["slot_arms"] for s in best["slots"]]
    record["unit_arms"] = sorted({len(f.arm_units) for f in fragments})
    record["seconds"] = time.perf_counter() - start
    return record


def _init_worker(topofile: str | None, max_rmsd: float) -> None:
    _WORKER["mofgen"] = Autografs(topofile=topofile) if topofile else Autografs()
    _WORKER["max_rmsd"] = max_rmsd


def _run_worker(path: str) -> tuple[str, dict]:
    name = Path(path).name
    try:
        return name, analyse_one(_WORKER["mofgen"], Path(path), _WORKER["max_rmsd"])
    except Exception as error:  # noqa: BLE001 - one bad CIF must not kill a sweep
        return name, {
            "outcome": "driver_error",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=3),
        }


def run(
    corpus: list[Path],
    *,
    max_rmsd: float = COMPATIBILITY_MAX_RMSD,
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
            with ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=_init_worker,
                initargs=(topofile, max_rmsd),
            ) as pool:
                for index, (name, record) in enumerate(
                    pool.map(_run_worker, [str(p) for p in todo], chunksize=4), 1
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
        "benchmark": "mapping_gap",
        "max_rmsd": max_rmsd,
        "n_structures": len(records),
        "outcomes": dict(Counter(r["outcome"] for r in records.values())),
        "structures": records,
    }


def _report(index: int, total: int, name: str, record: dict, handle) -> None:
    extra = ""
    if record.get("worst_rmsd"):
        extra = f"  (nearest fit at rmsd {record['worst_rmsd']})"
    print(f"[{index}/{total}] {name}: {record['outcome']}{extra}", flush=True)
    if handle:
        handle.write(json.dumps({"name": name, "record": record}, default=str) + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", help="directory, glob, manifest, or single CIF")
    parser.add_argument("-o", "--output", default="mapping-gap.json")
    parser.add_argument(
        "--only",
        default=None,
        help="JSON round-trip output; restrict to its no_mapping structures",
    )
    parser.add_argument("--max-rmsd", type=float, default=COMPATIBILITY_MAX_RMSD)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--topofile", default=None)
    args = parser.parse_args()

    corpus = _collect(args.corpus)
    if args.only:
        payload = json.loads(Path(args.only).read_text(encoding="utf-8"))
        wanted = {
            name
            for name, record in payload["structures"].items()
            if record.get("outcome") == "no_mapping"
        }
        corpus = [p for p in corpus if p.name in wanted]
        print(f"restricted to {len(corpus)} no_mapping structures")
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
