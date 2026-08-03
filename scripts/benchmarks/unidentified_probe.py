"""What the unidentified structures look like, since the census cannot say.

``roundtrip.py`` reports ``unidentified`` for a structure that
deconstructs and then matches no library net, and records nothing else
-- so the largest method-side bucket of the census is also its least
attributed. Cross-referencing the rod scan removes the part with a known
cause (1-periodic units must match on the contracted points-of-extension
tier); this driver re-deconstructs the remainder and records what the
identifier actually saw, so the bucket can be attributed instead of
pooled:

- the quotient graph's size and degree structure, contracted and not
  (a vertex of degree 8+ in the contracted graph is the signature of
  under-cutting or multi-bridged units, the same granularity artifact
  the mapping-gap sweep isolates on the identified population);
- whether ANY library net shares the reduced degree recipe, which
  separates "the degree profile itself is unmatched" (a hard coverage
  statement) from "candidates existed and every coordination sequence
  disagreed";
- per-subframework matches and fold, so interpenetration-consensus
  failures are visible.

Usage:
    python scripts/benchmarks/unidentified_probe.py CORPUS \
        --only e1.json --rod-scan e6.json -o unidentified-probe.json
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

_WORKER: dict = {}


def probe_one(mofgen: Autografs, path: Path) -> dict:
    # the prefilter's own helpers, so "no net shares this degree recipe"
    # is a statement about the shipped identifier and not a model of it
    from autografs.net import (  # noqa: PLC0415
        _degree_profile,
        contract_quotient_edges,
        net_signature,
    )

    record: dict = {"outcome": None, "seconds": 0.0}
    start = time.perf_counter()
    try:
        result = mofgen.deconstruct(str(path))
    except AutografsError as error:
        record["outcome"] = "deconstruction_failed"
        record["error"] = f"{type(error).__name__}: {error}"
        record["seconds"] = time.perf_counter() - start
        return record
    edges = result.quotient_edges
    degree: Counter = Counter()
    n_edges = 0
    for edge, count in edges.items():
        degree[edge[0]] += count
        degree[edge[1]] += count
        n_edges += count
    record["outcome"] = "identified" if result.net_candidates else "unidentified"
    record["net"] = list(result.net_candidates or [])
    record["n_units"] = len(result.units)
    record["n_fragment_types"] = len(result.fragments)
    record["kinds"] = dict(Counter(unit.kind for unit in result.units))
    record["fold"] = result.n_periodic_components
    record["rod_units"] = len(result.rod_units)
    record["n_vertices"] = len(degree)
    record["n_edges"] = n_edges
    record["sub_matches"] = [sorted(m) for m in result.subframework_nets]
    signature = net_signature(edges, contract=True)
    record["sig_empty"] = not signature
    if signature:
        profile = _degree_profile(
            [s[0] for s, multiplicity in signature for _ in range(multiplicity)]
        )
        record["profile"] = list(profile)
        record["n_prefilter_candidates"] = sum(
            1
            for _name, payload in mofgen.topologies.raw_items()
            if _degree_profile(
                [slot["species"].count("X") for slot in payload.get("slots", [])]
            )
            == profile
        )
    contracted = contract_quotient_edges(edges, contract=True)
    cdegree: Counter = Counter()
    for edge, count in contracted.items():
        cdegree[edge[0]] += count
        cdegree[edge[1]] += count
    record["contracted_vertices"] = len(cdegree)
    record["contracted_max_degree"] = max(cdegree.values(), default=0)
    record["seconds"] = time.perf_counter() - start
    return record


def _init_worker(topofile: str | None) -> None:
    _WORKER["mofgen"] = Autografs(topofile=topofile) if topofile else Autografs()


def _run_worker(path: str) -> tuple[str, dict]:
    name = Path(path).name
    try:
        return name, probe_one(_WORKER["mofgen"], Path(path))
    except Exception as error:  # noqa: BLE001 - one bad CIF must not kill a sweep
        return name, {
            "outcome": "driver_error",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=3),
        }


def run(
    corpus: list[Path],
    *,
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
                initargs=(topofile,),
            ) as pool:
                for index, (name, record) in enumerate(
                    pool.map(_run_worker, [str(p) for p in todo], chunksize=4), 1
                ):
                    records[name] = record
                    _report(index, len(todo), name, record, handle)
        else:
            _init_worker(topofile)
            for index, path in enumerate(todo, 1):
                name, record = _run_worker(str(path))
                records[name] = record
                _report(index, len(todo), name, record, handle)
    finally:
        if handle:
            handle.close()
    return {
        "benchmark": "unidentified_probe",
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
    parser.add_argument("-o", "--output", default="unidentified-probe.json")
    parser.add_argument(
        "--only",
        default=None,
        help="JSON round-trip output; restrict to its unidentified structures",
    )
    parser.add_argument(
        "--rod-scan",
        default=None,
        help="JSON rod round-trip output; drop structures it found rods in "
        "(their identification failure has a known, separate cause)",
    )
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
            if record.get("outcome") == "unidentified"
        }
        corpus = [p for p in corpus if p.name in wanted]
        print(f"restricted to {len(corpus)} unidentified structures")
    if args.rod_scan:
        payload = json.loads(Path(args.rod_scan).read_text(encoding="utf-8"))
        rods = {
            name
            for name, record in payload["structures"].items()
            if (record.get("n_rods") or 0) > 0
        }
        before = len(corpus)
        corpus = [p for p in corpus if p.name not in rods]
        print(f"dropped {before - len(corpus)} rod-containing structures")
    if args.limit is not None:
        corpus = corpus[: args.limit]
    if not corpus:
        raise SystemExit(f"no structures matched {args.corpus!r}")
    n_jobs = os.cpu_count() or 1 if args.n_jobs == -1 else args.n_jobs
    payload = run(
        corpus,
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
