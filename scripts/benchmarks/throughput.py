"""Build and identification throughput benchmark.

For each requested topology: build it once from the first compatible
SBU combination in the library (timed over repeats, median reported),
then time net identification of the built framework. Deterministic:
first-in-sorted-order SBUs, no sampling.

Usage:
    python scripts/benchmarks/throughput.py --topologies pcu,dia,srs -o throughput.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from autografs import Autografs
from autografs.exceptions import AutografsError
from autografs.net import framework_quotient_edges, identify_net


def time_one(mofgen: Autografs, name: str, repeats: int) -> dict:
    """Median build and identify wall times for one topology."""
    record: dict = {"error": None}
    if name not in mofgen.topologies:
        record["error"] = "unknown topology"
        return record
    topology = mofgen.topologies[name]
    available = mofgen.list_building_units(sieve=name, verbose=False)
    if len(available) != len(topology.mappings) or not all(available.values()):
        record["error"] = "no compatible SBU combination"
        return record
    mappings = {
        slot_type: sorted(options)[0] for slot_type, options in available.items()
    }
    record["sbus"] = sorted(set(mappings.values()))
    try:
        build_times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            framework = mofgen.build(topology, mappings=dict(mappings))
            build_times.append(time.perf_counter() - t0)
        identify_times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            candidates = identify_net(
                framework_quotient_edges(framework), mofgen.topologies
            )
            identify_times.append(time.perf_counter() - t0)
    except AutografsError as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    record["n_atoms"] = len(framework)
    record["build_seconds"] = statistics.median(build_times)
    record["identify_seconds"] = statistics.median(identify_times)
    record["identified_as"] = list(candidates)
    return record


def run(mofgen: Autografs, topologies: list[str], repeats: int) -> dict:
    """Time every requested topology; returns the results payload."""
    records = {}
    for name in topologies:
        records[name] = time_one(mofgen, name, repeats)
        timing = records[name].get("build_seconds")
        note = f"{timing:.2f}s" if timing is not None else records[name]["error"]
        print(f"  {name:<8} {note}")
    return {
        "benchmark": "throughput",
        "repeats": repeats,
        "topologies": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--topologies", required=True, help="comma-separated topology names"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("-o", "--output", default="throughput.json")
    parser.add_argument("--topofile", default=None, help="topology library override")
    args = parser.parse_args()

    names = [name.strip() for name in args.topologies.split(",") if name.strip()]
    mofgen = Autografs(topofile=args.topofile)
    payload = run(mofgen, names, args.repeats)
    Path(args.output).write_text(json.dumps(payload, indent=1, default=str))
    print(f"\n{len(names)} topologies -> {args.output}")


if __name__ == "__main__":
    main()
