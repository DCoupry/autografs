"""Round-trip closure benchmark: deconstruct -> rebuild -> verify.

For every structure in a corpus, deconstruct it, then try to rebuild it
from its own extracted fragments on every identified net candidate,
gating the rebuild with the exact net-verification gate. The per-
structure outcome taxonomy is the point - failures are data:

- ``closed``            rebuild passed verify_net on some candidate
- ``rebuild_failed``    every candidate/mapping combination was gated
- ``no_mapping``        no compatible fragment-to-slot assignment
- ``rod``               identified, but contains 1-periodic units the
                        forward pipeline cannot rebuild (Stage A)
- ``unidentified``      deconstructed, but no library net matched
- ``deconstruction_failed``  with the error message

Deterministic (no sampling, sorted corpus order); machine-readable
JSON out plus a summary table on stdout.

Usage:
    python scripts/benchmarks/roundtrip.py "corpus/*.cif" -o results.json
    python scripts/benchmarks/roundtrip.py corpus_dir -o results.json --max-rmsd 0.5
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import Counter
from pathlib import Path

from autografs import Autografs
from autografs.exceptions import AutografsError

# fragment-to-slot assignments tried per net candidate before giving
# up; multi-orbit nets with many interchangeable fragments explode
# combinatorially and a closure benchmark must terminate
MAX_MAPPINGS_PER_NET = 16


def candidate_mappings(topology, fragments: dict):
    """Yield fragment-per-slot-type assignments compatible by geometry."""
    slot_types = list(topology.mappings)
    options = []
    for slot_type in slot_types:
        fitting = [
            fragment
            for fragment in fragments.values()
            if fragment.has_compatible_symmetry(slot_type)
        ]
        if not fitting:
            return
        options.append(fitting)
    combos = itertools.product(*options)
    for combo in itertools.islice(combos, MAX_MAPPINGS_PER_NET):
        yield dict(zip(slot_types, combo, strict=True))


def roundtrip_one(mofgen: Autografs, source, max_rmsd: float) -> dict:
    """Deconstruct one structure and attempt the verified rebuild."""
    record: dict = {"outcome": None, "net": None, "tier": None, "error": None}
    t0 = time.perf_counter()
    try:
        result = mofgen.deconstruct(source)
    except (AutografsError, ValueError, KeyError, IndexError) as exc:
        record["outcome"] = "deconstruction_failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = time.perf_counter() - t0
        return record
    record["n_fragments"] = len(result.fragments)
    record["fold"] = result.n_periodic_components
    record["guests"] = result.guest_formulas
    if result.net_candidates:
        record["net"] = result.net_candidates
        record["tier"] = result.subframework_nets[0].tier
    if result.rod_units:
        # identified (often correctly) but not rebuildable: the forward
        # pipeline has no rod support yet (#91 Stage C)
        record["outcome"] = "rod" if result.net_candidates else "unidentified"
        record["seconds"] = time.perf_counter() - t0
        return record
    if not result.net_candidates:
        record["outcome"] = "unidentified"
        record["seconds"] = time.perf_counter() - t0
        return record
    saw_mapping = False
    for net in result.net_candidates:
        topology = mofgen.topologies[net]
        for mappings in candidate_mappings(topology, result.fragments):
            saw_mapping = True
            try:
                mofgen.build(
                    topology,
                    mappings=mappings,
                    max_rmsd=max_rmsd,
                    verify_net=True,
                )
            except AutografsError:
                continue
            record["outcome"] = "closed"
            record["rebuilt_net"] = net
            record["seconds"] = time.perf_counter() - t0
            return record
    record["outcome"] = "rebuild_failed" if saw_mapping else "no_mapping"
    record["seconds"] = time.perf_counter() - t0
    return record


def run(corpus: list[Path], mofgen: Autografs, max_rmsd: float) -> dict:
    """Round-trip every structure; returns the results payload."""
    records = {}
    for path in sorted(corpus):
        records[path.name] = roundtrip_one(mofgen, str(path), max_rmsd)
        outcome = records[path.name]["outcome"]
        print(f"  {path.name:<40} {outcome}")
    outcomes = Counter(record["outcome"] for record in records.values())
    return {
        "benchmark": "roundtrip",
        "max_rmsd": max_rmsd,
        "n_structures": len(records),
        "outcomes": dict(sorted(outcomes.items())),
        "structures": records,
    }


def _collect(spec: str) -> list[Path]:
    path = Path(spec)
    if path.is_dir():
        return sorted(path.glob("*.cif"))
    if any(char in spec for char in "*?["):
        base = Path(spec).parent
        return sorted(base.glob(Path(spec).name))
    return [path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", help="directory, glob, or single CIF")
    parser.add_argument("-o", "--output", default="roundtrip.json")
    parser.add_argument("--max-rmsd", type=float, default=0.5)
    parser.add_argument("--topofile", default=None, help="topology library override")
    args = parser.parse_args()

    corpus = _collect(args.corpus)
    if not corpus:
        raise SystemExit(f"no structures matched {args.corpus!r}")
    mofgen = Autografs(topofile=args.topofile)
    payload = run(corpus, mofgen, args.max_rmsd)
    Path(args.output).write_text(json.dumps(payload, indent=1, default=str))
    print(f"\n{payload['n_structures']} structures -> {args.output}")
    for outcome, count in payload["outcomes"].items():
        print(f"  {outcome:<24} {count}")


if __name__ == "__main__":
    main()
