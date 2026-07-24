"""Compare net identification against a file of reference labels.

Runs deconstruction-based net identification over a corpus and checks
the candidates against reference topology assignments, reporting
agreement split by matching tier.

The labels file is JSON: ``{"structure.cif": "pcu", ...}`` — values may
also be lists when the reference itself is ambiguous. A structure
agrees when the reference label (any of them) is among our candidates.

Outcomes per structure:
- ``agree`` / ``disagree``     identified, label present
- ``unidentified``             deconstructed, no library net matched
- ``unlabelled``               no reference label for this structure
- ``deconstruction_failed``    with the error message

Usage:
    python scripts/benchmarks/netid.py corpus_dir --labels labels.json -o netid.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from autografs import Autografs
from autografs.exceptions import AutografsError


def identify_one(mofgen: Autografs, source, labels: dict) -> dict:
    """Identify one structure and score it against its label."""
    record: dict = {"outcome": None, "net": None, "tier": None, "error": None}
    name = Path(str(source)).name
    reference = labels.get(name)
    if isinstance(reference, str):
        reference = [reference]
    t0 = time.perf_counter()
    try:
        result = mofgen.deconstruct(source)
    except (AutografsError, ValueError, KeyError, IndexError) as exc:
        record["outcome"] = "deconstruction_failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = time.perf_counter() - t0
        return record
    record["seconds"] = time.perf_counter() - t0
    record["net"] = list(result.net_candidates)
    if result.net_candidates:
        record["tier"] = result.subframework_nets[0].tier
    record["fold"] = result.n_periodic_components
    if reference is None:
        record["outcome"] = "unlabelled"
    elif not result.net_candidates:
        record["outcome"] = "unidentified"
        record["label"] = reference
    else:
        record["label"] = reference
        agrees = bool(set(reference) & set(result.net_candidates))
        record["outcome"] = "agree" if agrees else "disagree"
    return record


def run(corpus: list[Path], mofgen: Autografs, labels: dict) -> dict:
    """Score the whole corpus; returns the results payload."""
    records = {}
    for path in sorted(corpus):
        records[path.name] = identify_one(mofgen, str(path), labels)
        print(f"  {path.name:<40} {records[path.name]['outcome']}")
    outcomes = Counter(record["outcome"] for record in records.values())
    by_tier: Counter[str] = Counter()
    for record in records.values():
        if record["outcome"] in ("agree", "disagree"):
            by_tier[f"{record['outcome']}_{record['tier']}"] += 1
    scored = outcomes["agree"] + outcomes["disagree"]
    return {
        "benchmark": "netid",
        "n_structures": len(records),
        "outcomes": dict(sorted(outcomes.items())),
        "agreement_by_tier": dict(sorted(by_tier.items())),
        "agreement_rate": (outcomes["agree"] / scored) if scored else None,
        "structures": records,
    }


def _collect(spec: str) -> list[Path]:
    path = Path(spec)
    if path.is_dir():
        return sorted(path.glob("*.cif"))
    if any(char in spec for char in "*?["):
        return sorted(Path(spec).parent.glob(Path(spec).name))
    return [path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", help="directory, glob, or single CIF")
    parser.add_argument("--labels", required=True, help="JSON name -> net(s)")
    parser.add_argument("-o", "--output", default="netid.json")
    parser.add_argument("--topofile", default=None, help="topology library override")
    args = parser.parse_args()

    corpus = _collect(args.corpus)
    if not corpus:
        raise SystemExit(f"no structures matched {args.corpus!r}")
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    mofgen = Autografs(topofile=args.topofile)
    payload = run(corpus, mofgen, labels)
    Path(args.output).write_text(json.dumps(payload, indent=1, default=str))
    print(f"\n{payload['n_structures']} structures -> {args.output}")
    rate = payload["agreement_rate"]
    print(f"  agreement: {f'{rate:.1%}' if rate is not None else 'n/a'}")
    for outcome, count in payload["outcomes"].items():
        print(f"  {outcome:<24} {count}")


if __name__ == "__main__":
    main()
