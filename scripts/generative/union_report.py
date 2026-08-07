"""Combine several recombine.py arms into one generative yield.

Two selection strategies over the same nets and vocabulary produce
*disjoint* sets of usable assemblies (measured at small scale: overlap 0,
union 43 against 23 and 20 alone), so the honest yield of a generative
campaign is their union, not either arm's number.

Reports the union deduplicated on the assembly fingerprint - several
sampled mappings can land on the same (net, blocks, fold) assembly, and
that is one hypothetical material - and always splits genuine
frameworks from the degenerate ones a bare-metal pick produces.

Usage:
    python scripts/generative/union_report.py a.json b.json -o union.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recombine import is_genuine_framework  # noqa: E402


def _arm(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    built = [r for r in payload["records"] if r["outcome"] == "built"]
    usable_novel = [r for r in built if r.get("usable") and r.get("novel")]
    genuine = {r["fingerprint"]: r for r in usable_novel if is_genuine_framework(r)}
    return {
        "label": payload.get("selection") or path.stem,
        "path": str(path),
        "attempted": payload["summary"]["attempted"],
        "built": len(built),
        "closed": sum(1 for r in built if r.get("closed")),
        "clash_free": sum(1 for r in built if r.get("clash_free")),
        "usable": sum(1 for r in built if r.get("usable")),
        "genuine": genuine,
        "degenerate": len(usable_novel) - len(genuine),
        "median_contact": statistics.median([r["min_contact"] for r in built])
        if built
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", help="recombine.py output JSON files")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    arms = [_arm(Path(p)) for p in args.reports]
    header = (
        f"{'arm':<12}{'built':>7}{'closed':>8}{'clash-free':>12}"
        f"{'usable':>8}{'GENUINE':>9}{'nets':>6}{'degen':>7}{'contact':>9}"
    )
    print(header)
    for arm in arms:
        nets = len({r["net"] for r in arm["genuine"].values()})
        contact = f"{arm['median_contact']:.3f}" if arm["median_contact"] else "-"
        print(
            f"{arm['label']:<12}{arm['built']:>7}{arm['closed']:>8}"
            f"{arm['clash_free']:>12}{arm['usable']:>8}{len(arm['genuine']):>9}"
            f"{nets:>6}{arm['degenerate']:>7}{contact:>9}"
        )

    union: dict[str, dict] = {}
    for arm in arms:
        for fingerprint, record in arm["genuine"].items():
            union.setdefault(fingerprint, {**record, "_arm": arm["label"]})
    print()
    print(f"UNION of genuine usable novel assemblies : {len(union)}")
    print(
        f"  distinct nets                          : {len({r['net'] for r in union.values()})}"
    )
    for a in arms:
        for b in arms:
            if a is b:
                continue
            shared = set(a["genuine"]) & set(b["genuine"])
            print(f"  overlap {a['label']} n {b['label']}: {len(shared)}")
            break
    print(f"  contributed by arm: {dict(Counter(r['_arm'] for r in union.values()))}")

    sizes = sorted(r["n_atoms"] for r in union.values())
    if sizes:
        print(
            f"  atoms: min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}"
        )
    by_free = Counter(r.get("n_free") for r in union.values())
    print(
        f"  by blueprint freedom: {dict(sorted(by_free.items(), key=lambda kv: (kv[0] is None, kv[0])))}"
    )

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "arms": [
                        {k: v for k, v in arm.items() if k != "genuine"}
                        | {"n_genuine": len(arm["genuine"])}
                        for arm in arms
                    ],
                    "union_size": len(union),
                    "union": list(union.values()),
                },
                indent=1,
                default=str,
            )
        )
        print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
