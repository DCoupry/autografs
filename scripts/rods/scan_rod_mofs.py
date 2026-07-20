"""
Scan a directory of structures for rod (1-periodic) building units.

Runs ``Autografs.deconstruct`` over every CIF in a directory and reports
the ones that contain a rod building unit, with the rod's canonical
identity (formula, chemical repeat, screw order/angle, arm and bond
counts) and the identified net. Useful for finding rod-MOF fixtures and
building an acceptance set for the rod pipeline (rod Stage C / #158).

Nothing is redistributed: the tool reads a directory the user already
has (e.g. a local CoRE MOF copy) and writes a JSONL report next to
wherever they point it.

The element-palette filter is **optional and off by default** — every
structure is scanned and reported. Pass ``--elements`` to restrict the
report to rods whose atoms all fall in a chosen palette (e.g. to drop
f-block / exotic chemistries):

    # scan everything (default)
    python scripts/rods/scan_rod_mofs.py /path/to/cifs -o rods.jsonl

    # only common transition/main-group-metal MOFs, no f-block
    python scripts/rods/scan_rod_mofs.py /path/to/cifs -o rods.jsonl \\
        --elements common

    # an explicit palette
    python scripts/rods/scan_rod_mofs.py /path/to/cifs -o rods.jsonl \\
        --elements "H C N O Zn Cu Co Ni Mg"

Each output line is a JSON record; ``report`` re-reads a JSONL and
prints a ranked summary (straight/helical, in/out of the palette).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import warnings
from collections import Counter

# organic/ligand elements + common MOF metals; no f-block, noble, or
# radioactive main-group. Used only when --elements common is passed.
_ORGANIC = set("H B C N O F Si P S Cl Se Br Te I".split())
_METALS = set(
    "Li Na K Rb Cs Mg Ca Sr Ba Al Ga In Sn Pb Sb Bi Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd Hf Ta W Re Os Ir Pt Au Hg".split()
)
COMMON_PALETTE = _ORGANIC | _METALS

_MOFGEN = None


def _init_worker() -> None:
    global _MOFGEN
    sys.path.insert(0, os.path.join(os.getcwd(), "src"))
    warnings.filterwarnings("ignore")
    from autografs import Autografs

    _MOFGEN = Autografs()


def _elements(formula: str) -> set[str]:
    return set(re.findall(r"[A-Z][a-z]?", formula or ""))


def scan_one(path: str) -> dict:
    """Deconstruct one structure; report any rod units. Never raises."""
    from autografs.rods import rod_fragment

    record: dict = {"file": os.path.basename(path)}
    try:
        result = _MOFGEN.deconstruct(path)  # type: ignore[union-attr]
        record["nets"] = result.net_candidates
        record["n_periodic"] = result.n_periodic_components
        rods = []
        for unit in result.rod_units:
            entry: dict = {"n_atoms": len(unit.atom_indices)}
            try:
                fragment = rod_fragment(result.structure, unit)
                repeat = fragment.repeat
                entry.update(
                    formula=repeat.formula,
                    repeat_A=round(float(repeat.repeat_length), 3),
                    screw_order=repeat.screw_order,
                    screw_angle=round(float(repeat.screw_angle), 2),
                    straight=abs(repeat.screw_angle) <= 1.0,
                    n_arms=len(fragment.arms),
                    n_bonds=len(fragment.bonds),
                )
            except Exception as exc:  # noqa: BLE001
                entry["frag_error"] = f"{type(exc).__name__}: {exc}"
            rods.append(entry)
        record["rods"] = rods
        record["n_rods"] = len(rods)
    except Exception as exc:  # noqa: BLE001 - a scan must never abort on one file
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def scan_directory(
    directory: str, out_path: str, n_jobs: int = 1, pattern: str = "*.cif"
) -> int:
    """Scan every matching file; write one JSON record per line.

    Returns the number of structures found to contain a rod.
    """
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    n_rod = 0
    with open(out_path, "w", encoding="utf-8") as sink:
        if n_jobs > 1:
            from multiprocessing import Pool

            with Pool(n_jobs, initializer=_init_worker) as pool:
                results = pool.imap_unordered(scan_one, files, chunksize=4)
                for record in results:
                    sink.write(json.dumps(record) + "\n")
                    sink.flush()
                    n_rod += bool(record.get("n_rods"))
        else:
            _init_worker()
            for path in files:
                record = scan_one(path)
                sink.write(json.dumps(record) + "\n")
                sink.flush()
                n_rod += bool(record.get("n_rods"))
    return n_rod


def _resolve_palette(spec: str | None) -> set[str] | None:
    """None (default) = no filter; 'common' = COMMON_PALETTE; else a
    space/comma-separated element list."""
    if spec is None:
        return None
    if spec.lower() == "common":
        return COMMON_PALETTE
    return set(re.split(r"[,\s]+", spec.strip()))


def report(jsonl_path: str, palette: set[str] | None = None) -> None:
    """Print a ranked summary of a scan's rod hits.

    ``palette`` (off by default) restricts the report to rods whose
    elements all fall inside it.
    """
    records = [
        json.loads(line) for line in open(jsonl_path, encoding="utf-8") if line.strip()
    ]
    errors = sum(1 for r in records if "error" in r)
    with_rods = [r for r in records if r.get("n_rods")]

    kept, dropped = [], Counter()
    for record in with_rods:
        rod = next((x for x in record["rods"] if "formula" in x), None)
        if rod is None:
            continue
        if palette is not None:
            extra = _elements(rod["formula"]) - palette
            if extra:
                dropped.update(extra)
                continue
        kept.append((record, rod))

    scope = "all elements" if palette is None else f"palette of {len(palette)} elements"
    print(
        f"# {len(records)} scanned, {errors} errors, {len(with_rods)} with rods; "
        f"{len(kept)} reported ({scope})"
    )
    if dropped:
        print(f"# excluded by palette: {dict(dropped.most_common(12))}")

    straight = sorted(
        (kv for kv in kept if kv[1].get("straight") and kv[0].get("nets")),
        key=lambda kv: (kv[1].get("repeat_A") or 99, kv[1]["n_atoms"]),
    )
    helical = [kv for kv in kept if not kv[1].get("straight")]
    print(f"\n# straight + net-identified ({len(straight)}); top 20:")
    for record, rod in straight[:20]:
        print(
            f"  {record['file']:26s} {str(record['nets']):14s} {rod['formula']:12s} "
            f"rep={rod['repeat_A']} arms={rod['n_arms']} bonds={rod['n_bonds']}"
        )
    print(f"\n# helical ({len(helical)}); sample 12:")
    for record, rod in helical[:12]:
        print(
            f"  {record['file']:26s} {str(record['nets']):12s} {rod['formula']:12s} "
            f"screw={rod['screw_angle']} x{rod['screw_order']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="scan a directory of structures")
    scan.add_argument("directory")
    scan.add_argument("-o", "--output", required=True, help="JSONL report path")
    scan.add_argument("-j", "--jobs", type=int, default=1, help="parallel workers")
    scan.add_argument("--pattern", default="*.cif")

    rep = sub.add_parser("report", help="summarize a scan JSONL")
    rep.add_argument("jsonl")
    rep.add_argument(
        "--elements",
        default=None,
        help=(
            "OPTIONAL element-palette filter, OFF by default. 'common' for "
            "the built-in no-f-block palette, or an explicit list "
            "('H C N O Zn ...')."
        ),
    )

    args = parser.parse_args(argv)
    if args.command == "scan":
        n_rod = scan_directory(args.directory, args.output, args.jobs, args.pattern)
        print(f"wrote {args.output}: {n_rod} structures with rods", file=sys.stderr)
        return 0
    report(args.jsonl, _resolve_palette(args.elements))
    return 0


if __name__ == "__main__":
    sys.exit(main())
