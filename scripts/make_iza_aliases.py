"""Regenerate src/autografs/data/iza_aliases.json.

Maps IZA zeolite framework type codes onto RCSR nets already shipped in
the default topology library: RCSR deliberately reuses IZA codes in
lowercase for shared nets (FAU -> fau, LTA -> lta, ...). Only the code
mapping ships - the net data is the RCSR data already in the wheel - so
this carries no IZA licensing exposure (issue #128, phase 1).

Rules:
- interrupted frameworks (leading dash, e.g. -CLO) are skipped: they are
  not fully 4-coordinated nets and any same-named RCSR entry would not
  be the IZA framework.
- intergrowth end-members (leading asterisk, e.g. *BEA) alias under both
  the verbatim code and the starless form users actually type.
- a candidate only becomes an alias if the RCSR net's topology-bearing
  vertices (degree > 2; edge centers excluded) are all 4-coordinated -
  zeolite nets are 4-c, so this guards against accidental name
  collisions with non-zeolite RCSR codes.

The IZA code list below is the framework type table of the IZA-SC
Database of Zeolite Structures (europe.iza-structure.org/IZA-SC/
ftc_table.php), fetched 2026-07-16 (279 codes). Rerun after updating it
or after regenerating the topology library:

    python scripts/make_iza_aliases.py
"""

from __future__ import annotations

import json
from pathlib import Path

from autografs.data.iza_codes import IZA_CODES
from autografs.topology_io import load_topologies


def _is_uniform_4c(payload: dict) -> bool:
    """Topology-bearing vertices (degree > 2) are all 4-coordinated."""
    degrees = [slot["species"].count("X") for slot in payload["slots"]]
    bearing = [degree for degree in degrees if degree > 2]
    return bool(bearing) and all(degree == 4 for degree in bearing)


def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "src" / "autografs" / "data"
    library = load_topologies(data_dir / "topologies.json.gz")
    raw = dict(library.raw_items())

    aliases: dict[str, str] = {}
    skipped_not_4c: list[str] = []
    for code in IZA_CODES:
        if code.startswith("-") or code.startswith("*-"):
            continue  # interrupted framework: no proper 4-c net to alias
        stripped = code.lstrip("*")
        target = stripped.lower()
        if target not in raw:
            continue
        if not _is_uniform_4c(raw[target]):
            skipped_not_4c.append(f"{code} -> {target}")
            continue
        aliases[stripped] = target
        if code != stripped:
            aliases[code] = target  # the verbatim starred form too

    out = data_dir / "iza_aliases.json"
    out.write_text(json.dumps(aliases, indent=1, sort_keys=True) + "\n")
    print(f"{len(aliases)} aliases written to {out}")
    if skipped_not_4c:
        print("skipped (same-named RCSR net is not uniformly 4-c):")
        for entry in skipped_not_4c:
            print(f"  {entry}")


if __name__ == "__main__":
    main()
