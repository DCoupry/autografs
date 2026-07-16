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

from autografs.topology_io import load_topologies

# fmt: off
IZA_CODES = [
    "ABW", "ACO", "AEI", "AEL", "AEN", "AET", "AFG", "AFI", "AFN", "AFO",
    "AFR", "AFS", "AFT", "AFV", "AFX", "AFY", "AHT", "ANA", "ANO", "APC",
    "APD", "AST", "ASV", "ATN", "ATO", "ATS", "ATT", "ATV", "AVE", "AVL",
    "AWO", "AWW", "BCT", "BEC", "BIK", "BOF", "BOG", "BOZ", "BPH", "BRE",
    "BSV", "CAN", "CAS", "CDO", "CFI", "CGF", "CGS", "CHA", "-CHI", "-CLO",
    "CON", "CSV", "CZP", "DAC", "DDR", "DFO", "DFT", "DOH", "DON", "EAB",
    "EDI", "EEI", "EMT", "EON", "EOS", "EPI", "ERI", "-ERS", "ESV", "ETL",
    "ETR", "ETV", "EUO", "EWF", "EWO", "EWS", "-EWT", "EZT", "FAR", "FAU",
    "FER", "FRA", "GIS", "GIU", "GME", "GON", "GOO", "HEI", "HEU", "-HOS",
    "HZF", "IFO", "IFR", "-IFT", "-IFU", "IFW", "IFY", "IHW", "IMF", "-ION",
    "IRN", "IRR", "-IRT", "-IRY", "ISV", "ITE", "ITG", "ITH", "ITR", "ITT",
    "-ITV", "ITW", "-IVY", "IWR", "IWS", "IWV", "IWW", "JBW", "JNT", "JOZ",
    "JRY", "JSN", "JSR", "JST", "JSW", "JSY", "JZO", "JZT", "KFI", "KLW",
    "LAU", "LEV", "LIO", "-LIT", "LOS", "LOV", "LTA", "LTF", "LTJ", "LTL",
    "LTN", "MAN", "MAR", "MAZ", "MEI", "MEL", "MEP", "MER", "MFI", "MFS",
    "MON", "MOR", "MOZ", "MRT", "MSE", "MSO", "MTF", "MTN", "MTT", "MTW",
    "MVY", "MWF", "MWW", "NAB", "NAT", "NES", "NJO", "NJW", "NON", "NPO",
    "NPT", "NSI", "OBW", "OFF", "OKO", "OSI", "OSO", "OWE", "-PAR", "PAU",
    "PCR", "PHI", "PON", "POR", "POS", "PSI", "PTF", "PTO", "PTT", "PTY",
    "PUN", "PWN", "PWO", "PWW", "RFE", "RHO", "-RON", "RRO", "RSN", "RTE",
    "RTH", "RUT", "RWR", "RWY", "SAF", "SAO", "SAS", "SAT", "SAV", "SBE",
    "SBN", "SBS", "SBT", "SEW", "SFE", "SFF", "SFG", "SFH", "SFN", "SFO",
    "SFS", "SFW", "SGT", "SIV", "SOD", "SOF", "SOR", "SOS", "SOV", "SSF",
    "-SSO", "SSY", "STF", "STI", "STT", "STW", "-SVR", "SVV", "SWY", "-SYT",
    "SZR", "TER", "THO", "TOL", "TON", "TSC", "TUN", "UEI", "UFI", "UOS",
    "UOV", "UOZ", "USI", "UTL", "UWY", "VET", "VFI", "VNI", "VSV", "WEI",
    "-WEN", "YFI", "YUG", "ZJN", "ZON", "*BEA", "*CTH", "*-ITN", "*MRE",
    "*PCS", "*SFV", "*STO", "*-SVY", "*UOE",
]
# fmt: on


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
