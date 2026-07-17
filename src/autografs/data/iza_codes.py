"""
The IZA framework type codes (fetched 2026-07-16, 274 codes).

Official three-letter codes assigned by the IZA Structure Commission;
a ``-`` prefix marks interrupted frameworks, a ``*`` prefix marks
codes defined from a disordered material's representative polymorph.
The per-code idealized CIF on iza-structure.org uses the code with
both prefixes stripped as its filename.

Used by ``autografs-topologies --use_iza`` (local fetch; nothing
IZA-derived ships with the package) and by
``scripts/make_iza_aliases.py``.
"""

# fmt: off
IZA_CODES: tuple[str, ...] = (
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
)
# fmt: on


def cif_filename(code: str) -> str:
    """The iza-structure.org CIF filename for a framework code."""
    return f"{code.lstrip('*-')}.cif"
