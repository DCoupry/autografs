#!/usr/bin/env python3
"""
Generate the committed topology test fixture (tests/data/topologies_fixture.json).

The CGD entries below are verbatim from RCSRnets-2019-06-01.cgd (RCSR,
http://rcsr.anu.edu.au), including the '# EDGE_CENTER' lines: the CGD
parser strips the first two characters of every line, which deliberately
uncomments them - edge centers are what create the 2-connected slots
that linear SBUs occupy.

Run from the repository root:

    python scripts/make_test_fixture.py
"""

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from autografs import topology_io  # noqa: E402
from autografs.cgd import read_cgd_data  # noqa: E402

logging.basicConfig(level=logging.INFO)

# pcu: uninodal 6-c (alpha-Po, cubic); srs: uninodal 3-c (SrSi2,
# cubic); dia: uninodal 4-c (diamond, cubic); acs: uninodal 6-c
# trigonal-prismatic (hexagonal - exercises the 2-parameter cell
# path). All with edge-center 2-c slots.
FIXTURE_CGD = """CRYSTAL
  NAME pcu
  GROUP Pm-3m
  CELL 1.00000 1.00000 1.00000 90.0000 90.0000 90.0000
  NODE 1 6  0.00000 0.00000 0.00000
  EDGE  0.00000 0.00000 0.00000   0.00000 0.00000 1.00000
# EDGE_CENTER  0.00000 0.00000 0.50000
END
CRYSTAL
  NAME srs
  GROUP I4132
  CELL 2.82843 2.82843 2.82843 90.0000 90.0000 90.0000
  NODE 1 3  0.12500 0.12500 0.12500
  EDGE  0.12500 0.12500 0.12500   0.12500 -0.12500 0.37500
# EDGE_CENTER  0.12500 0.00000 0.25000
END
CRYSTAL
  NAME dia
  GROUP Fd-3m:2
  CELL 2.30940 2.30940 2.30940 90.0000 90.0000 90.0000
  NODE 1 4  0.12500 0.12500 0.62500
  EDGE  0.12500 0.12500 0.62500   0.37500 0.37500 0.37500
# EDGE_CENTER  0.25000 0.25000 0.50000
END
CRYSTAL
  NAME acs
  GROUP P63/mmc
  CELL 1.41420 1.41420 1.15471 90.0000 90.0000 120.0000
  NODE 1 6  0.66667 0.33333 0.25000
  EDGE  0.66667 0.33333 0.25000   0.33333 -0.33333 -0.25000
# EDGE_CENTER  0.50000 0.00000 0.00000
END
"""


def main() -> None:
    topologies = read_cgd_data(FIXTURE_CGD)
    expected = {"pcu", "srs", "dia", "acs"}
    missing = expected - topologies.keys()
    if missing:
        raise SystemExit(f"Fixture generation failed for: {sorted(missing)}")
    for name, topology in sorted(topologies.items()):
        types = {
            f"{key.pointgroup} {len(key.atoms.indices_from_symbol('X'))}-c": len(
                indices
            )
            for key, indices in topology.mappings.items()
        }
        print(f"{name}: {len(topology)} slots, sg {topology.spacegroup_number}")
        print(f"   slot types: {types}")
    out_path = REPO_ROOT / "tests" / "data" / "topologies_fixture.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    topology_io.save_topologies(topologies, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
