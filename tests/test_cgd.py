"""Tests for CGD parsing robustness (malformed and adversarial entries)."""

import pytest

from autografs.cgd import build_group_lookup, read_cgd_data, topology_from_string

# 3D group but no CELL line: must fail as ValueError (counted as a
# parse error), not UnboundLocalError (which crashed the whole run)
MISSING_CELL_CGD = """CRYSTAL
  NAME nocell
  GROUP Pm-3m
  NODE 1 6  0.00000 0.00000 0.00000
  EDGE  0.00000 0.00000 0.00000   0.50000 0.00000 0.00000
END
"""

# CELL but no GROUP line at all
MISSING_GROUP_CGD = """CRYSTAL
  NAME nogroup
  CELL 1.00000 1.00000 1.00000 90.0000 90.0000 90.0000
  NODE 1 6  0.00000 0.00000 0.00000
  EDGE  0.00000 0.00000 0.00000   0.50000 0.00000 0.00000
END
"""

# no NODE or EDGE lines: np.stack would raise a confusing error
EMPTY_SITES_CGD = """CRYSTAL
  NAME nosites
  GROUP p4mm
  CELL 1.00000 1.00000 90.0000
END
"""

# the entry's last content line ends with letters from {C,R,Y,S,T,A,L}:
# a str.strip("CRYSTAL") split silently ate them (METAL -> ME)
TRAILING_LETTERS_CGD = """CRYSTAL
  GROUP p4mm
  CELL 1.00000 1.00000 90.0000
  NODE 1 4  0.00000 0.00000
  EDGE  0.00000 0.00000   0.00000 1.00000
  EDGE  0.00000 0.00000   1.00000 0.00000
  NAME METAL
END
"""


def _entry(cgd: str) -> str:
    """One entry as read_cgd_data hands it to topology_from_string."""
    return cgd.split("END")[0].strip().removeprefix("CRYSTAL")


class TestMalformedEntries:
    def test_missing_cell_raises_value_error(self):
        with pytest.raises(ValueError, match="CELL"):
            topology_from_string(_entry(MISSING_CELL_CGD), build_group_lookup())

    def test_missing_group_raises_value_error(self):
        with pytest.raises(ValueError, match="GROUP"):
            topology_from_string(_entry(MISSING_GROUP_CGD), build_group_lookup())

    def test_missing_sites_raises_value_error(self):
        with pytest.raises(ValueError, match="NODE or EDGE"):
            topology_from_string(_entry(EMPTY_SITES_CGD), build_group_lookup())

    def test_malformed_entries_are_counted_not_fatal(self):
        """One bad entry must not abort the conversion of the rest."""
        topologies = read_cgd_data(
            MISSING_CELL_CGD + MISSING_GROUP_CGD + EMPTY_SITES_CGD
        )
        assert topologies == {}


class TestEntrySplitting:
    def test_trailing_crystal_letters_survive(self):
        """Entry text ending in C/R/Y/S/T/A/L letters is not eaten."""
        topologies = read_cgd_data(TRAILING_LETTERS_CGD)
        assert set(topologies) == {"METAL"}
