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


EPINET_DIALECT = """PERIODIC_GRAPH
ID sqctest
EDGES
  1 1 1 0 0
END

CRYSTAL
  ID sqctest_relaxed
  GROUP Pm-3m
  CELL 1.1 1.1 1.1 90.0 90.0 90.0
  ATOM\t1 6 0.0 0.0 0.0
  EDGE\t0.0 0.0 0.0 1.0 0.0 0.0
END

CRYSTAL
  ID sqctest_maximal
  GROUP Pm-3m
  CELL 1.0 1.0 1.0 90.0 90.0 90.0
  ATOM\t1 6 0.0 0.0 0.0
  EDGE\t0.0 0.0 0.0 1.0 0.0 0.0
\t0.0 0.0 0.0 0.0 1.0 0.0
\t0.0 0.0 0.0 0.0 0.0 1.0
END
"""


class TestEpinetDialect:
    """The EPINET .cgd dialect adapter (coverage plan stage 2).

    The fixture is a SYNTHETIC file - pcu written in EPINET's format
    conventions (PERIODIC_GRAPH preamble, several CRYSTAL blocks,
    ID/ATOM keywords, tab-led continuation rows). No EPINET data enters
    this repository: its CC BY-NC-ND license does not permit it.
    """

    def test_maximal_block_is_chosen_and_translated(self):
        from autografs.cgd import epinet_entries

        entry = epinet_entries(EPINET_DIALECT)
        assert "NAME sqctest" in entry
        assert "GROUP Pm-3m" in entry
        assert "CELL 1.0 1.0 1.0 90.0 90.0 90.0" in entry
        assert entry.count("NODE") == 1
        # the continuation rows became explicit EDGE lines, and every
        # edge gained the EDGE_CENTER the importer's slot stitching
        # needs (EPINET files carry none)
        assert entry.count("EDGE_CENTER") == 3
        assert entry.count("EDGE") == 6
        assert "1.1" not in entry

    def test_translated_entry_parses_and_identifies(self):
        import os

        from autografs.cgd import epinet_entries, read_cgd_data
        from autografs.net import identify_net, topology_quotient_edges
        from autografs.topology_io import load_topologies

        topologies = read_cgd_data(epinet_entries(EPINET_DIALECT))
        assert set(topologies) == {"sqctest"}
        fixture = os.path.join(
            os.path.dirname(__file__), "data", "topologies_fixture.json"
        )
        library = load_topologies(fixture)
        edges = topology_quotient_edges(topologies["sqctest"])
        assert identify_net(edges, library) == ["pcu"]

    def test_empty_input_is_empty_not_fatal(self):
        from autografs.cgd import epinet_entries

        assert epinet_entries("PERIODIC_GRAPH\nID x\nEND\n") == ""
