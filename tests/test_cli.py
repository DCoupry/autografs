"""
Tests for the interactive CLI's pure helpers.

The prompt flows themselves are exercised manually (they need a real
terminal); everything the prompts rely on - labels, sorting, the
topology metadata index, filename generation - is tested here against
the committed fixture library.
"""

import os

import pytest

from autografs.cli import (
    TopoInfo,
    build_topology_index,
    connectivity,
    default_output_name,
    main,
    slot_label,
    slot_labels,
    sorted_slot_types,
)

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "topologies_fixture.json"
)


@pytest.fixture(scope="module")
def mofgen():
    """Autografs with the default SBU library and fixture topologies."""
    from autografs import Autografs

    return Autografs(topofile=FIXTURE_PATH)


class TestSlotLabel:
    def test_trigonal(self, trigonal_fragment):
        assert connectivity(trigonal_fragment) == 3
        assert slot_label(trigonal_fragment, 2) == "3-connected, D3h - 2 slots"

    def test_singular_slot(self, tetrahedral_fragment):
        assert slot_label(tetrahedral_fragment, 1) == "4-connected, Td - 1 slot"

    def test_orbit_suffix(self, trigonal_fragment):
        trigonal_fragment.equivalence_class = 2
        label = slot_label(trigonal_fragment, 3, show_orbit=True)
        assert label == "3-connected, D3h - 3 slots (orbit 2)"

    def test_orbit_suffix_needs_a_class(self, trigonal_fragment):
        # SBUs have no orbit id; show_orbit must not crash on them
        assert "orbit" not in slot_label(trigonal_fragment, 1, show_orbit=True)


class TestDefaultOutputName:
    def test_cof1(self):
        name = default_output_name("hcb", ["Boroxine_triangle", "Benzene_linear"])
        assert name == "hcb_Boroxine_triangle_Benzene_linear.cif"

    def test_sanitizes_hostile_characters(self):
        name = default_output_name("net one", ["a/b", "c:d"])
        assert name == "net_one_a_b_c_d.cif"

    def test_deduplicates_repeated_sbus(self):
        name = default_output_name("pcu", ["Benzene", "Benzene"])
        assert name == "pcu_Benzene.cif"


class TestTopologyIndex:
    def test_lazy_library_scan(self, mofgen):
        index = build_topology_index(mofgen.topologies)
        by_name = {info.name: info for info in index}
        assert sorted(by_name) == ["acs", "dia", "hcb", "pcu", "sql", "srs"]
        assert by_name["hcb"].is_2d
        assert by_name["sql"].is_2d
        assert not by_name["pcu"].is_2d
        assert by_name["hcb"].connectivities == (2, 3)
        assert by_name["pcu"].connectivities == (2, 6)

    def test_lazy_scan_materializes_nothing(self, mofgen):
        build_topology_index(mofgen.topologies)
        assert not mofgen.topologies._cache

    def test_plain_dict_fallback_matches(self, mofgen):
        lazy = build_topology_index(mofgen.topologies)
        materialized = build_topology_index(dict(mofgen.topologies))
        assert materialized == lazy

    def test_index_entries_are_hashable_metadata(self):
        info = TopoInfo(name="pcu", is_2d=False, connectivities=(2, 6))
        assert {info} == {TopoInfo("pcu", False, (2, 6))}


class TestSlotTypeOrdering:
    def test_nodes_before_linkers(self, mofgen):
        slot_types = sorted_slot_types(mofgen.topologies["hcb"])
        assert [connectivity(f) for f in slot_types] == [3, 2]

    def test_labels_cover_every_slot_type(self, mofgen):
        topology = mofgen.topologies["hcb"]
        labels = slot_labels(topology)
        assert set(labels) == set(topology.mappings)
        n_labeled = sum(
            int(label.split(" - ")[1].split()[0]) for label in labels.values()
        )
        assert n_labeled == len(topology)


class TestEntryPoint:
    def test_help_exits_cleanly(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["--help"])
        assert excinfo.value.code == 0
        assert "MOF and COF" in capsys.readouterr().out
