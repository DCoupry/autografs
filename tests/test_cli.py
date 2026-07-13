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
    parse_indices,
    parse_multipliers,
    rotatable_slots,
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


class TestEditingHelpers:
    """Pure helpers behind the edit/export menu."""

    def test_parse_multipliers_single_value(self):
        assert parse_multipliers("2") == (2, 2, 2)

    def test_parse_multipliers_three_values(self):
        assert parse_multipliers("2, 1 3") == (2, 1, 3)

    def test_parse_multipliers_rejects_garbage(self):
        with pytest.raises(ValueError):
            parse_multipliers("2 2")
        with pytest.raises(ValueError):
            parse_multipliers("0")
        with pytest.raises(ValueError):
            parse_indices("")

    def test_parse_indices(self):
        assert parse_indices("3, 1 2, 3") == [1, 2, 3]

    def test_rotatable_slots_are_the_linkers(self, mofgen):
        topology = mofgen.topologies["pcu"]
        mappings = {}
        for key in topology.mappings:
            conn = len(key.atoms.indices_from_symbol("X"))
            mappings[key] = {6: "Zn_mof5_octahedral", 2: "Benzene_linear"}[conn]
        mof5 = mofgen.build(topology, mappings=mappings, refine_cell=False)
        linkers = rotatable_slots(mof5)
        assert linkers
        assert all(mof5.slots[s] == "Benzene_linear" for s in linkers)
        node = next(s for s, n in mof5.slots.items() if n == "Zn_mof5_octahedral")
        assert node not in linkers


class TestEntryPoint:
    def test_help_exits_cleanly(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["--help"])
        assert excinfo.value.code == 0
        assert "MOF and COF" in capsys.readouterr().out


class _Answer:
    """Stand-in for a questionary prompt with a scripted answer."""

    def __init__(self, value):
        self.value = value

    def ask(self):
        return self.value


def _script(monkeypatch, **by_kind):
    """Monkeypatch questionary.<kind> to return scripted answers in order."""
    import autografs.cli as cli

    for kind, values in by_kind.items():
        iterator = iter(values)

        def stub(*_args, _it=iterator, **_kwargs):
            return _Answer(next(_it))

        monkeypatch.setattr(cli.questionary, kind, stub)


class TestDeconstructWizard:
    """One scripted run through the deconstruct menu, guarding the wiring."""

    @pytest.fixture
    def session(self, mofgen):
        from autografs.cli import Session

        return Session(_gen=mofgen)

    @pytest.fixture
    def mof5_cif(self, mofgen, tmp_path):
        topology = mofgen.topologies["pcu"]
        mappings = {
            key: {6: "Zn_mof5_octahedral", 2: "Benzene_linear"}[
                len(key.atoms.indices_from_symbol("X"))
            ]
            for key in topology.mappings
        }
        mof = mofgen.build(topology, mappings=mappings, refine_cell=True, max_rmsd=0.5)
        path = tmp_path / "mof5.cif"
        mof.write_cif(path)
        return path

    def test_harvest_into_session_and_xyz(
        self, monkeypatch, session, mof5_cif, tmp_path
    ):
        from autografs.cli import deconstruct_wizard

        out = tmp_path / "mof5_sbus.xyz"
        _script(
            monkeypatch,
            path=[str(mof5_cif)],
            confirm=[True, True],  # add to session library, then write XYZ
            text=[str(out)],
        )
        deconstruct_wizard(session)
        # building units landed in the session library and on disk
        assert "node_C6O13Zn4_6X" in session.gen.sbu
        assert "linker_C6H4_2X" in session.gen.sbu
        assert out.is_file()
        assert "node_C6O13Zn4_6X" in out.read_text()

    def test_missing_file_is_reported(self, monkeypatch, session, capsys):
        from autografs.cli import deconstruct_wizard

        _script(monkeypatch, path=["does_not_exist.cif"])
        deconstruct_wizard(session)
        assert "No such file" in capsys.readouterr().out

    def test_metal_free_refusal_is_caught(
        self, monkeypatch, session, mofgen, tmp_path, capsys
    ):
        from autografs.cli import deconstruct_wizard

        topology = mofgen.topologies["hcb"]
        mappings = {
            key: {3: "Boroxine_triangle", 2: "Benzene_linear"}[
                len(key.atoms.indices_from_symbol("X"))
            ]
            for key in topology.mappings
        }
        cof = mofgen.build(topology, mappings=mappings, max_rmsd=0.5)
        path = tmp_path / "cof.cif"
        cof.write_cif(path)
        _script(monkeypatch, path=[str(path)])
        deconstruct_wizard(session)  # must not raise
        assert "Could not deconstruct" in capsys.readouterr().out
