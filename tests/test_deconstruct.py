"""Tests for framework deconstruction (autografs.deconstruct) and net
identification (autografs.net.identify_net)."""

import os
from collections import Counter

import pytest
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from autografs.deconstruct import _hill_formula, deconstruct
from autografs.exceptions import DeconstructionError
from autografs.net import (
    identify_net,
    net_signature,
    topology_quotient_edges,
)

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "topologies_fixture.json"
)

FIXTURE_NETS = ["acs", "dia", "hcb", "pcu", "sql", "srs"]

# the textbook coordination sequence of the primitive cubic net
PCU_SEQUENCE = (6, 18, 38, 66, 102, 146, 198, 258, 326, 402)


@pytest.fixture(scope="module")
def mofgen():
    from autografs import Autografs

    return Autografs(topofile=FIXTURE_PATH)


def build(mofgen, topo_name, choices, **kwargs):
    topology = mofgen.topologies[topo_name]
    mappings = {}
    for key in topology.mappings:
        conn = len(key.atoms.indices_from_symbol("X"))
        mappings[key] = choices[conn]
    kwargs.setdefault("max_rmsd", 0.5)
    return mofgen.build(topology, mappings=mappings, **kwargs)


@pytest.fixture(scope="module")
def mof5(mofgen):
    return build(
        mofgen,
        "pcu",
        {6: "Zn_mof5_octahedral", 2: "Benzene_linear"},
        refine_cell=True,
    )


@pytest.fixture(scope="module")
def mof5_deconstruction(mofgen, mof5):
    return mofgen.deconstruct(mof5.structure)


class TestNetSignature:
    def test_pcu_coordination_sequence(self, mofgen):
        edges = topology_quotient_edges(mofgen.topologies["pcu"])
        signature = net_signature(edges)
        assert signature == ((PCU_SEQUENCE, 1),)

    def test_fixture_signatures_all_distinct(self, mofgen):
        signatures = {}
        for name in FIXTURE_NETS:
            edges = topology_quotient_edges(mofgen.topologies[name])
            signatures[name] = net_signature(edges)
        assert len(set(signatures.values())) == len(FIXTURE_NETS)

    def test_caps_are_pruned(self, mofgen):
        """A dangling 1-coordinated vertex must not change the signature."""
        edges = topology_quotient_edges(mofgen.topologies["pcu"])
        with_cap = Counter(edges)
        with_cap[(0, 99, (0, 0, 0))] += 1
        assert net_signature(with_cap) == net_signature(edges)

    def test_edge_centers_are_contracted(self, mofgen):
        """The contracted signature ignores 2-coordinated decorations:
        pcu's blueprint (with edge centers) and its bare quotient (three
        self-loops on one vertex) reduce to the same signature."""
        bare: Counter = Counter(
            {
                (0, 0, (1, 0, 0)): 1,
                (0, 0, (0, 1, 0)): 1,
                (0, 0, (0, 0, 1)): 1,
            }
        )
        edges = topology_quotient_edges(mofgen.topologies["pcu"])
        assert net_signature(bare) == net_signature(edges)
        # while the uncontracted signature separates them
        assert net_signature(bare, contract=False) != net_signature(
            edges, contract=False
        )


class TestIdentifyNet:
    @pytest.mark.parametrize("name", FIXTURE_NETS)
    def test_each_fixture_net_identifies_itself(self, mofgen, name):
        edges = topology_quotient_edges(mofgen.topologies[name])
        assert identify_net(edges, mofgen.topologies) == [name]

    def test_unknown_net_returns_empty(self, mofgen):
        # the 4-coordinated nbo net is not in the fixture library
        nbo_like: Counter = Counter(
            {
                (0, 1, (0, 0, 0)): 1,
                (0, 1, (1, 0, 0)): 1,
                (0, 2, (0, 0, 0)): 1,
                (0, 2, (0, 0, 1)): 1,
                (1, 2, (0, 1, 0)): 1,
                (1, 2, (-1, 0, 1)): 1,
            }
        )
        assert identify_net(nbo_like, mofgen.topologies) == []


class TestDeconstructMOF5:
    def test_units(self, mof5_deconstruction):
        kinds = Counter(unit.kind for unit in mof5_deconstruction.units)
        assert kinds == {"node": 1, "linker": 3}
        by_kind = {unit.kind: unit for unit in mof5_deconstruction.units}
        assert by_kind["node"].n_connections == 6
        assert by_kind["linker"].n_connections == 2

    def test_fragments_are_library_ready(self, mof5_deconstruction):
        fragments = mof5_deconstruction.fragments
        assert len(fragments) == 2
        node = fragments["node_C6O13Zn4_6X"]
        linker = fragments["linker_C6H4_2X"]
        assert len(node.atoms.indices_from_symbol("X")) == 6
        assert len(linker.atoms.indices_from_symbol("X")) == 2

    def test_net_identified(self, mof5_deconstruction):
        assert mof5_deconstruction.net_candidates == ["pcu"]

    def test_unit_atoms_partition_the_structure(self, mof5_deconstruction):
        indices = [i for unit in mof5_deconstruction.units for i in unit.atom_indices]
        assert sorted(indices) == list(range(len(mof5_deconstruction.structure)))

    def test_rebuild_from_extracted_fragments(self, mofgen, mof5_deconstruction):
        """The extracted fragments must be buildable as-is."""
        topology = mofgen.topologies["pcu"]
        fragments = mof5_deconstruction.fragments
        mappings = {}
        for key in topology.mappings:
            conn = len(key.atoms.indices_from_symbol("X"))
            name = {6: "node_C6O13Zn4_6X", 2: "linker_C6H4_2X"}[conn]
            mappings[key] = fragments[name]
        rebuilt = mofgen.build(
            topology, mappings=mappings, refine_cell=True, max_rmsd=0.5
        )
        rebuilt.verify_net(topology)
        assert len(rebuilt) == len(mof5_deconstruction.structure)

    def test_write_xyz_roundtrip(self, mof5_deconstruction, tmp_path):
        from autografs.utils import xyz_to_sbu

        path = mof5_deconstruction.write_xyz(tmp_path / "harvested.xyz")
        loaded = xyz_to_sbu(str(path))
        assert set(loaded) == set(mof5_deconstruction.fragments)
        for name, fragment in loaded.items():
            original = mof5_deconstruction.fragments[name]
            assert len(fragment.atoms) == len(original.atoms)
            assert fragment.has_compatible_symmetry(original, max_rmsd=0.05)


class TestGuestRemoval:
    def test_free_guests_are_removed_and_reported(self, mofgen, mof5):
        structure = mof5.structure.copy()
        structure.append("Xe", [0.5, 0.5, 0.5])
        structure.append("O", [0.25, 0.25, 0.25])
        structure.append("H", [0.251, 0.25, 0.19])
        structure.append("H", [0.19, 0.25, 0.251])
        result = mofgen.deconstruct(structure)
        assert result.guest_formulas == ["H2O", "Xe"]
        assert result.net_candidates == ["pcu"]
        assert len(result.structure) == len(mof5.structure)


class TestPaddlewheel:
    def test_2d_paddlewheel_framework(self, mofgen):
        layer = build(mofgen, "sql", {4: "Zn_square_paddlewheel", 2: "Benzene_linear"})
        result = mofgen.deconstruct(layer.structure)
        assert result.net_candidates == ["sql"]
        assert "node_C4O8Zn2_4X" in result.fragments
        node = result.fragments["node_C4O8Zn2_4X"]
        assert len(node.atoms.indices_from_symbol("X")) == 4


class TestCatenation:
    @pytest.fixture(scope="class")
    def dia(self, mofgen):
        # a sparse dia (long linker) so a second net fits without clashes
        return build(
            mofgen,
            "dia",
            {4: "CdGaS_cluster_tetrahedral", 2: "Bis_phenylethynylbenzene_linear"},
            max_rmsd=0.6,
        )

    def test_single_framework_is_not_catenated(self, mofgen, dia):
        result = mofgen.deconstruct(dia.structure)
        assert result.n_periodic_components == 1
        assert result.is_catenated is False
        assert result.subframework_nets == [["dia"]]
        assert result.net_candidates == ["dia"]

    def test_two_fold_interpenetration_detected(self, mofgen, dia):
        catenated = dia.interpenetrate(2)
        result = mofgen.deconstruct(catenated.structure)
        assert result.n_periodic_components == 2
        assert result.is_catenated is True
        # each subframework identified independently...
        assert result.subframework_nets == [["dia"], ["dia"]]
        # ...and the consensus is the single realized net
        assert result.net_candidates == ["dia"]
        # the fold is surfaced in the repr
        assert "2-fold" in repr(result)


class TestCOF:
    """Metal-free frameworks take the branch-point path."""

    def test_2d_boroxine_cof_round_trip(self, mofgen):
        cof = build(mofgen, "hcb", {3: "Boroxine_triangle", 2: "Benzene_linear"})
        result = mofgen.deconstruct(cof.structure)
        assert result.net_candidates == ["hcb"]
        # boroxine ring is the 3-c node, benzene the 2-c linker
        assert "node_B3O3_3X" in result.fragments
        assert "linker_C6H4_2X" in result.fragments
        assert len(result.fragments["node_B3O3_3X"].atoms.indices_from_symbol("X")) == 3
        # rebuild from the extracted fragments and confirm the net
        topology = mofgen.topologies["hcb"]
        mappings = {
            key: result.fragments[
                {3: "node_B3O3_3X", 2: "linker_C6H4_2X"}[
                    len(key.atoms.indices_from_symbol("X"))
                ]
            ]
            for key in topology.mappings
        }
        rebuilt = mofgen.build(topology, mappings=mappings, max_rmsd=0.6)
        rebuilt.verify_net(topology)

    def test_3d_organic_srs(self, mofgen):
        cof = build(
            mofgen,
            "srs",
            {3: "Boroxine_triangle", 2: "Benzene_linear"},
            max_rmsd=0.6,
        )
        result = mofgen.deconstruct(cof.structure)
        assert result.net_candidates == ["srs"]
        assert all(unit.kind in ("node", "linker") for unit in result.units)

    def test_3d_organic_tetrahedral_dia(self, mofgen):
        cof = build(
            mofgen,
            "dia",
            {4: "N66_tetrahedral", 2: "Benzene_linear"},
            max_rmsd=0.7,
        )
        result = mofgen.deconstruct(cof.structure)
        assert result.net_candidates == ["dia"]
        node = next(unit for unit in result.units if unit.kind == "node")
        assert node.n_connections == 4

    def test_metal_path_unaffected(self, mofgen):
        """A metal MOF still uses metal-oxo clustering, not branch points."""
        mof = build(
            mofgen,
            "pcu",
            {6: "Zn_mof5_octahedral", 2: "Benzene_linear"},
            refine_cell=True,
        )
        result = mofgen.deconstruct(mof.structure)
        assert result.net_candidates == ["pcu"]
        assert "node_C6O13Zn4_6X" in result.fragments


class TestErrors:
    def test_molecular_crystal_rejected(self):
        structure = Structure(Lattice.cubic(20.0), ["He"], [[0.5, 0.5, 0.5]])
        with pytest.raises(DeconstructionError, match="periodic component"):
            deconstruct(structure)

    def test_disordered_structure_rejected(self):
        structure = Structure(
            Lattice.cubic(5.0), [{"Fe": 0.5, "Co": 0.5}], [[0.0, 0.0, 0.0]]
        )
        with pytest.raises(DeconstructionError, match="occupied"):
            deconstruct(structure)

    def test_rod_mof_rejected(self):
        """A chain of corner-sharing metal octahedra is a rod SBU."""
        structure = Structure(
            Lattice.orthorhombic(3.8, 15.0, 15.0),
            ["Zn", "O"],
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        )
        with pytest.raises(DeconstructionError, match="rod|periodic"):
            deconstruct(structure)


def test_hill_formula():
    assert _hill_formula(["C", "H", "H", "C", "O"]) == "C2H2O"
    assert _hill_formula(["Zn", "O", "Zn"]) == "OZn2"
    assert _hill_formula(["H", "O", "H"]) == "H2O"
