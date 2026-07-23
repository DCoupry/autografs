"""Tests for deliberately emptied blueprint slots (#179).

A slot (or slot type) mapped to None places nothing; its two
neighbours bond directly - the finite mirror of rod_build's contracted
lateral slots, and what edge-decorated nets like tbo need before real
materials (HKUST-1) can round-trip on them.
"""

import os

import numpy as np
import pytest

from autografs.exceptions import NetMismatchError

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "topologies_fixture.json"
)


@pytest.fixture(scope="module")
def mofgen():
    from autografs import Autografs

    return Autografs(topofile=FIXTURE_PATH)


def _mappings_with_empty_edges(topology):
    mappings = {}
    for key in topology.mappings:
        conn = len(key.atoms.indices_from_symbol("X"))
        mappings[key] = "Zn_mof5_octahedral" if conn == 6 else None
    return mappings


@pytest.fixture(scope="module")
def empty_pcu(mofgen):
    topology = mofgen.topologies["pcu"]
    return mofgen.build(
        topology,
        mappings=_mappings_with_empty_edges(topology),
        max_rmsd=0.5,
        verify_net=True,
    )


class TestEmptySlotBuild:
    def test_builds_and_verifies(self, mofgen, empty_pcu):
        """pcu with its edge centers empty: nodes bond their own
        periodic images directly, and verify_net (which contracts the
        emptied slots on the blueprint side) accepts. The fixture
        build passing verify_net IS the assertion; here we pin the
        composition: one node, nothing else."""
        assert empty_pcu.structure.composition.reduced_formula == "Zn4C6O13"
        assert empty_pcu.graph.graph["empty_slots"] == [1, 2, 3]

    def test_neighbours_bond_through_the_boundary(self, empty_pcu):
        """The direct node-node bonds are quotient self-loops: same
        slot, nonzero crossing."""
        graph = empty_pcu.graph
        cell = np.asarray(graph.graph["cell"], dtype=float)
        inverse = np.linalg.inv(cell)
        crossings = 0
        for u, v in graph.edges():
            delta = np.asarray(graph.nodes[u]["coord"]) - np.asarray(
                graph.nodes[v]["coord"]
            )
            if np.any(np.round(delta @ inverse) != 0):
                crossings += 1
        assert crossings == 3  # one direct bond per cell axis

    def test_bond_length_targets_the_neighbour_anchors(self, empty_pcu):
        """The cell shrinks to put the two anchors one covalent bond
        apart - no phantom room is left for the absent linker."""
        a = empty_pcu.structure.lattice.abc[0]
        assert a < 10.0  # MOF-5 with benzene linkers sits at ~12.9

    def test_slot_type_none_and_index_none_agree(self, mofgen):
        topology = mofgen.topologies["pcu"]
        by_index = {0: "Zn_mof5_octahedral", 1: None, 2: None, 3: None}
        framework = mofgen.build(
            topology, mappings=by_index, max_rmsd=0.5, verify_net=True
        )
        assert framework.graph.graph["empty_slots"] == [1, 2, 3]

    def test_emptying_a_node_slot_is_rejected(self, mofgen):
        topology = mofgen.topologies["pcu"]
        mappings = {}
        for key in topology.mappings:
            conn = len(key.atoms.indices_from_symbol("X"))
            mappings[key] = None if conn == 6 else "Benzene_linear"
        with pytest.raises(ValueError, match="2-connected"):
            mofgen.build(topology, mappings=mappings)

    def test_partial_emptying_by_index(self, mofgen):
        """Emptying one edge center and filling the other two is a
        different net than pcu - verify_net must say so."""
        topology = mofgen.topologies["pcu"]
        mappings = {}
        for key in topology.mappings:
            conn = len(key.atoms.indices_from_symbol("X"))
            if conn == 6:
                mappings[key] = "Zn_mof5_octahedral"
            else:
                mappings[key] = "Benzene_linear"
        mappings[1] = None  # override one edge center only
        framework = mofgen.build(topology, mappings=mappings, max_rmsd=0.5)
        assert framework.graph.graph["empty_slots"] == [1]
        framework.verify_net(topology)  # contraction keeps them comparable


class TestEmptySlotPersistence:
    def test_marker_survives_save_load(self, empty_pcu, mofgen, tmp_path):
        from autografs.framework import Framework

        path = tmp_path / "empty_pcu.json"
        empty_pcu.save(path)
        loaded = Framework.load(path)
        assert loaded.graph.graph["empty_slots"] == [1, 2, 3]
        loaded.verify_net(mofgen.topologies["pcu"])  # must not raise

    def test_marker_survives_supercell(self, empty_pcu):
        doubled = empty_pcu.supercell((2, 1, 1))
        assert doubled.graph.graph["empty_slots"] == [1, 2, 3]


class TestBlueprintContraction:
    def test_contracting_a_node_slot_raises(self, mofgen, empty_pcu):
        """A framework claiming a 6-connected slot was emptied must be
        rejected, not silently contracted."""
        from autografs.framework import Framework

        graph = empty_pcu.graph.copy()
        graph.graph["empty_slots"] = [0]
        with pytest.raises(NetMismatchError):
            Framework(graph, name="rigged").verify_net(mofgen.topologies["pcu"])
