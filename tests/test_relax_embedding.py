"""Tests for embedding relaxation (#174): symmetry-constrained slot
displacements + anchor-direction objective in the finite pipeline."""

import copy
import os

import numpy as np
import pytest
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Molecule

from autografs.alignment import prepare_build
from autografs.builder import build_framework
from autografs.fragment import Fragment
from autografs.topology import Topology

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "topologies_fixture.json"
)


@pytest.fixture(scope="module")
def mofgen():
    from autografs import Autografs

    return Autografs(topofile=FIXTURE_PATH)


BOND_CC = 2 * 0.76  # Cordero C radius x2: the C-C pair target


def _slot(zs, tags, equivalence_class):
    coords = np.array([[0.0, 0.0, z] for z in zs])
    molecule = Molecule(["X"] * len(zs), coords, site_properties={"tags": list(tags)})
    return Fragment(atoms=molecule, name="slot", equivalence_class=equivalence_class)


def _linear_sbu(name, carbons, arm):
    """A bonded carbon chain along z with a dummy off each end.

    ``carbons`` are the z positions of the chain (consecutive spacing
    must be bonding distance so tag transfer walks the real graph);
    the outermost carbons are the anchors, dummies sit at +-arm.
    """
    zs = sorted(carbons)
    species = ["C"] * len(zs) + ["X", "X"]
    coords = [[0.0, 0.0, z] for z in zs] + [[0.0, 0.0, arm], [0.0, 0.0, -arm]]
    return Fragment(atoms=Molecule(species, np.array(coords)), name=name)


def _alternating_chain():
    """A -L1- A -L2- A ... chain along c whose two edges are
    inequivalent orbits.

    The idealized embedding spaces the nodes evenly (z = 0, 0.5), so
    both half-segments get c/4 - but the real L1 is long (anchor span
    2.0) and L2 short (anchor span 0.5), which no single cell parameter
    can serve. The only symmetry-allowed freedom is the node orbit's z
    (the z -> 0.5 - z mirror pins both linkers and moves the two nodes
    oppositely), and exactly that freedom closes both bonds.
    """
    c = 8.0
    cell = Lattice.tetragonal(10.0, c)
    slots = [
        _slot([0.125 * c, -0.125 * c], [1, 4], equivalence_class=0),  # A at z=0
        _slot([0.125 * c, 0.375 * c], [1, 2], equivalence_class=1),  # L1 at 0.25
        _slot([0.375 * c, 0.625 * c], [2, 3], equivalence_class=0),  # A at z=0.5
        _slot([0.625 * c, 0.875 * c], [3, 4], equivalence_class=2),  # L2 at 0.75
    ]
    return Topology("alternating_chain", slots, cell)


def _mappings():
    # anchor spans: node 0.6, long linker 2.0, short linker 0.5 - the
    # two edges want 0.6 + 1.52 + 2.0 = 4.12 A and 2.62 A of centre
    # separation respectively, a ratio no equal-edge embedding admits
    node = _linear_sbu("node", carbons=[-0.6, 0.6], arm=1.4)
    return {
        0: copy.deepcopy(node),
        1: _linear_sbu("long_linker", carbons=[-2.0, -0.7, 0.7, 2.0], arm=2.7),
        2: copy.deepcopy(node),
        3: _linear_sbu("short_linker", carbons=[-0.5, 0.5], arm=1.2),
    }


def _inter_sbu_gaps(framework):
    """|bond length - covalent target| for every inter-SBU bond."""
    graph = framework.graph
    cell = np.asarray(graph.graph["cell"], dtype=float)
    inverse = np.linalg.inv(cell)
    gaps = []
    for a, b in graph.edges():
        if graph.nodes[a]["slot"] == graph.nodes[b]["slot"]:
            continue
        delta = np.asarray(graph.nodes[a]["coord"]) - np.asarray(
            graph.nodes[b]["coord"]
        )
        delta -= np.round(delta @ inverse) @ cell
        gaps.append(abs(float(np.linalg.norm(delta)) - BOND_CC))
    return gaps


class TestPlanParametrization:
    def test_fixed_slot_plan_is_unchanged(self):
        """Without the flag the plan carries no slot block and the
        parameter vector is the cell block alone - the legacy path,
        bit-for-bit."""
        plan = prepare_build(_alternating_chain(), _mappings())
        assert plan.slot_disp is None
        assert plan.n_slot_free == 0
        assert len(plan.initial_parameters()) == plan.cell_param.n_free

    def test_relaxed_plan_finds_the_node_z_freedom(self):
        """The chain's only symmetry-allowed displacement is the node
        orbit's z: one extra parameter, linkers pinned by the mirror."""
        plan = prepare_build(_alternating_chain(), _mappings(), relax_embedding=True)
        assert plan.slot_disp is not None
        assert plan.n_slot_free == 1
        parameters = plan.initial_parameters()
        assert len(parameters) == plan.cell_param.n_free + 1
        # displacements start at the idealized embedding
        assert np.allclose(parameters[plan.cell_param.n_free :], 0.0)


class TestRelaxedBuild:
    def test_fixed_slots_cannot_close_both_edges(self):
        """The idealized proportions force a compromise: one bond too
        long, the other too short, by well over the optimizer
        tolerance. This is #174 in miniature."""
        framework = build_framework(_alternating_chain(), _mappings())
        gaps = _inter_sbu_gaps(framework)
        assert len(gaps) == 4
        assert max(gaps) > 0.5

    def test_relaxation_closes_both_edges(self):
        """Freeing the node orbit's z closes every bond to its
        covalent target."""
        framework = build_framework(
            _alternating_chain(), _mappings(), relax_embedding=True
        )
        gaps = _inter_sbu_gaps(framework)
        assert len(gaps) == 4
        assert max(gaps) < 0.1

    def test_relaxation_preserves_the_mirror(self):
        """The two nodes must move oppositely (z -> 0.5 - z survives):
        their separation along c changes, but their midpoint stays at
        the mirror plane z = 0.25c."""
        framework = build_framework(
            _alternating_chain(), _mappings(), relax_embedding=True
        )
        graph = framework.graph
        cell = np.asarray(graph.graph["cell"], dtype=float)
        member_z: dict[int, list[float]] = {0: [], 2: []}
        for _, data in graph.nodes(data=True):
            if data["symbol"] == "C" and data["slot"] in member_z:
                member_z[data["slot"]].append(float(data["coord"][2]))
        node_z = {slot: float(np.mean(zs)) for slot, zs in member_z.items()}
        midpoint = (node_z[0] + node_z[2]) / 2.0
        assert midpoint == pytest.approx(0.25 * cell[2, 2], abs=1e-3)
        # and the segments are genuinely unequal now - the proportion
        # the idealization could not express
        separation = abs(node_z[2] - node_z[0])
        assert abs(separation - 0.5 * cell[2, 2]) > 0.5

    def test_linker_orbits_are_pinned_by_the_mirror(self):
        """The chain's mirror fixes both linker sites: only the node
        orbit carries freedom."""
        plan = prepare_build(_alternating_chain(), _mappings(), relax_embedding=True)
        assert plan.slot_disp is not None
        for orbit, basis in plan.slot_disp.bases.items():
            if plan.slot_disp.representatives[orbit] in (1, 3):
                assert len(basis) == 0

    def test_pinned_net_builds_identically_under_the_flag(self, mofgen):
        """A fully pinned blueprint has nothing to relax, and the
        augmented objective must NOT apply either - measured on real
        structures, the direction terms on a pinned net can only trade
        bond closure against an unresolvable arm mismatch. pcu under
        the flag must reproduce the default build exactly."""
        topology = mofgen.topologies["pcu"]
        mappings = {}
        for key in topology.mappings:
            conn = len(key.atoms.indices_from_symbol("X"))
            mappings[key] = {6: "Zn_mof5_octahedral", 2: "Benzene_linear"}[conn]
        default = mofgen.build(topology, mappings=mappings, max_rmsd=0.5)
        relaxed = mofgen.build(
            topology, mappings=mappings, max_rmsd=0.5, relax_embedding=True
        )
        assert np.allclose(relaxed.cell, default.cell, atol=1e-9)
        default_coords = np.array(
            [data["coord"] for _, data in sorted(default.graph.nodes(data=True))]
        )
        relaxed_coords = np.array(
            [data["coord"] for _, data in sorted(relaxed.graph.nodes(data=True))]
        )
        assert np.allclose(relaxed_coords, default_coords, atol=1e-9)
