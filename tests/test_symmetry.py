"""Tests for blueprint symmetry operations (autografs.symmetry)."""

import os

import numpy as np
import pytest
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Molecule

from autografs.fragment import Fragment
from autografs.symmetry import blueprint_operations, orbit_displacements
from autografs.topology import Topology

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "topologies_fixture.json"
)


@pytest.fixture(scope="module")
def mofgen():
    from autografs import Autografs

    return Autografs(topofile=FIXTURE_PATH)


def _slot(coords, tags, equivalence_class):
    species = ["X"] * len(coords)
    molecule = Molecule(species, coords, site_properties={"tags": list(tags)})
    return Fragment(atoms=molecule, name="slot", equivalence_class=equivalence_class)


def _p1_topology():
    """Two inequivalent 2-c slots at general positions: no symmetry
    beyond the identity."""
    cell = Lattice.orthorhombic(10.0, 11.0, 12.0)
    slot_a = _slot(
        np.array([[1.1, 2.3, 3.7], [2.1, 2.9, 4.9]]), [1, 2], equivalence_class=0
    )
    slot_b = _slot(
        np.array([[5.3, 6.1, 7.9], [6.7, 7.3, 9.1]]), [1, 2], equivalence_class=1
    )
    return Topology("p1_test", [slot_a, slot_b], cell)


def _pillar_topology():
    """One orbit of two 2-c slots at (0, 0, +-z) on a tetragonal axis.

    The site symmetry (4mm about c) pins x and y but leaves z free -
    the free fractional coordinate of a real Wyckoff 2g-type position -
    and the mirror relating the two slots must propagate the
    representative's +z displacement as -z on its partner.
    """
    cell = Lattice.tetragonal(10.0, 12.0)
    upper = _slot(
        np.array([[0.0, 0.0, 2.6], [0.0, 0.0, 4.6]]), [1, 2], equivalence_class=0
    )
    lower = _slot(
        np.array([[0.0, 0.0, -2.6], [0.0, 0.0, -4.6]]), [1, 2], equivalence_class=0
    )
    return Topology("pillar_test", [upper, lower], cell)


class TestBlueprintOperations:
    def test_pcu_recovers_the_full_cubic_group(self, mofgen):
        """pcu's node sits on an m-3m site and its three edge centers
        permute under the cubic group: 48 operations, half improper."""
        operations = blueprint_operations(mofgen.topologies["pcu"])
        assert len(operations) == 48
        assert sum(op.is_proper for op in operations) == 24
        identity = operations[0]
        assert identity.rotation == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        assert identity.slot_image == tuple(range(len(identity.slot_image)))

    def test_operations_permute_slots_consistently(self, mofgen):
        """Applying an operation to a slot centre must land on the
        centre of the slot it claims to map to."""
        topology = mofgen.topologies["dia"]
        operations = blueprint_operations(topology)
        assert len(operations) > 1
        centers = []
        for slot in topology.slots:
            dummies = [i for i, s in enumerate(slot.atoms) if s.specie.symbol == "X"]
            frac = topology.cell.get_fractional_coords(slot.atoms.cart_coords[dummies])
            centers.append(frac.mean(axis=0))
        centers = np.asarray(centers)
        for operation in operations[:10]:
            mapped = centers @ operation.rotation_array.T + np.asarray(
                operation.translation
            )
            for slot, target in enumerate(operation.slot_image):
                delta = mapped[slot] - centers[target]
                delta -= np.round(delta)
                assert np.abs(delta).max() < 1e-3

    def test_orbit_labels_are_preserved(self, mofgen):
        topology = mofgen.topologies["pcu"]
        orbits = [slot.equivalence_class for slot in topology.slots]
        for operation in blueprint_operations(topology):
            for slot, target in enumerate(operation.slot_image):
                assert orbits[slot] == orbits[target]

    def test_p1_blueprint_keeps_only_the_identity(self):
        operations = blueprint_operations(_p1_topology())
        assert len(operations) == 1
        assert operations[0].rotation == ((1, 0, 0), (0, 1, 0), (0, 0, 1))


class TestOrbitDisplacements:
    def test_pcu_is_fully_pinned(self, mofgen):
        """Every pcu site sits on a special position with no free
        direction - which is exactly why pcu builds come out right
        today (#174)."""
        displacements = orbit_displacements(mofgen.topologies["pcu"])
        assert displacements.n_free == 0
        moved = displacements.expand(np.array([]))
        assert moved.shape == (len(mofgen.topologies["pcu"]), 3)
        assert np.abs(moved).max() == 0.0

    def test_p1_blueprint_is_fully_free(self):
        """With only the identity, every orbit gets all three
        directions."""
        topology = _p1_topology()
        displacements = orbit_displacements(topology)
        assert displacements.n_free == 6  # 2 orbits x 3 directions
        moved = displacements.expand(np.array([0.1, 0.0, 0.0, 0.0, 0.2, 0.0]))
        assert moved.shape == (2, 3)
        assert np.linalg.norm(moved[0]) > 0
        assert np.linalg.norm(moved[1]) > 0

    def test_free_z_coordinate_is_found_and_pinned_directions_are_not(self):
        """The pillar orbit's site symmetry allows exactly the z
        direction: one free scalar for the whole (two-slot) orbit."""
        displacements = orbit_displacements(_pillar_topology())
        assert displacements.n_free == 1
        (basis,) = displacements.bases.values()
        # the single free direction is +-z, x and y are pinned
        assert np.allclose(basis[0][:2], 0.0, atol=1e-8)
        assert abs(basis[0][2]) > 0

    def test_displacement_propagates_by_the_relating_operation(self):
        """The two pillar slots are mirror images through z -> -z:
        moving the representative up must move its partner down by
        exactly as much, keeping the orbit one orbit."""
        topology = _pillar_topology()
        operations = blueprint_operations(topology)
        displacements = orbit_displacements(topology, operations)
        moved = displacements.expand(np.array([0.05]))
        assert np.allclose(moved[0], -moved[1], atol=1e-10)

        # displaced centres must still map onto each other under every
        # original operation
        centers = []
        for slot in topology.slots:
            dummies = [i for i, s in enumerate(slot.atoms) if s.specie.symbol == "X"]
            frac = topology.cell.get_fractional_coords(slot.atoms.cart_coords[dummies])
            centers.append(frac.mean(axis=0))
        displaced = np.asarray(centers) + moved
        for operation in operations:
            mapped = displaced @ operation.rotation_array.T + np.asarray(
                operation.translation
            )
            for slot, target in enumerate(operation.slot_image):
                delta = mapped[slot] - displaced[target]
                delta -= np.round(delta)
                assert np.abs(delta).max() < 1e-6

    def test_expand_rejects_wrong_length(self, mofgen):
        displacements = orbit_displacements(mofgen.topologies["pcu"])
        with pytest.raises(ValueError, match="parameters"):
            displacements.expand(np.array([1.0]))

    def test_bases_are_orthonormal_in_cartesian(self):
        """Basis rows, mapped to cartesian, must be unit and mutually
        orthogonal - the optimizer's step size then means Angstroms."""
        topology = _p1_topology()
        displacements = orbit_displacements(topology)
        matrix = np.asarray(topology.cell.matrix)
        for basis in displacements.bases.values():
            cartesian = basis @ matrix
            gram = cartesian @ cartesian.T
            assert np.allclose(gram, np.eye(len(basis)), atol=1e-8)
