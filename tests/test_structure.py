"""
Unit tests for the autografs.structure module.

Tests cover the Fragment and Topology classes with their methods,
using pytest and hypothesis for property-based testing.
"""

import copy
import math

import numpy as np
import pytest
from hypothesis import given, strategies as st, assume, settings
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

from autografs.structure import Fragment, Topology


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def linear_fragment():
    """Create a simple linear fragment with 2 dummies."""
    mol = Molecule(
        ["C", "X", "X", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [-1.5, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
    )
    # Create symmetry from dummies only
    symm_mol = Molecule(["H", "H"], [[1.5, 0.0, 0.0], [-1.5, 0.0, 0.0]], charge=2)
    pg = PointGroupAnalyzer(symm_mol, tolerance=0.1)
    return Fragment(atoms=mol, symmetry=pg, name="test_linear")


@pytest.fixture
def trigonal_fragment():
    """Create a trigonal planar fragment with 3 dummies (D3h symmetry)."""
    # Equilateral triangle of dummies
    coords = [
        [0.0, 0.0, 0.0],  # Central C
        [1.5, 0.0, 0.0],  # X1
        [-0.75, 1.299, 0.0],  # X2
        [-0.75, -1.299, 0.0],  # X3
    ]
    mol = Molecule(["C", "X", "X", "X"], coords)
    symm_mol = Molecule(["H", "H", "H"], coords[1:], charge=3)
    pg = PointGroupAnalyzer(symm_mol, tolerance=0.1)
    return Fragment(atoms=mol, symmetry=pg, name="test_trigonal")


@pytest.fixture
def tetrahedral_fragment():
    """Create a tetrahedral fragment with 4 dummies (Td symmetry)."""
    # Tetrahedral arrangement
    coords = [
        [0.0, 0.0, 0.0],  # Central atom
        [1.0, 1.0, 1.0],  # X1
        [1.0, -1.0, -1.0],  # X2
        [-1.0, 1.0, -1.0],  # X3
        [-1.0, -1.0, 1.0],  # X4
    ]
    mol = Molecule(["Zn", "X", "X", "X", "X"], coords)
    symm_mol = Molecule(["H", "H", "H", "H"], coords[1:], charge=4)
    pg = PointGroupAnalyzer(symm_mol, tolerance=0.1)
    return Fragment(atoms=mol, symmetry=pg, name="test_tetrahedral")


@pytest.fixture
def simple_topology(linear_fragment):
    """Create a simple topology with linear slots."""
    cell = np.eye(3) * 10.0
    slots = [linear_fragment.copy(), linear_fragment.copy()]
    return Topology(name="test_topo", slots=slots, cell=cell)


# =============================================================================
# Fragment Tests
# =============================================================================


class TestFragmentInit:
    """Test Fragment initialization."""

    def test_init_basic(self, linear_fragment):
        """Test basic initialization."""
        assert linear_fragment.name == "test_linear"
        assert len(linear_fragment.atoms) == 5
        assert linear_fragment.symmetry is not None

    def test_init_with_empty_name(self):
        """Test initialization with default empty name."""
        mol = Molecule(["C", "X"], [[0, 0, 0], [1, 0, 0]])
        symm_mol = Molecule(["H"], [[1, 0, 0]], charge=1)
        pg = PointGroupAnalyzer(symm_mol, tolerance=0.1)
        frag = Fragment(atoms=mol, symmetry=pg)
        assert frag.name == ""


class TestFragmentStr:
    """Test Fragment string representations."""

    def test_str(self, linear_fragment):
        """Test __str__ method."""
        result = str(linear_fragment)
        assert "2" in result  # 2 dummies

    def test_repr(self, linear_fragment):
        """Test __repr__ method."""
        result = repr(linear_fragment)
        assert "test_linear" in result
        assert "valent" in result


class TestFragmentEquality:
    """Test Fragment equality comparisons."""

    def test_equal_fragments(self, linear_fragment):
        """Test that copied fragments are equal."""
        frag_copy = linear_fragment.copy()
        assert linear_fragment == frag_copy

    def test_not_equal_different_symmetry(self, linear_fragment, trigonal_fragment):
        """Test that fragments with different symmetry are not equal."""
        assert linear_fragment != trigonal_fragment

    def test_not_equal_different_type(self, linear_fragment):
        """Test that Fragment != non-Fragment returns NotImplemented."""
        result = linear_fragment.__eq__("not a fragment")
        assert result is NotImplemented

    def test_ne_operator(self, linear_fragment, trigonal_fragment):
        """Test __ne__ operator."""
        assert linear_fragment != trigonal_fragment


class TestFragmentHash:
    """Test Fragment hashing."""

    def test_hash_equal_fragments(self, linear_fragment):
        """Test that equal fragments have same hash."""
        frag_copy = linear_fragment.copy()
        assert hash(linear_fragment) == hash(frag_copy)

    def test_hash_in_set(self, linear_fragment, trigonal_fragment):
        """Test that fragments can be used in sets."""
        fragment_set = {linear_fragment, trigonal_fragment, linear_fragment.copy()}
        # Equal fragments should dedupe
        assert len(fragment_set) == 2


class TestFragmentCopy:
    """Test Fragment copy functionality."""

    def test_copy_is_deep(self, linear_fragment):
        """Test that copy is a deep copy."""
        frag_copy = linear_fragment.copy()
        # Modify original
        original_coords = linear_fragment.atoms.cart_coords.copy()
        linear_fragment.atoms.translate_sites(
            range(len(linear_fragment.atoms)), [1, 0, 0]
        )
        # Copy should be unchanged
        np.testing.assert_array_almost_equal(
            frag_copy.atoms.cart_coords, original_coords
        )

    def test_copy_preserves_name(self, linear_fragment):
        """Test that copy preserves the name."""
        frag_copy = linear_fragment.copy()
        assert frag_copy.name == linear_fragment.name


class TestFragmentExtractDummies:
    """Test Fragment.extract_dummies method."""

    def test_extract_dummies_count(self, linear_fragment):
        """Test that correct number of dummies are extracted."""
        dummies = linear_fragment.extract_dummies()
        assert len(dummies) == 2

    def test_extract_dummies_species(self, linear_fragment):
        """Test that extracted dummies have X species."""
        dummies = linear_fragment.extract_dummies()
        for site in dummies:
            assert str(site.specie) == "X"

    def test_extract_dummies_trigonal(self, trigonal_fragment):
        """Test extraction for trigonal fragment."""
        dummies = trigonal_fragment.extract_dummies()
        assert len(dummies) == 3


class TestFragmentMaxDummyDistance:
    """Test Fragment.max_dummy_distance property."""

    def test_linear_distance(self, linear_fragment):
        """Test max distance for linear fragment."""
        # Dummies at [1.5, 0, 0] and [-1.5, 0, 0], distance = 3.0
        assert abs(linear_fragment.max_dummy_distance - 3.0) < 0.01

    def test_cached_property(self, linear_fragment):
        """Test that max_dummy_distance is cached."""
        dist1 = linear_fragment.max_dummy_distance
        dist2 = linear_fragment.max_dummy_distance
        assert dist1 is dist2  # Same object due to caching


class TestFragmentSymmetryCompatibility:
    """Test Fragment.has_compatible_symmetry method."""

    def test_same_fragment_compatible(self, linear_fragment):
        """Test that identical fragments are compatible."""
        assert linear_fragment.has_compatible_symmetry(linear_fragment.copy())

    def test_different_size_incompatible(self, linear_fragment, trigonal_fragment):
        """Test that different sized fragments are incompatible."""
        assert not linear_fragment.has_compatible_symmetry(trigonal_fragment)

    def test_small_fragments_compatible(self):
        """Test that small fragments (<=3 dummies) with same size are compatible."""
        # Create two linear fragments
        mol1 = Molecule(["C", "X", "X"], [[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
        symm1 = Molecule(["H", "H"], [[1, 0, 0], [-1, 0, 0]], charge=2)
        pg1 = PointGroupAnalyzer(symm1, tolerance=0.1)
        frag1 = Fragment(atoms=mol1, symmetry=pg1, name="linear1")

        mol2 = Molecule(["N", "X", "X"], [[0, 0, 0], [0, 1, 0], [0, -1, 0]])
        symm2 = Molecule(["H", "H"], [[0, 1, 0], [0, -1, 0]], charge=2)
        pg2 = PointGroupAnalyzer(symm2, tolerance=0.1)
        frag2 = Fragment(atoms=mol2, symmetry=pg2, name="linear2")

        assert frag1.has_compatible_symmetry(frag2)


class TestFragmentRotate:
    """Test Fragment.rotate method."""

    def test_rotate_preserves_center(self, linear_fragment):
        """Test that rotation preserves the center of mass of dummies."""
        dummies_before = linear_fragment.extract_dummies()
        center_before = dummies_before.cart_coords.mean(axis=0)

        linear_fragment.rotate(theta=math.pi / 4)

        dummies_after = linear_fragment.extract_dummies()
        center_after = dummies_after.cart_coords.mean(axis=0)

        np.testing.assert_array_almost_equal(center_before, center_after, decimal=5)

    def test_rotate_changes_coords(self, linear_fragment):
        """Test that rotation actually changes coordinates."""
        coords_before = linear_fragment.atoms.cart_coords.copy()
        linear_fragment.rotate(theta=math.pi / 2)
        coords_after = linear_fragment.atoms.cart_coords

        # Should not be equal after rotation
        assert not np.allclose(coords_before, coords_after)

    def test_rotate_zero_angle(self, linear_fragment):
        """Test that zero rotation doesn't change anything."""
        coords_before = linear_fragment.atoms.cart_coords.copy()
        linear_fragment.rotate(theta=0.0)
        coords_after = linear_fragment.atoms.cart_coords

        np.testing.assert_array_almost_equal(coords_before, coords_after)


class TestFragmentFlip:
    """Test Fragment.flip method."""

    def test_flip_not_implemented(self, linear_fragment):
        """Test that flip raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            linear_fragment.flip()


# =============================================================================
# Topology Tests
# =============================================================================


class TestTopologyInit:
    """Test Topology initialization."""

    def test_init_basic(self, simple_topology):
        """Test basic initialization."""
        assert simple_topology.name == "test_topo"
        assert len(simple_topology) == 2

    def test_init_with_numpy_cell(self, linear_fragment):
        """Test initialization with numpy array cell."""
        cell = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=float)
        topo = Topology(name="test", slots=[linear_fragment], cell=cell)
        assert isinstance(topo.cell, Lattice)
        np.testing.assert_array_almost_equal(topo.cell.matrix, cell)

    def test_init_with_lattice_cell(self, linear_fragment):
        """Test initialization with Lattice object cell."""
        lattice = Lattice.cubic(10.0)
        topo = Topology(name="test", slots=[linear_fragment], cell=lattice)
        assert topo.cell is lattice

    def test_mappings_created(self, linear_fragment, trigonal_fragment):
        """Test that mappings are correctly created."""
        slots = [linear_fragment, trigonal_fragment, linear_fragment.copy()]
        topo = Topology(name="test", slots=slots, cell=np.eye(3) * 10)
        # Should have 2 unique slot types
        assert len(topo.mappings) == 2


class TestTopologyLen:
    """Test Topology __len__ method."""

    def test_len(self, simple_topology):
        """Test that len returns number of slots."""
        assert len(simple_topology) == 2


class TestTopologyRepr:
    """Test Topology __repr__ method."""

    def test_repr(self, simple_topology):
        """Test repr returns name."""
        assert repr(simple_topology) == "test_topo"


class TestTopologyCopy:
    """Test Topology.copy method."""

    def test_copy_is_deep(self, simple_topology):
        """Test that copy is a deep copy."""
        topo_copy = simple_topology.copy()
        # Modify original
        simple_topology.name = "modified"
        # Copy should be unchanged
        assert topo_copy.name == "test_topo"


class TestTopologyGetCompatibleSlots:
    """Test Topology.get_compatible_slots method."""

    def test_compatible_slots_returned(self, simple_topology, linear_fragment):
        """Test that compatible slots are found."""
        candidate = linear_fragment.copy()
        result = simple_topology.get_compatible_slots(candidate)
        # Should find slots since candidate matches
        assert len(result) > 0

    def test_incompatible_slots_empty(self, simple_topology, tetrahedral_fragment):
        """Test that incompatible slots return empty lists."""
        result = simple_topology.get_compatible_slots(tetrahedral_fragment)
        # All values should be empty lists
        for slot_type, indices in result.items():
            assert indices == []


class TestTopologyScaleSlots:
    """Test Topology.scale_slots method."""

    def test_scale_changes_cell(self, simple_topology):
        """Test that scaling changes cell parameters."""
        original_abc = simple_topology.cell.abc
        simple_topology.scale_slots(scales=(20.0, 20.0, 20.0))
        new_abc = simple_topology.cell.abc
        assert new_abc[0] == 20.0
        assert new_abc[1] == 20.0
        assert new_abc[2] == 20.0

    def test_scale_preserves_angles(self, linear_fragment):
        """Test that scaling preserves cell angles."""
        lattice = Lattice.from_parameters(10, 10, 10, 90, 90, 120)
        topo = Topology(name="test", slots=[linear_fragment], cell=lattice)
        original_angles = topo.cell.angles
        topo.scale_slots(scales=(5.0, 5.0, 5.0))
        np.testing.assert_array_almost_equal(topo.cell.angles, original_angles)

    def test_scale_default_values(self, simple_topology):
        """Test scaling with default values (1.0, 1.0, 1.0)."""
        original_abc = simple_topology.cell.abc
        simple_topology.scale_slots()  # Use defaults
        new_abc = simple_topology.cell.abc
        np.testing.assert_array_almost_equal(new_abc, (1.0, 1.0, 1.0))


# =============================================================================
# Hypothesis Property-Based Tests
# =============================================================================


class TestFragmentProperties:
    """Property-based tests for Fragment using Hypothesis."""

    @given(st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False))
    @settings(max_examples=20)
    def test_rotate_preserves_atom_count(self, theta):
        """Test that rotation preserves the number of atoms."""
        mol = Molecule(["C", "X", "X"], [[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
        symm_mol = Molecule(["H", "H"], [[1, 0, 0], [-1, 0, 0]], charge=2)
        pg = PointGroupAnalyzer(symm_mol, tolerance=0.1)
        frag = Fragment(atoms=mol, symmetry=pg, name="test")

        original_count = len(frag.atoms)
        frag.rotate(theta=theta)
        assert len(frag.atoms) == original_count

    @given(st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False))
    @settings(max_examples=20)
    def test_rotate_preserves_dummy_distance(self, theta):
        """Test that rotation preserves the max dummy distance."""
        mol = Molecule(["C", "X", "X"], [[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
        symm_mol = Molecule(["H", "H"], [[1, 0, 0], [-1, 0, 0]], charge=2)
        pg = PointGroupAnalyzer(symm_mol, tolerance=0.1)
        frag = Fragment(atoms=mol, symmetry=pg, name="test")

        # Clear cache and get distance
        if "max_dummy_distance" in frag.__dict__:
            del frag.__dict__["max_dummy_distance"]
        original_dist = frag.max_dummy_distance

        frag.rotate(theta=theta)

        # Clear cache and get new distance
        if "max_dummy_distance" in frag.__dict__:
            del frag.__dict__["max_dummy_distance"]
        new_dist = frag.max_dummy_distance

        assert abs(original_dist - new_dist) < 0.01


class TestTopologyProperties:
    """Property-based tests for Topology using Hypothesis."""

    @given(
        st.tuples(
            st.floats(min_value=1.0, max_value=100.0, allow_nan=False),
            st.floats(min_value=1.0, max_value=100.0, allow_nan=False),
            st.floats(min_value=1.0, max_value=100.0, allow_nan=False),
        )
    )
    @settings(max_examples=20)
    def test_scale_slots_sets_correct_lengths(self, scales):
        """Test that scale_slots sets correct cell lengths."""
        mol = Molecule(["C", "X", "X"], [[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
        symm_mol = Molecule(["H", "H"], [[1, 0, 0], [-1, 0, 0]], charge=2)
        pg = PointGroupAnalyzer(symm_mol, tolerance=0.1)
        frag = Fragment(atoms=mol, symmetry=pg, name="test")

        topo = Topology(name="test", slots=[frag], cell=np.eye(3) * 10)
        topo.scale_slots(scales=scales)

        np.testing.assert_array_almost_equal(topo.cell.abc, scales, decimal=5)

    @given(st.integers(min_value=1, max_value=10))
    @settings(max_examples=10)
    def test_topology_len_matches_slots(self, n_slots):
        """Test that len(topology) matches number of slots."""
        mol = Molecule(["C", "X", "X"], [[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
        symm_mol = Molecule(["H", "H"], [[1, 0, 0], [-1, 0, 0]], charge=2)
        pg = PointGroupAnalyzer(symm_mol, tolerance=0.1)
        frag = Fragment(atoms=mol, symmetry=pg, name="test")

        slots = [frag.copy() for _ in range(n_slots)]
        topo = Topology(name="test", slots=slots, cell=np.eye(3) * 10)

        assert len(topo) == n_slots
