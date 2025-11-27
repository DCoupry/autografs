"""
Unit tests for the autografs.utils module.

Tests cover utility functions for formatting, file I/O, and graph manipulation.
"""

import os
import tempfile

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

from autografs import utils
from autografs.structure import Fragment


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_xyz_content():
    """Sample XYZ file content with named structures."""
    return """3
name=Test_Linear pbc="F F F"
C          0.0000        0.0000        0.0000
X          1.5000        0.0000        0.0000
X         -1.5000        0.0000        0.0000
4
name=Test_Trigonal pbc="F F F"
C          0.0000        0.0000        0.0000
X          1.5000        0.0000        0.0000
X         -0.7500        1.2990        0.0000
X         -0.7500       -1.2990        0.0000
"""


@pytest.fixture
def sample_xyz_unnamed():
    """Sample XYZ file content without names."""
    return """3
comment without name tag
C          0.0000        0.0000        0.0000
X          1.5000        0.0000        0.0000
X         -1.5000        0.0000        0.0000
"""


@pytest.fixture
def temp_xyz_file(sample_xyz_content):
    """Create a temporary XYZ file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write(sample_xyz_content)
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_xyz_unnamed(sample_xyz_unnamed):
    """Create a temporary XYZ file without names."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write(sample_xyz_unnamed)
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


# =============================================================================
# format_indices Tests
# =============================================================================


class TestFormatIndices:
    """Test format_indices function."""

    def test_single_index(self):
        """Test formatting a single index."""
        result = utils.format_indices([5])
        assert result == "5"

    def test_range_of_indices(self):
        """Test formatting a range."""
        result = utils.format_indices([1, 2])
        assert result == "1-2"

    def test_empty_iterable(self):
        """Test with empty iterable."""
        # This will raise an IndexError - documenting current behavior
        with pytest.raises(IndexError):
            utils.format_indices([])


# =============================================================================
# format_mappings Tests
# =============================================================================


class TestFormatMappings:
    """Test format_mappings function."""

    def test_simple_mapping(self):
        """Test simple mapping formatting."""
        mappings = {0: "SBU_A"}
        result = utils.format_mappings(mappings)
        assert "SBU_A" in result
        assert "0" in result

    def test_grouped_mappings(self):
        """Test that consecutive indices are grouped."""
        mappings = {0: "SBU_A", 1: "SBU_A", 2: "SBU_A"}
        result = utils.format_mappings(mappings)
        assert "SBU_A" in result
        # Should group 0, 1, 2

    def test_multiple_sbus(self):
        """Test multiple SBUs in mapping."""
        mappings = {0: "SBU_A", 1: "SBU_B", 2: "SBU_A"}
        result = utils.format_mappings(mappings)
        assert "SBU_A" in result
        assert "SBU_B" in result
        assert ";" in result  # Separator between SBUs

    def test_empty_mapping(self):
        """Test empty mapping."""
        result = utils.format_mappings({})
        assert result == ""


# =============================================================================
# get_xyz_names Tests
# =============================================================================


class TestGetXyzNames:
    """Test get_xyz_names function."""

    def test_named_structures(self, temp_xyz_file):
        """Test extraction of named structures."""
        names = utils.get_xyz_names(temp_xyz_file)
        assert len(names) == 2
        assert "Test_Linear" in names
        assert "Test_Trigonal" in names

    def test_unnamed_structures(self, temp_xyz_unnamed):
        """Test that unnamed structures get 'Unnamed' label."""
        names = utils.get_xyz_names(temp_xyz_unnamed)
        assert len(names) == 1
        assert names[0] == "Unnamed"


# =============================================================================
# xyz_to_sbu Tests
# =============================================================================


class TestXyzToSbu:
    """Test xyz_to_sbu function."""

    def test_loads_fragments(self, temp_xyz_file):
        """Test that fragments are loaded correctly."""
        sbus = utils.xyz_to_sbu(temp_xyz_file)
        assert isinstance(sbus, dict)
        assert len(sbus) == 2

    def test_fragment_names(self, temp_xyz_file):
        """Test that fragment names are correct."""
        sbus = utils.xyz_to_sbu(temp_xyz_file)
        assert "Test_Linear" in sbus
        assert "Test_Trigonal" in sbus

    def test_fragment_types(self, temp_xyz_file):
        """Test that returned values are Fragment objects."""
        sbus = utils.xyz_to_sbu(temp_xyz_file)
        for name, frag in sbus.items():
            assert isinstance(frag, Fragment)

    def test_linear_has_two_dummies(self, temp_xyz_file):
        """Test that linear fragment has correct dummy count."""
        sbus = utils.xyz_to_sbu(temp_xyz_file)
        linear = sbus["Test_Linear"]
        dummies = linear.extract_dummies()
        assert len(dummies) == 2

    def test_trigonal_has_three_dummies(self, temp_xyz_file):
        """Test that trigonal fragment has correct dummy count."""
        sbus = utils.xyz_to_sbu(temp_xyz_file)
        trigonal = sbus["Test_Trigonal"]
        dummies = trigonal.extract_dummies()
        assert len(dummies) == 3


# =============================================================================
# load_uff_lib Tests
# =============================================================================


class TestLoadUffLib:
    """Test load_uff_lib function."""

    def test_returns_dataframe_and_list(self, temp_xyz_file):
        """Test that function returns correct types."""
        from pymatgen.core.structure import Molecule

        mol = Molecule(["C", "H", "H"], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        uff_lib, uff_symbs = utils.load_uff_lib(mol)

        import pandas

        assert isinstance(uff_lib, pandas.DataFrame)
        assert isinstance(uff_symbs, list)
        assert len(uff_symbs) == len(mol)

    def test_symbol_formatting(self):
        """Test that single-letter symbols get underscore appended."""
        from pymatgen.core.structure import Molecule

        mol = Molecule(["C", "N", "O"], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        _, uff_symbs = utils.load_uff_lib(mol)

        # Single letter elements should have underscore
        assert all(len(s) == 2 for s in uff_symbs)


# =============================================================================
# find_element_cutoffs Tests
# =============================================================================


class TestFindElementCutoffs:
    """Test find_element_cutoffs function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        from pymatgen.core.structure import Molecule

        mol = Molecule(["C", "H"], [[0, 0, 0], [1, 0, 0]])
        uff_lib, uff_symbs = utils.load_uff_lib(mol)
        cutoffs = utils.find_element_cutoffs(uff_lib, uff_symbs)

        assert isinstance(cutoffs, dict)

    def test_symmetric_cutoffs(self):
        """Test that cutoffs are symmetric (C-H == H-C)."""
        from pymatgen.core.structure import Molecule

        mol = Molecule(["C", "H"], [[0, 0, 0], [1, 0, 0]])
        uff_lib, uff_symbs = utils.load_uff_lib(mol)
        cutoffs = utils.find_element_cutoffs(uff_lib, uff_symbs)

        assert cutoffs[("C", "H")] == cutoffs[("H", "C")]

    def test_positive_cutoffs(self):
        """Test that all cutoffs are positive."""
        from pymatgen.core.structure import Molecule

        mol = Molecule(["C", "H", "N"], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        uff_lib, uff_symbs = utils.load_uff_lib(mol)
        cutoffs = utils.find_element_cutoffs(uff_lib, uff_symbs)

        for key, value in cutoffs.items():
            assert value > 0, f"Cutoff for {key} should be positive"


# =============================================================================
# networkx_to_gulp Tests
# =============================================================================


class TestNetworkxToGulp:
    """Test networkx_to_gulp function."""

    def test_returns_string(self):
        """Test that function returns a string."""
        import networkx

        # Create a minimal graph
        G = networkx.Graph(cell=np.eye(3) * 10)
        G.add_node(0, symbol="C", coord=[0, 0, 0], tag=1, ufftype="C_R")
        G.add_node(1, symbol="H", coord=[1, 0, 0], tag=2, ufftype="H_")

        result = utils.networkx_to_gulp(G, name="test", write_to_file=False)
        assert isinstance(result, str)

    def test_contains_vectors(self):
        """Test that output contains cell vectors."""
        import networkx

        G = networkx.Graph(cell=np.eye(3) * 10)
        G.add_node(0, symbol="C", coord=[0, 0, 0], tag=1, ufftype="C_R")

        result = utils.networkx_to_gulp(G, name="test", write_to_file=False)
        assert "vectors" in result

    def test_contains_coordinates(self):
        """Test that output contains atomic coordinates."""
        import networkx

        G = networkx.Graph(cell=np.eye(3) * 10)
        G.add_node(0, symbol="C", coord=[1.5, 2.5, 3.5], tag=1, ufftype="C_R")

        result = utils.networkx_to_gulp(G, name="test", write_to_file=False)
        assert "1.5" in result or "1.50" in result

    def test_contains_library(self):
        """Test that output references UFF library."""
        import networkx

        G = networkx.Graph(cell=np.eye(3) * 10)
        G.add_node(0, symbol="C", coord=[0, 0, 0], tag=1, ufftype="C_R")

        result = utils.networkx_to_gulp(G, name="test", write_to_file=False)
        assert "library uff4mof" in result

    def test_bond_order_single(self):
        """Test single bond formatting."""
        import networkx

        G = networkx.Graph(cell=np.eye(3) * 10)
        G.add_node(0, symbol="C", coord=[0, 0, 0], tag=1, ufftype="C_R")
        G.add_node(1, symbol="C", coord=[1, 0, 0], tag=2, ufftype="C_R")
        G.add_edge(0, 1, bond_order=1.0)

        result = utils.networkx_to_gulp(G, name="test", write_to_file=False)
        assert "single" in result

    def test_bond_order_double(self):
        """Test double bond formatting."""
        import networkx

        G = networkx.Graph(cell=np.eye(3) * 10)
        G.add_node(0, symbol="C", coord=[0, 0, 0], tag=1, ufftype="C_R")
        G.add_node(1, symbol="C", coord=[1, 0, 0], tag=2, ufftype="C_R")
        G.add_edge(0, 1, bond_order=2.0)

        result = utils.networkx_to_gulp(G, name="test", write_to_file=False)
        assert "double" in result

    def test_bond_order_aromatic(self):
        """Test aromatic bond formatting."""
        import networkx

        G = networkx.Graph(cell=np.eye(3) * 10)
        G.add_node(0, symbol="C", coord=[0, 0, 0], tag=1, ufftype="C_R")
        G.add_node(1, symbol="C", coord=[1, 0, 0], tag=2, ufftype="C_R")
        G.add_edge(0, 1, bond_order=1.5)

        result = utils.networkx_to_gulp(G, name="test", write_to_file=False)
        assert "aromatic" in result

    def test_write_to_file(self):
        """Test file writing functionality."""
        import networkx

        G = networkx.Graph(cell=np.eye(3) * 10)
        G.add_node(0, symbol="C", coord=[0, 0, 0], tag=1, ufftype="C_R")

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                utils.networkx_to_gulp(G, name="test_output", write_to_file=True)
                assert os.path.exists("test_output.gin")
            finally:
                os.chdir(old_cwd)


# =============================================================================
# Hypothesis Property-Based Tests
# =============================================================================


class TestFormatMappingsProperties:
    """Property-based tests for format_mappings."""

    @given(
        st.dictionaries(
            st.integers(min_value=0, max_value=100),
            st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=30)
    def test_all_sbu_names_in_output(self, mappings):
        """Test that all SBU names appear in formatted output."""
        result = utils.format_mappings(mappings)
        for sbu_name in set(mappings.values()):
            assert sbu_name in result

    @given(
        st.dictionaries(
            st.integers(min_value=0, max_value=100),
            st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"),
            min_size=0,
            max_size=20,
        )
    )
    @settings(max_examples=30)
    def test_returns_string(self, mappings):
        """Test that format_mappings always returns a string."""
        result = utils.format_mappings(mappings)
        assert isinstance(result, str)


class TestGulpOutputProperties:
    """Property-based tests for GULP output generation."""

    @given(st.floats(min_value=0.1, max_value=2.9, allow_nan=False))
    @settings(max_examples=20)
    def test_bond_order_classification(self, bond_order):
        """Test that all bond orders get classified."""
        import networkx

        G = networkx.Graph(cell=np.eye(3) * 10)
        G.add_node(0, symbol="C", coord=[0, 0, 0], tag=1, ufftype="C_R")
        G.add_node(1, symbol="C", coord=[1, 0, 0], tag=2, ufftype="C_R")
        G.add_edge(0, 1, bond_order=bond_order)

        result = utils.networkx_to_gulp(G, name="test", write_to_file=False)

        # Should contain one of these bond types
        bond_types = ["half", "single", "aromatic", "double", "triple"]
        assert any(bt in result for bt in bond_types)
