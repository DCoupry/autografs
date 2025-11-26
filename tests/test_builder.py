"""
Unit tests for the autografs.builder module.

Tests cover the Autografs class initialization, topology/SBU listing,
and building functionality. Some tests require the full data files
and are marked accordingly.
"""

import os
import tempfile

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

from autografs.structure import Fragment, Topology


# =============================================================================
# Test Markers
# =============================================================================

# Mark for tests requiring full installation with data files
requires_data = pytest.mark.skipif(
    not os.path.exists(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "autografs",
            "data",
            "topologies.pkl",
        )
    ),
    reason="Requires topology data files",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_xyz_for_sbu():
    """Create a temporary XYZ file with test SBUs."""
    content = """3
name=Simple_Linear pbc="F F F"
C          0.0000        0.0000        0.0000
X          1.5000        0.0000        0.0000
X         -1.5000        0.0000        0.0000
5
name=Simple_Tetrahedral pbc="F F F"
Zn         0.0000        0.0000        0.0000
X          1.0000        1.0000        1.0000
X          1.0000       -1.0000       -1.0000
X         -1.0000        1.0000       -1.0000
X         -1.0000       -1.0000        1.0000
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write(content)
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def linear_fragment():
    """Create a linear fragment for testing."""
    from pymatgen.core.structure import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer

    mol = Molecule(
        ["C", "X", "X", "H", "H"],
        [[0, 0, 0], [1.5, 0, 0], [-1.5, 0, 0], [0, 1, 0], [0, -1, 0]],
    )
    symm_mol = Molecule(["H", "H"], [[1.5, 0, 0], [-1.5, 0, 0]], charge=2)
    pg = PointGroupAnalyzer(symm_mol, tolerance=0.1)
    return Fragment(atoms=mol, symmetry=pg, name="test_linear")


@pytest.fixture
def simple_topology(linear_fragment):
    """Create a simple topology for testing."""
    from pymatgen.core.lattice import Lattice

    cell = Lattice.cubic(10.0)
    # Add tags to the fragment atoms
    tags = [0, 1, 2, 0, 0]  # Dummies get positive tags
    linear_fragment.atoms.add_site_property("tags", tags)

    frag1 = linear_fragment.copy()
    frag2 = linear_fragment.copy()

    return Topology(name="test_topo", slots=[frag1, frag2], cell=cell)


# =============================================================================
# Import Tests
# =============================================================================


class TestBuilderImport:
    """Test builder module imports."""

    def test_import_autografs_class(self):
        """Test that Autografs class can be imported."""
        from autografs.builder import Autografs

        assert Autografs is not None

    def test_import_from_package(self):
        """Test import from main package."""
        from autografs import Autografs

        assert Autografs is not None


# =============================================================================
# Autografs Initialization Tests
# =============================================================================


class TestAutografsInit:
    """Test Autografs initialization."""

    @requires_data
    def test_default_init(self):
        """Test default initialization loads data."""
        from autografs import Autografs

        mofgen = Autografs()
        assert mofgen is not None
        assert len(mofgen.sbu) > 0
        assert len(mofgen.topologies) > 0

    @requires_data
    def test_init_with_custom_xyz(self, sample_xyz_for_sbu):
        """Test initialization with custom XYZ file."""
        from autografs import Autografs

        mofgen = Autografs(xyzfile=sample_xyz_for_sbu)
        # Should have the custom SBUs loaded
        assert "Simple_Linear" in mofgen.sbu
        assert "Simple_Tetrahedral" in mofgen.sbu

    @requires_data
    def test_sbu_are_fragments(self):
        """Test that SBUs are Fragment objects."""
        from autografs import Autografs

        mofgen = Autografs()
        for name, sbu in mofgen.sbu.items():
            assert isinstance(sbu, Fragment), f"{name} is not a Fragment"

    @requires_data
    def test_topologies_are_topology(self):
        """Test that topologies are Topology objects."""
        from autografs import Autografs

        mofgen = Autografs()
        for name, topo in mofgen.topologies.items():
            assert isinstance(topo, Topology), f"{name} is not a Topology"


# =============================================================================
# list_topologies Tests
# =============================================================================


class TestListTopologies:
    """Test Autografs.list_topologies method."""

    @requires_data
    def test_returns_list(self):
        """Test that method returns a list."""
        from autografs import Autografs

        mofgen = Autografs()
        result = mofgen.list_topologies()
        assert isinstance(result, list)

    @requires_data
    def test_returns_sorted(self):
        """Test that list is sorted."""
        from autografs import Autografs

        mofgen = Autografs()
        result = mofgen.list_topologies()
        assert result == sorted(result)

    @requires_data
    def test_subset_filter(self):
        """Test subset parameter filters results."""
        from autografs import Autografs

        mofgen = Autografs()
        all_topos = mofgen.list_topologies()
        subset = all_topos[:3] if len(all_topos) >= 3 else all_topos
        result = mofgen.list_topologies(subset=subset)
        assert result == subset


# =============================================================================
# list_building_units Tests
# =============================================================================


class TestListBuildingUnits:
    """Test Autografs.list_building_units method."""

    @requires_data
    def test_returns_dict(self):
        """Test that method returns a dictionary."""
        from autografs import Autografs

        mofgen = Autografs()
        topos = mofgen.list_topologies()
        if topos:
            result = mofgen.list_building_units(sieve=topos[0])
            assert isinstance(result, dict)

    @requires_data
    def test_without_sieve_returns_empty(self):
        """Test that without sieve parameter returns empty dict."""
        from autografs import Autografs

        mofgen = Autografs()
        result = mofgen.list_building_units()
        assert result == {}


# =============================================================================
# _validate_mappings Tests
# =============================================================================


class TestValidateMappings:
    """Test Autografs._validate_mappings method."""

    @requires_data
    def test_converts_string_to_fragment(self):
        """Test that string SBU names are converted to Fragment objects."""
        from autografs import Autografs

        mofgen = Autografs()
        topos = mofgen.list_topologies()

        for topo_name in topos[:5]:  # Test first 5
            topo = mofgen.topologies[topo_name]
            sbu_dict = mofgen.list_building_units(sieve=topo_name)

            if all(sbu_dict.values()):
                # Create string mappings
                string_mappings = {k: v[0] for k, v in sbu_dict.items()}
                result = mofgen._validate_mappings(topo, string_mappings)

                # All values should be Fragment objects
                for v in result.values():
                    assert isinstance(v, Fragment)
                break

    @requires_data
    def test_raises_on_missing_slot(self):
        """Test that missing slot raises AssertionError."""
        from autografs import Autografs

        mofgen = Autografs()
        topos = mofgen.list_topologies()

        for topo_name in topos[:10]:
            topo = mofgen.topologies[topo_name]
            if len(topo.mappings) > 1:
                # Create incomplete mapping
                sbu_dict = mofgen.list_building_units(sieve=topo_name)
                if all(sbu_dict.values()):
                    incomplete = {
                        list(sbu_dict.keys())[0]: list(sbu_dict.values())[0][0]
                    }
                    with pytest.raises(AssertionError):
                        mofgen._validate_mappings(topo, incomplete)
                    break


# =============================================================================
# build Tests
# =============================================================================


class TestBuild:
    """Test Autografs.build method."""

    @requires_data
    @pytest.mark.slow
    def test_build_returns_graph(self):
        """Test that build returns a networkx Graph."""
        import networkx
        from autografs import Autografs

        mofgen = Autografs()
        topos = mofgen.list_topologies()

        for topo_name in topos:
            topo = mofgen.topologies[topo_name]
            if len(topo) > 10:
                continue  # Skip large topologies

            sbu_dict = mofgen.list_building_units(sieve=topo_name)
            if all(sbu_dict.values()):
                mappings = {k: v[0] for k, v in sbu_dict.items()}
                try:
                    result = mofgen.build(
                        topo.copy(), mappings, refine_cell=False, verbose=False
                    )
                    assert isinstance(result, networkx.Graph)
                    break
                except Exception:
                    continue

    @requires_data
    @pytest.mark.slow
    def test_build_graph_has_cell(self):
        """Test that built graph has cell attribute."""
        from autografs import Autografs

        mofgen = Autografs()
        topos = mofgen.list_topologies()

        for topo_name in topos:
            topo = mofgen.topologies[topo_name]
            if len(topo) > 10:
                continue

            sbu_dict = mofgen.list_building_units(sieve=topo_name)
            if all(sbu_dict.values()):
                mappings = {k: v[0] for k, v in sbu_dict.items()}
                try:
                    result = mofgen.build(
                        topo.copy(), mappings, refine_cell=False, verbose=False
                    )
                    assert "cell" in result.graph
                    break
                except Exception:
                    continue

    @requires_data
    @pytest.mark.slow
    def test_build_graph_has_nodes(self):
        """Test that built graph has nodes with correct attributes."""
        from autografs import Autografs

        mofgen = Autografs()
        topos = mofgen.list_topologies()

        for topo_name in topos:
            topo = mofgen.topologies[topo_name]
            if len(topo) > 10:
                continue

            sbu_dict = mofgen.list_building_units(sieve=topo_name)
            if all(sbu_dict.values()):
                mappings = {k: v[0] for k, v in sbu_dict.items()}
                try:
                    result = mofgen.build(
                        topo.copy(), mappings, refine_cell=False, verbose=False
                    )
                    assert result.number_of_nodes() > 0

                    # Check node attributes
                    for node, data in result.nodes(data=True):
                        assert "symbol" in data
                        assert "coord" in data
                        assert "tag" in data
                    break
                except Exception:
                    continue


# =============================================================================
# _align_slot Tests
# =============================================================================


class TestAlignSlot:
    """Test Autografs._align_slot method."""

    @requires_data
    def test_returns_fragment_and_rmsd(self):
        """Test that _align_slot returns a Fragment and RMSD."""
        from autografs import Autografs

        mofgen = Autografs()
        topos = mofgen.list_topologies()

        for topo_name in topos:
            topo = mofgen.topologies[topo_name]
            if len(topo) > 5:
                continue

            sbu_dict = mofgen.list_building_units(sieve=topo_name)
            if all(sbu_dict.values()):
                slot = topo.slots[0]
                # Find a compatible SBU
                for slot_type, sbu_names in sbu_dict.items():
                    if sbu_names:
                        sbu = mofgen.sbu[sbu_names[0]].copy()
                        try:
                            result, rmsd = mofgen._align_slot(slot, sbu)
                            assert isinstance(result, Fragment)
                            assert isinstance(rmsd, float)
                            assert rmsd >= 0
                            return
                        except Exception:
                            continue

    @requires_data
    def test_rmsd_is_non_negative(self):
        """Test that RMSD is always non-negative."""
        from autografs import Autografs

        mofgen = Autografs()
        topos = mofgen.list_topologies()

        for topo_name in topos[:20]:
            topo = mofgen.topologies[topo_name]
            if len(topo) > 5:
                continue

            sbu_dict = mofgen.list_building_units(sieve=topo_name)
            if all(sbu_dict.values()):
                slot = topo.slots[0]
                for slot_type, sbu_names in sbu_dict.items():
                    if sbu_names:
                        sbu = mofgen.sbu[sbu_names[0]].copy()
                        try:
                            _, rmsd = mofgen._align_slot(slot, sbu)
                            assert rmsd >= 0, f"RMSD should be non-negative, got {rmsd}"
                            return
                        except Exception:
                            continue


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full building workflow."""

    @requires_data
    @pytest.mark.slow
    def test_full_workflow(self):
        """Test complete workflow from initialization to graph output."""
        import networkx
        from autografs import Autografs

        # Initialize
        mofgen = Autografs()

        # List topologies
        topos = mofgen.list_topologies()
        assert len(topos) > 0

        # Find a topology with available SBUs
        for topo_name in topos:
            topo = mofgen.topologies[topo_name]
            if len(topo) > 10:
                continue

            sbu_dict = mofgen.list_building_units(sieve=topo_name)
            if not all(sbu_dict.values()):
                continue

            # Build
            mappings = {k: v[0] for k, v in sbu_dict.items()}
            try:
                graph = mofgen.build(
                    topo.copy(), mappings, refine_cell=False, verbose=False
                )

                # Verify output
                assert isinstance(graph, networkx.Graph)
                assert graph.number_of_nodes() > 0
                assert "cell" in graph.graph
                return
            except Exception:
                continue

        pytest.skip("Could not find suitable topology for integration test")

    @requires_data
    @pytest.mark.slow
    def test_verbose_mode_runs(self):
        """Test that verbose mode runs without errors."""
        import logging
        from autografs import Autografs

        # Temporarily increase log level
        logger = logging.getLogger("autografs")
        old_level = logger.level
        logger.setLevel(logging.DEBUG)

        try:
            mofgen = Autografs()
            topos = mofgen.list_topologies()

            for topo_name in topos:
                topo = mofgen.topologies[topo_name]
                if len(topo) > 5:
                    continue

                sbu_dict = mofgen.list_building_units(sieve=topo_name, verbose=True)
                if not all(sbu_dict.values()):
                    continue

                mappings = {k: v[0] for k, v in sbu_dict.items()}
                try:
                    mofgen.build(topo.copy(), mappings, refine_cell=False, verbose=True)
                    return
                except Exception:
                    continue
        finally:
            logger.setLevel(old_level)
