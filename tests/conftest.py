"""
Pytest configuration and shared fixtures for AuToGraFS tests.

This module provides shared fixtures and configuration for all test modules.
"""

import os
import tempfile

import pytest


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless explicitly requested."""
    if config.getoption("-m"):
        # If markers are explicitly specified, respect them
        return

    skip_slow = pytest.mark.skip(reason="use -m slow to run slow tests")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# =============================================================================
# Shared Fixtures
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
5
name=Test_Tetrahedral pbc="F F F"
Zn         0.0000        0.0000        0.0000
X          1.0000        1.0000        1.0000
X          1.0000       -1.0000       -1.0000
X         -1.0000        1.0000       -1.0000
X         -1.0000       -1.0000        1.0000
"""


@pytest.fixture
def temp_xyz_file(sample_xyz_content, tmp_path):
    """Create a temporary XYZ file for testing."""
    xyz_file = tmp_path / "test_sbus.xyz"
    xyz_file.write_text(sample_xyz_content)
    return str(xyz_file)


@pytest.fixture
def linear_molecule():
    """Create a simple linear molecule with 2 dummies."""
    from pymatgen.core.structure import Molecule

    return Molecule(
        ["C", "X", "X", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [-1.5, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
    )


@pytest.fixture
def linear_fragment(linear_molecule):
    """Create a linear Fragment with 2 dummies."""
    from pymatgen.core.structure import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer

    from autografs.fragment import Fragment

    # Create symmetry from dummies only
    symm_mol = Molecule(["H", "H"], [[1.5, 0.0, 0.0], [-1.5, 0.0, 0.0]], charge=2)
    pg = PointGroupAnalyzer(symm_mol, tolerance=0.1)
    return Fragment(atoms=linear_molecule, symmetry=pg, name="test_linear")


@pytest.fixture
def trigonal_fragment():
    """Create a trigonal planar fragment with 3 dummies."""
    from pymatgen.core.structure import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer

    from autografs.fragment import Fragment

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
    """Create a tetrahedral fragment with 4 dummies."""
    from pymatgen.core.structure import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer

    from autografs.fragment import Fragment

    coords = [
        [0.0, 0.0, 0.0],  # Central Zn
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
    import numpy as np

    from autografs.topology import Topology

    cell = np.eye(3) * 10.0
    # Add tags to the fragment atoms
    tags = [0, 1, 2, 0, 0]
    linear_fragment.atoms.add_site_property("tags", tags)

    frag1 = linear_fragment.copy()
    frag2 = linear_fragment.copy()
    return Topology(name="test_topo", slots=[frag1, frag2], cell=cell)


@pytest.fixture
def data_files_available():
    """Check if topology data files are available."""
    import autografs.data

    topo_path = os.path.join(autografs.data.__path__[0], "topologies.pkl")
    return os.path.exists(topo_path)
