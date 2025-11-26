"""
Basic tests for AuToGraFS package.
"""

import pytest


def test_import():
    """Test that the package can be imported."""
    import autografs

    assert hasattr(autografs, "__version__")


def test_version():
    """Test that version is defined."""
    from autografs import __version__

    assert __version__ == "3.0.0"


def test_submodules_importable():
    """Test that submodules can be imported."""
    from autografs import utils
    from autografs import structure
    from autografs import builder

    assert utils is not None
    assert structure is not None
    assert builder is not None


class TestAutografsInit:
    """Test Autografs initialization."""

    @pytest.mark.skip(reason="Requires full installation with data files")
    def test_autografs_init(self):
        """Test that Autografs can be initialized."""
        from autografs import Autografs

        mofgen = Autografs()
        assert mofgen is not None
