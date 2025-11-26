"""
Basic tests for the AuToGraFS package.

This module contains unit tests verifying package imports, version
information, and basic module accessibility.
"""

import pytest


class TestPackageImports:
    """Test package import functionality."""

    def test_import_package(self):
        """Test that the package can be imported."""
        import autografs

        assert hasattr(autografs, "__version__")

    def test_version_defined(self):
        """Test that version is defined and valid."""
        from autografs import __version__

        assert __version__ == "3.0.0"
        assert isinstance(__version__, str)

    def test_version_format(self):
        """Test version follows semantic versioning format."""
        from autografs import __version__

        parts = __version__.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)


class TestSubmoduleImports:
    """Test that submodules can be imported."""

    def test_utils_import(self):
        """Test utils module import."""
        from autografs import utils

        assert utils is not None

    def test_structure_import(self):
        """Test structure module import."""
        from autografs import structure

        assert structure is not None

    def test_builder_import(self):
        """Test builder module import."""
        from autografs import builder

        assert builder is not None


class TestClassImports:
    """Test that main classes can be imported."""

    def test_autografs_class(self):
        """Test Autografs class import."""
        from autografs import Autografs

        assert Autografs is not None

    def test_fragment_class(self):
        """Test Fragment class import."""
        from autografs import Fragment

        assert Fragment is not None

    def test_topology_class(self):
        """Test Topology class import."""
        from autografs import Topology

        assert Topology is not None


class TestExports:
    """Test that __all__ exports are accessible."""

    def test_all_exports(self):
        """Test all items in __all__ are accessible."""
        import autografs

        for name in autografs.__all__:
            assert hasattr(autografs, name), f"{name} not accessible from package"


class TestMetadata:
    """Test package metadata."""

    def test_author_defined(self):
        """Test author is defined."""
        from autografs import __author__

        assert __author__ is not None
        assert isinstance(__author__, str)

    def test_license_defined(self):
        """Test license is defined."""
        from autografs import __license__

        assert __license__ == "MIT"

    def test_status_defined(self):
        """Test status is defined."""
        from autografs import __status__

        assert __status__ == "production"


class TestAutografsInit:
    """Test Autografs initialization."""

    @pytest.mark.skip(reason="Requires full installation with data files")
    def test_autografs_init(self):
        """Test that Autografs can be initialized."""
        from autografs import Autografs

        mofgen = Autografs()
        assert mofgen is not None
