"""
Tests for the Framework result class (autografs.framework).

Uses the committed 3-net fixture to build a real MOF-5 prototype and
verifies every export path.
"""

import os

import numpy as np
import pytest

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "topologies_fixture.json"
)


@pytest.fixture(scope="module")
def mof5():
    """A MOF-5 prototype framework built from the fixture."""
    from autografs import Autografs

    mofgen = Autografs(topofile=FIXTURE_PATH)
    topology = mofgen.topologies["pcu"]
    mappings = {}
    for key in topology.mappings:
        conn = len(key.atoms.indices_from_symbol("X"))
        mappings[key] = "Zn_mof5_octahedral" if conn == 6 else "Benzene_linear"
    return mofgen.build(topology, mappings=mappings, refine_cell=True)


class TestFrameworkViews:
    def test_len_and_repr(self, mof5):
        assert len(mof5) == 53
        text = repr(mof5)
        assert "pcu" in text and "Zn" in text

    def test_cell_and_lattice(self, mof5):
        assert mof5.cell.shape == (3, 3)
        assert 12.0 < mof5.lattice.a < 13.5

    def test_node_order_consistency(self, mof5):
        assert len(mof5.symbols) == len(mof5)
        assert mof5.cart_coords.shape == (len(mof5), 3)
        assert len(mof5.mmtypes) == len(mof5)
        # every mmtype starts with its element symbol
        for symbol, mmtype in zip(mof5.symbols, mof5.mmtypes, strict=True):
            assert mmtype.startswith(symbol[0])

    def test_bonds(self, mof5):
        bonds = mof5.bonds
        assert len(bonds) == mof5.graph.number_of_edges()
        for i, j, order in bonds:
            assert 0 <= i < j < len(mof5)
            assert order > 0


class TestStructureExport:
    def test_structure_matches_graph(self, mof5):
        structure = mof5.structure
        assert len(structure) == len(mof5)
        assert [site.specie.symbol for site in structure] == mof5.symbols

    def test_structure_is_wrapped(self, mof5):
        frac = mof5.structure.frac_coords
        assert frac.min() >= 0.0
        assert frac.max() < 1.0

    def test_site_properties_carried(self, mof5):
        props = mof5.structure.site_properties
        assert "tags" in props
        assert "ufftype" in props


class TestFileExports:
    def test_write_cif_roundtrip(self, mof5, tmp_path):
        from pymatgen.core.structure import Structure

        path = mof5.write_cif(tmp_path / "mof5.cif")
        assert path.exists()
        reread = Structure.from_file(str(path))
        assert len(reread) == len(mof5)
        assert reread.composition == mof5.structure.composition
        np.testing.assert_allclose(
            sorted(reread.lattice.abc), sorted(mof5.lattice.abc), rtol=1e-4
        )

    def test_to_ase(self, mof5):
        atoms = mof5.to_ase()
        assert len(atoms) == len(mof5)
        assert atoms.pbc.all()
        np.testing.assert_allclose(np.asarray(atoms.cell), mof5.cell, rtol=1e-8)
        assert atoms.get_chemical_symbols() == mof5.symbols

    def test_to_gulp(self, mof5):
        gulp = mof5.to_gulp(write_to_file=False)
        assert "vectors" in gulp
        assert "connect" in gulp
        assert "library uff4mof" in gulp
