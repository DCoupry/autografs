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


@pytest.fixture
def layer():
    """A hand-built flat layer in a padded slab, like a 2D build."""
    import networkx

    from autografs.framework import Framework

    cell = np.diag([5.0, 5.0, 10.0])
    graph = networkx.Graph(cell=cell)
    atoms = [
        ("C", [0.0, 0.0, 0.0], 1),
        ("C", [1.5, 0.0, 0.1], 2),
        ("O", [3.0, 0.0, -0.1], 0),
    ]
    for i, (symbol, coord, tag) in enumerate(atoms):
        graph.add_node(
            i, symbol=symbol, coord=np.array(coord), tag=tag, ufftype=f"{symbol}_R"
        )
    graph.add_edge(0, 1, bond_order=1.0)
    graph.add_edge(1, 2, bond_order=1.0)
    return Framework(graph, name="layer")


class TestStacking:
    def test_aa_replaces_c_with_interlayer(self, layer):
        cof = layer.stack(mode="AA", interlayer=3.35)
        assert len(cof) == len(layer)
        assert cof.graph.number_of_edges() == layer.graph.number_of_edges()
        np.testing.assert_allclose(cof.cell[2], [0.0, 0.0, 3.35])
        np.testing.assert_allclose(cof.cell[:2], layer.cell[:2])
        # the input framework is untouched
        assert layer.cell[2, 2] == 10.0
        assert cof.name == "layer_AA"

    def test_ab_doubles_layer_with_offset(self, layer):
        cof = layer.stack(mode="AB", interlayer=3.0)
        assert len(cof) == 2 * len(layer)
        assert cof.graph.number_of_edges() == 2 * layer.graph.number_of_edges()
        np.testing.assert_allclose(cof.cell[2], [0.0, 0.0, 6.0])
        # second layer sits at the AB offset (1/3, 2/3) plus interlayer
        shift = layer.cell[0] / 3.0 + 2.0 * layer.cell[1] / 3.0 + [0, 0, 3.0]
        n = len(layer)
        for i in range(n):
            np.testing.assert_allclose(
                cof.graph.nodes[i + n]["coord"],
                layer.graph.nodes[i]["coord"] + shift,
            )
            assert cof.graph.nodes[i + n]["symbol"] == layer.graph.nodes[i]["symbol"]

    def test_no_interlayer_bonds(self, layer):
        cof = layer.stack(mode="AB")
        n = len(layer)
        for i, j in cof.graph.edges():
            assert (i < n) == (j < n)

    def test_custom_offset(self, layer):
        cof = layer.stack(mode="serrated", interlayer=3.5, offset=(0.5, 0.0))
        n = len(layer)
        shift = 0.5 * layer.cell[0] + [0, 0, 3.5]
        np.testing.assert_allclose(
            cof.graph.nodes[n]["coord"], layer.graph.nodes[0]["coord"] + shift
        )

    def test_second_layer_tags_stay_unique(self, layer):
        cof = layer.stack(mode="staggered")
        tags = [
            data["tag"] for _, data in cof.graph.nodes(data=True) if data["tag"] > 0
        ]
        assert len(tags) == len(set(tags))

    def test_stacked_structure_wraps_cleanly(self, layer):
        structure = layer.stack(mode="AB").structure
        assert structure.frac_coords.min() >= 0.0
        assert structure.frac_coords.max() < 1.0

    def test_bad_arguments_rejected(self, layer):
        with pytest.raises(ValueError, match="Unknown stacking mode"):
            layer.stack(mode="ABC")
        with pytest.raises(ValueError, match="offset"):
            layer.stack(mode="AA", offset=(0.5, 0.5))
        with pytest.raises(ValueError, match="positive"):
            layer.stack(interlayer=-1.0)

    def test_non_layered_input_rejected(self, mof5):
        from autografs.exceptions import StackingError

        with pytest.raises(StackingError, match="not a 2D layer"):
            mof5.stack()

    def test_skewed_c_axis_rejected(self, layer):
        from autografs.exceptions import StackingError
        from autografs.framework import Framework

        skewed = layer.graph.copy()
        skewed.graph["cell"] = np.array([[5.0, 0, 0], [0, 5.0, 0], [2.0, 0, 10.0]])
        with pytest.raises(StackingError, match="perpendicular"):
            Framework(skewed, name="skewed").stack()
