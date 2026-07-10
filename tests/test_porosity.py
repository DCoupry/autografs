"""Tests for the geometric porosity descriptors.

MOF-5 (built from the committed fixture) is the reference: its
experimental descriptors are well known (density ~0.6 g/cm3, LCD
~15 A, PLD ~8 A), so coarse-grid values must land in generous windows
around them and obey the exact identities (PLD <= LCD, void fraction
monotone in probe radius).
"""

import os

import networkx
import numpy as np
import pytest

from autografs.framework import Framework

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "topologies_fixture.json"
)

# coarse grid: fast, and the assertion windows account for it
SPACING = 0.5


@pytest.fixture(scope="module")
def mofgen():
    from autografs import Autografs

    return Autografs(topofile=FIXTURE_PATH)


@pytest.fixture(scope="module")
def mof5(mofgen):
    topology = mofgen.topologies["pcu"]
    mappings = {}
    for key in topology.mappings:
        conn = len(key.atoms.indices_from_symbol("X"))
        mappings[key] = {6: "Zn_mof5_octahedral", 2: "Benzene_linear"}[conn]
    return mofgen.build(topology, mappings=mappings, refine_cell=True, max_rmsd=0.5)


@pytest.fixture(scope="module")
def dense():
    """A dense fcc-like metal: no pores at all."""
    cell = np.eye(3) * 3.6
    graph = networkx.Graph(cell=cell)
    positions = [[0, 0, 0], [1.8, 1.8, 0], [1.8, 0, 1.8], [0, 1.8, 1.8]]
    for i, coord in enumerate(positions):
        graph.add_node(
            i, symbol="Cu", coord=np.array(coord, dtype=float), tag=0, ufftype="Cu3+1"
        )
    for i in range(3):
        graph.add_edge(i, i + 1, bond_order=1.0)
    return Framework(graph, name="fcc_copper")


class TestDensity:
    def test_mof5_density(self, mof5):
        # experimental MOF-5: ~0.59-0.62 g/cm3
        assert 0.45 < mof5.density < 0.8

    def test_dense_metal_density(self, dense):
        # fcc copper: ~8.9 g/cm3
        assert 7.0 < dense.density < 11.0


class TestVoidFraction:
    def test_mof5_is_mostly_empty(self, mof5):
        vf = mof5.void_fraction(spacing=SPACING)
        assert 0.5 < vf < 0.95

    def test_monotone_in_probe_radius(self, mof5):
        geometric = mof5.void_fraction(probe_radius=0.0, spacing=SPACING)
        helium = mof5.void_fraction(probe_radius=1.2, spacing=SPACING)
        nitrogen = mof5.void_fraction(probe_radius=1.86, spacing=SPACING)
        assert geometric > helium > nitrogen > 0.0

    def test_dense_metal_has_no_void(self, dense):
        assert dense.void_fraction(spacing=0.3) < 0.05


class TestPoreDiameters:
    def test_mof5_lcd(self, mof5):
        # literature LCD for MOF-5: ~15.1 A
        lcd = mof5.largest_cavity_diameter(spacing=SPACING)
        assert 11.0 < lcd < 18.0

    def test_mof5_pld(self, mof5):
        # literature PLD for MOF-5: ~7.8-8.0 A
        pld = mof5.pore_limiting_diameter(spacing=SPACING)
        assert 5.5 < pld < 10.5

    def test_pld_never_exceeds_lcd(self, mof5):
        lcd = mof5.largest_cavity_diameter(spacing=SPACING)
        pld = mof5.pore_limiting_diameter(spacing=SPACING)
        assert pld <= lcd

    def test_dense_metal_has_no_pores(self, dense):
        assert dense.largest_cavity_diameter(spacing=0.3) == 0.0
        assert dense.pore_limiting_diameter(spacing=0.3) == 0.0
