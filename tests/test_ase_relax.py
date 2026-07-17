"""
Tests for the ASE-calculator relaxation bridge (autografs.ase_relax).

The bridge machinery is exercised end-to-end with ASE's pure-python
Lennard-Jones calculator (energy, forces and stress, no external
binaries), so the graph-preservation and unit-conversion guarantees
are tested everywhere. The heavyweight backends (xtb, tblite, DFTB+)
are covered by name-dispatch and error-path tests; their imports are
made to fail deterministically instead of depending on what happens
to be installed.
"""

import sys

import networkx
import numpy as np
import pytest

from autografs.ase_relax import (
    EV_TO_KCAL_PER_MOL,
    make_calculator,
    relax_framework_ase,
)
from autografs.exceptions import RelaxationError
from autografs.framework import Framework

SIGMA = 2.5  # LJ minimum at 2^(1/6) * sigma ~ 2.81 A


def _pair_framework(distance, cell_length=30.0):
    """Two bonded carbons at the given separation in a cubic cell."""
    graph = networkx.Graph(cell=np.diag([cell_length] * 3))
    graph.add_node(0, symbol="C", coord=np.array([5.0, 5.0, 5.0]), tag=0, ufftype="C_R")
    graph.add_node(
        1, symbol="C", coord=np.array([5.0 + distance, 5.0, 5.0]), tag=1, ufftype="C_R"
    )
    graph.add_edge(0, 1, bond_order=1.0)
    return Framework(graph, name="pair")


def _lj():
    from ase.calculators.lj import LennardJones

    return LennardJones(sigma=SIGMA, epsilon=0.5, rc=3.0 * SIGMA)


class TestRelaxWithCalculatorInstance:
    def test_pair_relaxes_to_lj_minimum(self):
        framework = _pair_framework(distance=2.2)
        relaxed = relax_framework_ase(
            framework, _lj(), relax_cell=False, fmax=1e-4, steps=500
        )
        d = np.linalg.norm(relaxed.cart_coords[1] - relaxed.cart_coords[0])
        assert d == pytest.approx(2 ** (1 / 6) * SIGMA, abs=1e-2)

    def test_graph_preserved_and_input_untouched(self):
        framework = _pair_framework(distance=2.2)
        before = framework.cart_coords.copy()
        relaxed = relax_framework_ase(
            framework, _lj(), relax_cell=False, fmax=1e-3, steps=200
        )
        assert sorted(relaxed.graph) == sorted(framework.graph)
        assert relaxed.symbols == framework.symbols
        assert list(relaxed.graph.edges(data=True)) == list(
            framework.graph.edges(data=True)
        )
        np.testing.assert_allclose(framework.cart_coords, before)
        assert framework.energy is None

    def test_energy_converted_to_kcal(self):
        framework = _pair_framework(distance=2.2)
        relaxed = relax_framework_ase(
            framework, _lj(), relax_cell=False, fmax=1e-4, steps=500
        )
        atoms = relaxed.to_ase()
        atoms.calc = _lj()
        expected = atoms.get_potential_energy() * EV_TO_KCAL_PER_MOL
        assert relaxed.energy == pytest.approx(expected, rel=1e-6)

    def test_cell_relaxation_returns_valid_cell(self):
        framework = _pair_framework(distance=2.8)
        relaxed = relax_framework_ase(
            framework, _lj(), relax_cell=True, fmax=0.05, steps=50
        )
        assert relaxed.cell.shape == (3, 3)
        assert np.linalg.det(relaxed.cell) > 0

    def test_framework_relax_dispatches_calculator(self):
        framework = _pair_framework(distance=2.2)
        relaxed = framework.relax(calculator=_lj(), relax_cell=False, fmax=1e-3)
        d = np.linalg.norm(relaxed.cart_coords[1] - relaxed.cart_coords[0])
        assert d == pytest.approx(2 ** (1 / 6) * SIGMA, abs=0.05)

    def test_nonconvergence_warns_not_raises(self, caplog):
        framework = _pair_framework(distance=2.2)
        with caplog.at_level("WARNING"):
            relaxed = relax_framework_ase(
                framework, _lj(), relax_cell=False, fmax=1e-12, steps=2
            )
        assert relaxed is not None
        assert any("did not reach" in message for message in caplog.messages)


class TestMakeCalculator:
    def test_gfn2_rejected_for_periodic(self):
        with pytest.raises(RelaxationError, match="no periodic"):
            make_calculator("gfn2")

    def test_unknown_name_lists_options(self):
        with pytest.raises(RelaxationError, match="gfn-ff"):
            make_calculator("b3lyp")

    def test_missing_tblite_hint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "tblite", None)
        monkeypatch.setitem(sys.modules, "tblite.ase", None)
        with pytest.raises(RelaxationError, match="pip install tblite"):
            make_calculator("gfn1")

    def test_missing_xtb_hint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "xtb", None)
        monkeypatch.setitem(sys.modules, "xtb.ase.calculator", None)
        with pytest.raises(RelaxationError, match="xtb-python"):
            make_calculator("gfn-ff")

    def test_string_dispatch_through_relax(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "tblite", None)
        monkeypatch.setitem(sys.modules, "tblite.ase", None)
        framework = _pair_framework(distance=2.2)
        with pytest.raises(RelaxationError, match="tblite"):
            framework.relax(calculator="gfn1")
