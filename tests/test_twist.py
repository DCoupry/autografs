"""
Tests for commensurate twist search and moiré bilayers (autografs.twist).

The angle search is validated against the two closed-form CSL
families: hexagonal lattices (cos theta = (m^2 + 4mn + n^2) /
(2 (m^2 + mn + n^2)), sigma = m^2 + mn + n^2) and square lattices
(theta = 2 atan(n/m), sigma = m^2 + n^2). The search itself never
uses these formulas — it enumerates coincidence vector pairs — so the
agreement is an independent check. Builder tests run on a small
synthetic square layer.
"""

import math

import networkx
import numpy as np
import pytest

from autografs.exceptions import StackingError
from autografs.framework import Framework
from autografs.twist import commensurate_twists, twisted_bilayer


def _hexagonal_cell(a=10.0, pad=15.0):
    return np.array(
        [
            [a, 0.0, 0.0],
            [a / 2.0, a * math.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, pad],
        ]
    )


def _square_cell(a=5.0, pad=10.0):
    return np.diag([a, a, pad])


def _hex_family_angle(m, n):
    """The closed-form hexagonal CSL angle (degrees)."""
    sigma = m * m + m * n + n * n
    return math.degrees(math.acos((m * m + 4 * m * n + n * n) / (2.0 * sigma)))


def _family_match(candidates, angle, n_cells, symmetry_angle):
    """True when some candidate matches the angle modulo the lattice's
    own rotational symmetry (theta and symmetry - theta are the same
    moiré family, mirrored)."""
    for c in candidates:
        folded = min(
            c.angle % symmetry_angle, symmetry_angle - c.angle % symmetry_angle
        )
        target = min(angle, symmetry_angle - angle)
        if abs(folded - target) < 1e-6 and c.n_cells == n_cells:
            return True
    return False


def _square_layer(a=5.0, pad=10.0):
    """One-carbon square layer: the smallest twistable framework."""
    graph = networkx.Graph(cell=_square_cell(a, pad))
    graph.add_node(0, symbol="C", coord=np.array([0.0, 0.0, 0.0]), tag=0, ufftype="C_R")
    return Framework(graph, name="sqlayer")


class TestCommensurateSearch:
    def test_hexagonal_family_recovered(self):
        candidates = commensurate_twists(_hexagonal_cell(), max_index=4)
        for m, n in [(1, 2), (1, 3)]:
            angle = _hex_family_angle(m, n)
            sigma = m * m + m * n + n * n
            assert _family_match(candidates, angle, sigma, 60.0), (m, n, angle)

    def test_hexagonal_higher_sigma_needs_larger_index(self):
        # sigma-19's second independent coincidence vector has index 5:
        # bounded enumeration finds the family only once max_index
        # covers it
        angle = _hex_family_angle(2, 3)
        small = commensurate_twists(_hexagonal_cell(), max_index=4)
        assert not _family_match(small, angle, 19, 60.0)
        large = commensurate_twists(_hexagonal_cell(), max_index=6)
        assert _family_match(large, angle, 19, 60.0)

    def test_square_family_recovered(self):
        # opposite-parity (m, n): sigma = m^2 + n^2, theta = 2 atan(n/m)
        candidates = commensurate_twists(_square_cell(), max_index=4)
        for m, n in [(2, 1), (3, 2), (4, 1)]:
            angle = math.degrees(2.0 * math.atan2(n, m))
            sigma = m * m + n * n
            assert _family_match(candidates, angle, sigma, 90.0), (m, n, angle)

    def test_square_both_odd_family(self):
        # both-odd (m, n) halve: (3, 1) is the sigma-5 pair at 36.87
        candidates = commensurate_twists(_square_cell(), max_index=4)
        assert _family_match(candidates, math.degrees(2.0 * math.atan2(1, 3)), 5, 90.0)

    def test_exact_candidates_have_zero_strain(self):
        for candidate in commensurate_twists(_hexagonal_cell(), max_index=3):
            assert candidate.strain < 1e-9
            # the transform is then a pure rotation: orthogonal,
            # determinant +1
            t = candidate.transform
            np.testing.assert_allclose(t @ t.T, np.eye(2), atol=1e-9)
            assert np.linalg.det(t) == pytest.approx(1.0)

    def test_supercell_maps_both_lattices(self):
        # every supercell vector must be an integer combination of the
        # unrotated lattice AND of the transformed lattice
        cell = _hexagonal_cell()
        basis = cell[:2, :2]
        for candidate in commensurate_twists(cell, max_index=3):
            super2d = candidate.supercell.astype(float) @ basis
            top_basis = basis @ candidate.transform
            in_top = super2d @ np.linalg.inv(top_basis)
            np.testing.assert_allclose(in_top, np.round(in_top), atol=1e-6)

    def test_oblique_needs_strain(self):
        oblique = np.array([[7.0, 0.0, 0.0], [2.3, 5.1, 0.0], [0.0, 0.0, 10.0]])
        exact = commensurate_twists(oblique, max_index=4, max_strain=0.0)
        # a generic oblique lattice admits no exact CSL rotation in a
        # small index range; a strain tolerance opens candidates up
        strained = commensurate_twists(oblique, max_index=4, max_strain=0.02)
        assert len(strained) > len(exact)
        assert all(c.strain <= 0.02 + 1e-12 for c in strained)


class TestTwistedBilayer:
    def test_square_snap_and_counts(self):
        layer = _square_layer()
        moire = layer.stack(mode="twisted", angle=37.0, interlayer=3.4)
        # snapped to the sigma-5 square CSL angle 36.87
        assert moire.graph.graph["twist_angle"] == pytest.approx(
            math.degrees(2 * math.atan2(1, 3)), abs=1e-6
        )
        assert moire.graph.graph["twist_strain"] == pytest.approx(0.0, abs=1e-12)
        # 5 cells per layer, 2 layers, 1 atom per cell
        assert len(moire) == 10
        assert float(moire.cell[2, 2]) == pytest.approx(2 * 3.4)
        # exactly half the atoms in each layer plane
        z = moire.cart_coords[:, 2]
        assert (np.abs(z - 0.0) < 1e-9).sum() == 5
        assert (np.abs(z - 3.4) < 1e-9).sum() == 5

    def test_bonds_duplicated_per_copy(self):
        graph = networkx.Graph(cell=_square_cell())
        graph.add_node(
            0, symbol="C", coord=np.array([0.0, 0.0, 0.0]), tag=1, ufftype="C_R"
        )
        graph.add_node(
            1, symbol="O", coord=np.array([1.4, 0.0, 0.0]), tag=2, ufftype="O_R"
        )
        graph.add_edge(0, 1, bond_order=1.0)
        layer = Framework(graph, name="pairlayer")
        moire = layer.stack(mode="twisted", angle=37.0)
        assert len(moire) == 20
        assert moire.graph.number_of_edges() == 10
        # tag uniqueness across all copies and both layers
        tags = [d["tag"] for _, d in moire.graph.nodes(data=True) if d["tag"] > 0]
        assert len(tags) == len(set(tags))
        # node ids contiguous (Framework invariant)
        assert sorted(moire.graph) == list(range(20))
        # bond lengths preserved by the pure rotation
        for i, j in moire.graph.edges():
            d = np.linalg.norm(
                moire.graph.nodes[i]["coord"] - moire.graph.nodes[j]["coord"]
            )
            assert d == pytest.approx(1.4, abs=1e-9)

    def test_atom_guardrail(self):
        layer = _square_layer()
        with pytest.raises(StackingError, match="max_atoms"):
            layer.stack(mode="twisted", angle=37.0, max_atoms=5)

    def test_no_angle_in_tolerance_lists_alternatives(self):
        layer = _square_layer()
        with pytest.raises(StackingError, match="nearest"):
            layer.stack(mode="twisted", angle=10.0, angle_tolerance=0.5, max_index=3)

    def test_requires_angle(self):
        with pytest.raises(ValueError, match="needs an angle"):
            _square_layer().stack(mode="twisted")

    def test_angle_only_for_twisted(self):
        with pytest.raises(ValueError, match="twisted"):
            _square_layer().stack(mode="AA", angle=30.0)

    def test_not_a_layer_rejected(self):
        graph = networkx.Graph(cell=np.diag([5.0, 5.0, 5.0]))
        for k in range(2):
            graph.add_node(
                k,
                symbol="C",
                coord=np.array([0.0, 0.0, 2.6 * k]),
                tag=0,
                ufftype="C_R",
            )
        tall = Framework(graph, name="notalayer")
        with pytest.raises(StackingError, match="not a 2D layer"):
            tall.stack(mode="twisted", angle=37.0)

    def test_twisted_wraps_and_exports(self, tmp_path):
        moire = _square_layer().stack(mode="twisted", angle=37.0)
        structure = moire.structure
        assert len(structure) == len(moire)
        path = moire.write_cif(tmp_path / "moire.cif")
        assert path.exists()

    def test_built_hcb_cof_layer(self):
        # the real use case: a boroxine-benzene COF on the honeycomb
        # net, twisted at the sigma-7 hexagonal CSL angle
        import os

        from autografs import Autografs

        fixture = os.path.join(
            os.path.dirname(__file__), "data", "topologies_fixture.json"
        )
        mofgen = Autografs(topofile=fixture)
        hcb = mofgen.topologies["hcb"]
        mappings = {}
        for key in hcb.mappings:
            conn = len(key.atoms.indices_from_symbol("X"))
            mappings[key] = {3: "Boroxine_triangle", 2: "Benzene_linear"}[conn]
        layer = mofgen.build(hcb, mappings=mappings, max_rmsd=0.5)

        moire = layer.stack(mode="twisted", angle=21.8, max_atoms=2000)
        assert moire.graph.graph["twist_angle"] == pytest.approx(
            _hex_family_angle(1, 2), abs=1e-6
        )
        assert len(moire) == 2 * 7 * len(layer)
        assert moire.graph.number_of_edges() == 2 * 7 * layer.graph.number_of_edges()
        # vdW-stacked layers must not clash
        assert moire.min_contact() > 1.5

    def test_direct_call_matches_stack(self):
        layer = _square_layer()
        via_stack = layer.stack(mode="twisted", angle=37.0)
        direct = twisted_bilayer(layer, 37.0, interlayer=3.35)
        assert len(via_stack) == len(direct)
        assert via_stack.graph.graph["twist_angle"] == pytest.approx(
            direct.graph.graph["twist_angle"]
        )
