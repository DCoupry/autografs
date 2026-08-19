"""
Unit tests for the numpy alignment core (autografs.alignment).
"""

import numpy as np
import pytest

from autografs.alignment import kabsch, match_directions
from autografs.exceptions import AlignmentError


def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


TETRAHEDRON = np.array(
    [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float
) / np.sqrt(3)

SQUARE = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=float)

# the 12 vertices of a cuboctahedron: the arm shape of a 12-connected
# fcu node (UiO-66's Zr6), whose rotation group IS the octahedral group
CUBOCTAHEDRON = np.array(
    [
        [1, 1, 0],
        [1, -1, 0],
        [-1, 1, 0],
        [-1, -1, 0],
        [1, 0, 1],
        [1, 0, -1],
        [-1, 0, 1],
        [-1, 0, -1],
        [0, 1, 1],
        [0, 1, -1],
        [0, -1, 1],
        [0, -1, -1],
    ],
    dtype=float,
) / np.sqrt(2)


class TestKabsch:
    def test_recovers_known_rotation(self):
        rng = np.random.default_rng(7)
        rotation = rotation_z(0.83)
        sources = rng.normal(size=(8, 3))
        targets = sources @ rotation.T
        np.testing.assert_allclose(kabsch(sources, targets), rotation, atol=1e-12)

    def test_always_proper(self):
        """Even for mirrored data the result is a proper rotation."""
        rng = np.random.default_rng(11)
        sources = rng.normal(size=(5, 3))
        targets = sources * np.array([1.0, 1.0, -1.0])  # reflection
        rotation = kabsch(sources, targets)
        assert np.isclose(np.linalg.det(rotation), 1.0)


class TestMatchDirections:
    def test_recovers_rotation_and_permutation(self):
        rotation = rotation_z(0.7)
        shuffled = TETRAHEDRON[[2, 0, 3, 1]] @ rotation.T
        found_rotation, perm, rmsd = match_directions(shuffled, TETRAHEDRON)
        assert rmsd < 1e-9
        assert sorted(perm.tolist()) == [0, 1, 2, 3]
        np.testing.assert_allclose(
            TETRAHEDRON[perm] @ found_rotation.T, shuffled, atol=1e-9
        )

    def test_chirality_is_preserved(self):
        """A chiral star does not match its mirror image."""
        chiral = np.array(
            [
                [1.0, 0.1, 0.3],
                [-0.9, 1.1, -0.2],
                [0.2, -1.0, 1.4],
                [0.5, 0.7, -1.2],
            ]
        )
        chiral /= np.linalg.norm(chiral, axis=1, keepdims=True)
        mirror = chiral * np.array([1.0, 1.0, -1.0])

        _, _, rmsd_self = match_directions(chiral @ rotation_z(0.5).T, chiral)
        _, _, rmsd_mirror = match_directions(mirror, chiral)
        assert rmsd_self < 1e-9
        assert rmsd_mirror > 0.1

    def test_shape_mismatch_scores_high(self):
        """Square planar vs tetrahedral: the gate signal."""
        _, _, rmsd = match_directions(SQUARE, TETRAHEDRON)
        assert rmsd > 0.3

    def test_linear_pair(self):
        targets = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
        arms = np.array([[0, 0, 1.0], [0, 0, -1.0]])
        _, _, rmsd = match_directions(targets, arms)
        assert rmsd < 1e-9

    def test_count_mismatch_raises(self):
        with pytest.raises(AlignmentError, match="Cannot match"):
            match_directions(SQUARE, TETRAHEDRON[:3])

    def test_deterministic(self):
        results = [match_directions(SQUARE, TETRAHEDRON) for _ in range(3)]
        for rotation, perm, rmsd in results[1:]:
            np.testing.assert_array_equal(rotation, results[0][0])
            np.testing.assert_array_equal(perm, results[0][1])
            assert rmsd == results[0][2]

    def test_cuboctahedron_matches_itself_exactly(self):
        """#178: a shape whose rotation group IS the octahedral group
        collapses every cube-rotation start into one - the generic
        twist starts must still find the exact match. This is UiO-66's
        12-c node against fcu's slot, which used to score 0.363 and be
        rejected as incompatible."""
        rotation = rotation_z(0.9)
        rotated = CUBOCTAHEDRON[np.random.default_rng(5).permutation(12)]
        _, _, rmsd = match_directions(rotated @ rotation.T, CUBOCTAHEDRON)
        assert rmsd < 1e-6

    def test_octahedron_still_matches(self):
        """The easy high-symmetry case must not regress."""
        octahedron = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=float,
        )
        _, _, rmsd = match_directions(octahedron @ rotation_z(0.4).T, octahedron)
        assert rmsd < 1e-6

    def test_icosahedral_arms_match(self):
        """A 12-arm icosahedral star (rotation group disjoint from the
        cubic starts in a different way) also matches itself."""
        phi = (1 + np.sqrt(5)) / 2
        icosahedron = np.array(
            [
                [0, 1, phi],
                [0, -1, phi],
                [0, 1, -phi],
                [0, -1, -phi],
                [1, phi, 0],
                [-1, phi, 0],
                [1, -phi, 0],
                [-1, -phi, 0],
                [phi, 0, 1],
                [phi, 0, -1],
                [-phi, 0, 1],
                [-phi, 0, -1],
            ],
            dtype=float,
        )
        icosahedron /= np.linalg.norm(icosahedron, axis=1, keepdims=True)
        _, _, rmsd = match_directions(icosahedron @ rotation_z(0.6).T, icosahedron)
        assert rmsd < 1e-6


class TestCellParametrization:
    """Free cell parameters per crystal system."""

    @staticmethod
    def _param(sg, abc=(1.0, 1.0, 1.0), angles=(90.0, 90.0, 90.0)):
        from autografs.alignment import CellParametrization

        return CellParametrization(
            spacegroup_number=sg, blueprint_abc=abc, blueprint_angles=angles
        )

    def test_cubic_single_parameter(self):
        param = self._param(221)
        assert param.system == "cubic"
        assert param.n_free == 1
        assert param.expand([5.0]) == (5.0, 5.0, 5.0, 90.0, 90.0, 90.0)
        np.testing.assert_allclose(param.seed(np.array([4.0, 5.0, 6.0])), [5.0])

    def test_hexagonal(self):
        param = self._param(194, angles=(90.0, 90.0, 120.0))
        assert param.system == "hexagonal"
        assert param.expand([3.0, 4.0]) == (3.0, 3.0, 4.0, 90.0, 90.0, 120.0)

    def test_rhombohedral(self):
        param = self._param(148, angles=(80.0, 80.0, 80.0))
        assert param.system == "rhombohedral"
        assert param.expand([3.0, 70.0]) == (3.0, 3.0, 3.0, 70.0, 70.0, 70.0)

    def test_tetragonal(self):
        param = self._param(100)
        assert param.expand([3.0, 4.0]) == (3.0, 3.0, 4.0, 90.0, 90.0, 90.0)

    def test_orthorhombic(self):
        param = self._param(20)
        assert param.expand([2.0, 3.0, 4.0]) == (2.0, 3.0, 4.0, 90.0, 90.0, 90.0)

    def test_monoclinic_frees_unique_angle(self):
        param = self._param(14, angles=(90.0, 110.0, 90.0))
        assert param.system == "monoclinic"
        assert param.n_free == 4
        assert param.expand([2.0, 3.0, 4.0, 100.0]) == (
            2.0,
            3.0,
            4.0,
            90.0,
            100.0,
            90.0,
        )
        # the blueprint's unique angle seeds the free parameter
        np.testing.assert_allclose(
            param.seed(np.array([2.0, 3.0, 4.0])), [2.0, 3.0, 4.0, 110.0]
        )

    def test_triclinic_frees_everything(self):
        param = self._param(1, angles=(85.0, 95.0, 100.0))
        assert param.n_free == 6
        assert param.expand([2, 3, 4, 80, 90, 100]) == (
            2.0,
            3.0,
            4.0,
            80.0,
            90.0,
            100.0,
        )

    def test_unknown_keeps_blueprint_angles(self):
        param = self._param(None, angles=(90.0, 90.0, 120.0))
        assert param.system == "unknown"
        assert param.expand([2.0, 3.0, 4.0]) == (2.0, 3.0, 4.0, 90.0, 90.0, 120.0)

    def test_angles_clipped_to_sane_range(self):
        param = self._param(14, angles=(90.0, 110.0, 90.0))
        expanded = param.expand([2.0, 3.0, 4.0, 5.0])
        assert expanded[4] == 30.0  # clipped, not a degenerate 5-degree cell


class TestLayerCellParametrization:
    """Layer mode for 2D nets: c exactly frozen at the slab padding."""

    @staticmethod
    def _param(plane_group, abc=(1.0, 1.0, 10.0), angles=(90.0, 90.0, 90.0)):
        from autografs.alignment import CellParametrization

        return CellParametrization(
            spacegroup_number=plane_group,
            blueprint_abc=abc,
            blueprint_angles=angles,
            is_2d=True,
        )

    def test_hexagonal_layer_single_parameter(self):
        param = self._param(17, angles=(90.0, 90.0, 120.0))  # p6mm
        assert param.system == "layer_hexagonal"
        assert param.n_free == 1
        assert param.expand([25.0]) == (25.0, 25.0, 10.0, 90.0, 90.0, 120.0)
        np.testing.assert_allclose(param.seed(np.array([24.0, 26.0, 10.0])), [25.0])

    def test_square_layer_single_parameter(self):
        param = self._param(11)  # p4mm
        assert param.system == "layer_square"
        assert param.n_free == 1
        assert param.expand([7.0]) == (7.0, 7.0, 10.0, 90.0, 90.0, 90.0)

    def test_rectangular_layer_two_parameters(self):
        param = self._param(8)  # p2gg
        assert param.system == "layer_rectangular"
        assert param.n_free == 2
        assert param.expand([3.0, 4.0]) == (3.0, 4.0, 10.0, 90.0, 90.0, 90.0)
        np.testing.assert_allclose(param.seed(np.array([3.0, 4.0, 10.0])), [3.0, 4.0])

    def test_oblique_layer_three_parameters(self):
        param = self._param(2, angles=(90.0, 90.0, 105.0))  # p2
        assert param.system == "layer_oblique"
        assert param.n_free == 3
        assert param.expand([3.0, 4.0, 100.0]) == (
            3.0,
            4.0,
            10.0,
            90.0,
            90.0,
            100.0,
        )
        # blueprint gamma seeds the free angle
        np.testing.assert_allclose(
            param.seed(np.array([3.0, 4.0, 10.0])), [3.0, 4.0, 105.0]
        )

    def test_c_frozen_at_any_free_parameters(self):
        """The checklist item: c never moves, whatever the optimizer does."""
        param = self._param(17, abc=(1.7, 1.7, 10.0), angles=(90.0, 90.0, 120.0))
        for a in (0.1, 1.0, 42.0, 1234.5):
            assert param.expand([a])[2] == 10.0

    def test_missing_group_number_falls_back_to_oblique(self):
        param = self._param(None, angles=(90.0, 90.0, 100.0))
        assert param.system == "layer_oblique"
        assert param.expand([3.0, 4.0, 95.0])[2] == 10.0


class TestBondTargetScale:
    """An opt-in multiplier on the inter-unit covalent bond target.

    It exists for one measured purpose: a strained build can place
    atoms close enough that a force-field relaxation cannot descend
    from it, and an inflated start can. Building a *reported* structure
    with it would be wrong - the bonds are deliberately too long - so
    the default must stay exactly 1.0 and cost nothing.
    """

    FIXTURE = "tests/data/topologies_fixture.json"

    def _mof5(self):
        from autografs import Autografs

        mofgen = Autografs(topofile=self.FIXTURE)
        topology = mofgen.topologies["pcu"]
        mappings = {}
        for slot in topology.mappings:
            arms = len(slot.atoms.indices_from_symbol("X"))
            mappings[slot] = {6: "Zn_mof5_octahedral", 2: "Benzene_linear"}[arms]
        return mofgen, topology, mappings

    def test_the_default_is_exactly_the_unscaled_build(self):
        mofgen, topology, mappings = self._mof5()
        plain = mofgen.build(topology, mappings, max_rmsd=0.5)
        explicit = mofgen.build(topology, mappings, max_rmsd=0.5, bond_target_scale=1.0)
        np.testing.assert_allclose(
            np.asarray(plain.graph.graph["cell"], float),
            np.asarray(explicit.graph.graph["cell"], float),
            atol=1e-12,
        )

    def test_a_larger_target_opens_the_cell(self):
        mofgen, topology, mappings = self._mof5()
        built = {}
        for scale in (1.0, 1.5, 2.0):
            framework = mofgen.build(
                topology, mappings, max_rmsd=0.5, bond_target_scale=scale
            )
            built[scale] = float(
                np.linalg.norm(np.asarray(framework.graph.graph["cell"], float)[0])
            )
        assert built[1.0] < built[1.5] < built[2.0]
        # MOF-5 is cubic a = 12.9 A; the inflation is on the inter-unit
        # bond only, so the cell grows by well under the scale factor
        assert built[1.0] == pytest.approx(12.9, abs=0.2)

    def test_the_scale_reaches_the_objective_not_just_the_signature(self):
        """A build plan's bond targets must actually carry the factor."""
        from autografs.alignment import prepare_build

        mofgen, topology, mappings = self._mof5()
        validated, empty = mofgen._validate_mappings(topology, mappings)
        one = prepare_build(topology, validated, empty_slots=empty)
        two = prepare_build(
            topology, validated, empty_slots=empty, bond_target_scale=2.0
        )
        assert one.pairs, "fixture must produce paired anchors"
        index_a, target_a, index_b, target_b, _ = one.pairs[0]
        assert two._pair_bond_length(
            index_a, target_a, index_b, target_b
        ) == pytest.approx(
            2.0 * one._pair_bond_length(index_a, target_a, index_b, target_b)
        )


class TestReliefPass:
    """Spinning 2-connected units about their own anchor line.

    The finite counterpart of the rod pipeline's relief pass. Its whole
    claim is that it opens packing while leaving closure alone, so both
    halves are pinned here.
    """

    FIXTURE = "tests/data/topologies_fixture.json"

    def _worst_bond(self, framework):
        from autografs.validation import inter_unit_bond_deviations

        deviations = np.asarray(
            inter_unit_bond_deviations(framework), dtype=float
        ).ravel()
        return float(np.nanmax(deviations)) if deviations.size else float("nan")

    def _pair(self, net, **kwargs):
        from autografs import Autografs

        mofgen = Autografs(topofile=self.FIXTURE)
        topology = mofgen.topologies[net]
        mappings = mofgen.suggest_mappings(topology)[0]
        options = {"max_rmsd": 1.0, "min_distance": 0.0, **kwargs}
        return (
            mofgen.build(topology, mappings, **options),
            mofgen.build(topology, mappings, relieve=True, **options),
        )

    def test_the_axis_is_the_anchors_not_the_dummies(self):
        """Real linkers are bent, so the two lines are NOT the same.

        Measured on the shipped library: anchors sit 0.28-0.56 A off
        the dummy-to-dummy line. Spinning about that line would swing
        the anchors and change every bond they make.
        """
        from autografs.alignment import _relief_axis

        anchors = np.array([[-2.0, 0.4, 0.0], [2.0, 0.4, 0.0]])
        axis = _relief_axis(anchors)
        assert axis is not None
        np.testing.assert_allclose(axis, [1.0, 0.0, 0.0], atol=1e-12)
        # both anchors are equidistant from the axis line through them
        for anchor in anchors:
            along = (anchor - anchors[0]) @ axis
            offset = anchor - anchors[0] - along * axis
            assert np.linalg.norm(offset) < 1e-12

    def test_a_polytopic_unit_has_no_relief_axis(self):
        from autografs.alignment import _relief_axis

        assert _relief_axis(np.eye(3)) is None
        assert _relief_axis(np.zeros((1, 3))) is None

    def test_closure_is_untouched(self):
        """The guarantee: the anchors do not move, so no bond changes."""
        plain, relieved = self._pair("pcu")
        assert self._worst_bond(relieved) == pytest.approx(
            self._worst_bond(plain), abs=1e-9
        )

    def test_packing_improves_where_units_can_turn(self):
        """Improvement is a population claim (better on 71 of 102
        library nets, worse on 0): any SINGLE net's minimum contact
        can be pinned by an unturnable node-node pair, and which pair
        limits shifts with platform/BLAS rounding — one fixed net is
        not a stable oracle (pcu came back exactly unimproved on
        linux/py3.11 while improving on 3.12/3.13). Never-worse must
        hold on every net; a genuine improvement on at least one.
        """
        improvements = []
        for net in ("pcu", "sql", "srs", "hcb"):
            plain, relieved = self._pair(net)
            before, after = plain.min_contact(), relieved.min_contact()
            assert after >= before - 1e-9
            if after > before + 1e-6:
                marker = relieved.graph.graph["relief"]
                assert marker["contact_after"] > marker["contact_before"]
            improvements.append(after - before)
        assert max(improvements) > 1e-6

    def test_the_default_build_is_unchanged(self):
        from autografs import Autografs

        mofgen = Autografs(topofile=self.FIXTURE)
        topology = mofgen.topologies["pcu"]
        mappings = mofgen.suggest_mappings(topology)[0]
        plain = mofgen.build(topology, mappings, max_rmsd=1.0, min_distance=0.0)
        assert "relief" not in plain.graph.graph

    def test_the_net_is_preserved(self):
        """Bonds unchanged means the quotient graph is unchanged."""
        plain, relieved = self._pair("pcu")
        assert relieved.graph.number_of_edges() == plain.graph.number_of_edges()
        assert {frozenset(e) for e in relieved.graph.edges()} == {
            frozenset(e) for e in plain.graph.edges()
        }
