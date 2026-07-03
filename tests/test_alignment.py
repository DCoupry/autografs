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
