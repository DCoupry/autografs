"""
Tests for the elastic-constants layer (autografs.elastic).

The tensor algebra (Voigt conversions, VRH averages, directional
Young's modulus, point-group projection) is pure numpy and always
runs, checked against independent analytic references (isotropic
closed forms, cubic copper single-crystal formulas). The end-to-end
finite-difference computation needs the optional LAMMPS backends and
is exercised in CI's relax job; it skips cleanly elsewhere.
"""

import os

import numpy as np
import pytest

from autografs.elastic import (
    ElasticProperties,
    full_to_voigt,
    symmetrized_stiffness,
    voigt_to_full,
)

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "topologies_fixture.json"
)


def _isotropic(bulk: float, shear: float) -> np.ndarray:
    c12 = bulk - 2.0 * shear / 3.0
    c11 = bulk + 4.0 * shear / 3.0
    voigt = np.diag([c11, c11, c11, shear, shear, shear])
    for i in range(3):
        for j in range(3):
            if i != j:
                voigt[i, j] = c12
    return voigt


def _cubic(c11: float, c12: float, c44: float) -> np.ndarray:
    voigt = np.diag([c11, c11, c11, c44, c44, c44])
    for i in range(3):
        for j in range(3):
            if i != j:
                voigt[i, j] = c12
    return voigt


# copper single-crystal constants (GPa), a standard anisotropic case
CU = {"c11": 168.4, "c12": 121.4, "c44": 75.4}


def _cubic_young(direction, c11, c12, c44):
    """Independent analytic oracle for cubic directional E.

    1/E(n) = S11 - 2 (S11 - S12 - S44/2)(n1²n2² + n2²n3² + n3²n1²)
    """
    s11 = (c11 + c12) / ((c11 - c12) * (c11 + 2.0 * c12))
    s12 = -c12 / ((c11 - c12) * (c11 + 2.0 * c12))
    s44 = 1.0 / c44
    n = np.asarray(direction, dtype=float)
    n = n / np.linalg.norm(n)
    cross = (n[0] * n[1]) ** 2 + (n[1] * n[2]) ** 2 + (n[2] * n[0]) ** 2
    return 1.0 / (s11 - 2.0 * (s11 - s12 - 0.5 * s44) * cross)


class TestVoigtConversion:
    def test_roundtrip(self):
        rng = np.random.default_rng(3)
        voigt = rng.normal(size=(6, 6))
        voigt = 0.5 * (voigt + voigt.T)
        np.testing.assert_allclose(full_to_voigt(voigt_to_full(voigt)), voigt)

    def test_full_tensor_symmetries(self):
        full = voigt_to_full(_cubic(**CU))
        # minor symmetries: ijkl == jikl == ijlk
        np.testing.assert_allclose(full, full.transpose(1, 0, 2, 3))
        np.testing.assert_allclose(full, full.transpose(0, 1, 3, 2))
        # major symmetry: ijkl == klij
        np.testing.assert_allclose(full, full.transpose(2, 3, 0, 1))


class TestIsotropicModuli:
    def test_all_schemes_agree(self):
        props = ElasticProperties(stiffness=_isotropic(bulk=100.0, shear=50.0))
        assert props.is_stable
        for value in (props.bulk_voigt, props.bulk_reuss, props.bulk_hill):
            assert value == pytest.approx(100.0)
        for value in (props.shear_voigt, props.shear_reuss, props.shear_hill):
            assert value == pytest.approx(50.0)
        # E = 9BG / (3B + G), nu = (3B - 2G) / (2(3B + G))
        assert props.young_hill == pytest.approx(9.0 * 100.0 * 50.0 / 350.0)
        assert props.poisson_hill == pytest.approx(200.0 / 700.0)

    def test_direction_independent(self):
        props = ElasticProperties(stiffness=_isotropic(bulk=100.0, shear=50.0))
        expected = props.young_hill
        assert props.young_modulus([1, 0, 0]) == pytest.approx(expected)
        assert props.young_modulus([1, 1, 1]) == pytest.approx(expected)
        assert props.young_min == pytest.approx(expected)
        assert props.young_max == pytest.approx(expected)


class TestCubicDirectional:
    def test_against_analytic_formula(self):
        props = ElasticProperties(stiffness=_cubic(**CU))
        for direction in ([1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 1, 0]):
            assert props.young_modulus(direction) == pytest.approx(
                _cubic_young(direction, **CU)
            )

    def test_extrema_at_high_symmetry(self):
        # copper: Zener ratio > 1, so E is softest along <100> and
        # stiffest along <111>
        props = ElasticProperties(stiffness=_cubic(**CU))
        assert props.young_min == pytest.approx(_cubic_young([1, 0, 0], **CU))
        assert props.young_max == pytest.approx(_cubic_young([1, 1, 1], **CU))
        assert props.young_min < props.young_hill < props.young_max

    def test_instability_detected(self):
        # pure shear instability: C44 < 0
        props = ElasticProperties(stiffness=_cubic(c11=100.0, c12=50.0, c44=-5.0))
        assert not props.is_stable


class TestSymmetrization:
    def test_projects_noise_onto_cubic_pattern(self):
        from pymatgen.core import Lattice, Structure

        rocksalt = Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0, 0]]
        )
        rng = np.random.default_rng(11)
        noisy = _cubic(**CU) + rng.normal(scale=2.0, size=(6, 6))
        projected = symmetrized_stiffness(noisy, rocksalt)
        # cubic pattern restored: three independent constants
        np.testing.assert_allclose(projected, projected.T, atol=1e-9)
        assert projected[0, 0] == pytest.approx(projected[1, 1])
        assert projected[0, 0] == pytest.approx(projected[2, 2])
        assert projected[0, 1] == pytest.approx(projected[0, 2])
        assert projected[3, 3] == pytest.approx(projected[5, 5])
        np.testing.assert_allclose(projected[:3, 3:], 0.0, atol=1e-9)
        # the projection is an average: the isotropic part is preserved
        noisy_sym = 0.5 * (noisy + noisy.T)
        assert ElasticProperties(projected).bulk_voigt == pytest.approx(
            ElasticProperties(noisy_sym).bulk_voigt
        )

    def test_identity_on_already_symmetric(self):
        from pymatgen.core import Lattice, Structure

        rocksalt = Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0, 0]]
        )
        cubic = _cubic(**CU)
        np.testing.assert_allclose(
            symmetrized_stiffness(cubic, rocksalt), cubic, atol=1e-9
        )


def _lammps_runtime_available() -> bool:
    """True when the optional backends import AND the runtime loads."""
    try:
        import lammps
        import lammps_interface  # noqa: F401

        lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none"])
        lmp.close()
        return True
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.skipif(
    not _lammps_runtime_available(),
    reason="LAMMPS backend not installed or runtime not loadable",
)
class TestElasticIntegration:
    @pytest.fixture(scope="class")
    def mof5(self):
        from autografs import Autografs

        mofgen = Autografs(topofile=FIXTURE_PATH)
        topology = mofgen.topologies["pcu"]
        mappings = {}
        for key in topology.mappings:
            conn = len(key.atoms.indices_from_symbol("X"))
            mappings[key] = "Zn_mof5_octahedral" if conn == 6 else "Benzene_linear"
        return mofgen.build(topology, mappings=mappings)

    def test_mof5_elastic(self, mof5):
        props = mof5.elastic_properties()
        stiffness = props.stiffness
        # mechanically stable, symmetric tensor
        assert props.is_stable
        np.testing.assert_allclose(stiffness, stiffness.T, atol=1e-9)
        # a single-cell build keeps each benzene ring in a fixed
        # plane, which breaks MOF-5's cubic symmetry down to
        # orthorhombic (P222) - the projection must zero the
        # normal-shear couplings, and the normal block should still be
        # near-cubic numerically
        np.testing.assert_allclose(stiffness[:3, 3:], 0.0, atol=1e-9)
        diagonal = np.diag(stiffness)[:3]
        assert diagonal.max() / diagonal.min() < 1.10
        # UFF-level MOF-5 moduli: order of magnitude, not truth level
        # (experiment/DFT put B near 16 GPa)
        assert 2.0 < props.bulk_hill < 60.0
        assert 0.0 < props.young_min <= props.young_max < 150.0
        # the input framework is untouched
        assert mof5.energy is None
