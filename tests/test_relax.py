"""
Tests for the LAMMPS/UFF4MOF relaxation layer (autografs.relax).

The mapping helpers are pure numpy and always run. The end-to-end
relaxation needs the optional backends (pip install autografs[relax])
plus a loadable LAMMPS runtime, and is exercised in CI's relax job;
it skips cleanly anywhere the runtime is unavailable (e.g. Windows
without the Microsoft MPI redistributable).
"""

import os

import numpy as np
import pytest

from autografs.exceptions import RelaxationError
from autografs.relax import _match_displacements, _parse_type_elements

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "topologies_fixture.json"
)


class TestMatchDisplacements:
    def test_recovers_small_displacements(self):
        rng = np.random.default_rng(7)
        cell = np.diag([10.0, 10.0, 10.0])
        orig = np.array([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0.9, 0.2, 0.7]])
        species = ["C", "C", "O"]
        shift = rng.uniform(-0.02, 0.02, orig.shape)
        relaxed = (orig + shift) % 1.0
        found = _match_displacements(orig, species, relaxed, species, cell)
        np.testing.assert_allclose(found, shift, atol=1e-12)

    def test_replicas_fold_onto_originals(self):
        # 8 supercell replicas of each atom fold to the same fractional
        # position; any replica is an acceptable match
        cell = np.diag([10.0, 10.0, 10.0])
        orig = np.array([[0.25, 0.25, 0.25]])
        relaxed = np.tile(orig + 0.01, (8, 1)) % 1.0
        found = _match_displacements(orig, ["C"], relaxed, ["C"] * 8, cell)
        np.testing.assert_allclose(found, [[0.01, 0.01, 0.01]], atol=1e-12)

    def test_minimum_image_across_boundary(self):
        cell = np.diag([10.0, 10.0, 10.0])
        orig = np.array([[0.99, 0.5, 0.5]])
        relaxed = np.array([[0.01, 0.5, 0.5]])
        found = _match_displacements(orig, ["C"], relaxed, ["C"], cell)
        np.testing.assert_allclose(found, [[0.02, 0.0, 0.0]], atol=1e-12)

    def test_species_constrained(self):
        # the nearest atom is the wrong element; matching must skip it
        cell = np.diag([10.0, 10.0, 10.0])
        orig = np.array([[0.5, 0.5, 0.5]])
        relaxed = np.array([[0.51, 0.5, 0.5], [0.6, 0.5, 0.5]])
        found = _match_displacements(orig, ["C"], relaxed, ["O", "C"], cell)
        np.testing.assert_allclose(found, [[0.1, 0.0, 0.0]], atol=1e-12)

    def test_missing_species_raises(self):
        cell = np.eye(3)
        with pytest.raises(RelaxationError, match="0 C atoms"):
            _match_displacements(
                np.array([[0.5, 0.5, 0.5]]),
                ["C"],
                np.array([[0.5, 0.5, 0.5]]),
                ["O"],
                cell,
            )


class TestParseTypeElements:
    def test_reads_masses_block(self, tmp_path):
        data = tmp_path / "data.test"
        data.write_text(
            "test data file\n\n"
            "2 atom types\n\n"
            "Masses\n\n"
            "1 12.0107 # C_R\n"
            "2 65.38 # Zn4+2\n\n"
            "Bond Coeffs\n\n"
            "1 100.0 1.4\n"
        )
        assert _parse_type_elements(data) == {1: "C", 2: "Zn"}

    def test_missing_block_raises(self, tmp_path):
        data = tmp_path / "data.test"
        data.write_text("no masses here\n")
        with pytest.raises(RelaxationError, match="Masses"):
            _parse_type_elements(data)


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
class TestRelaxIntegration:
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

    def test_relax_mof5(self, mof5):
        relaxed = mof5.relax()
        # same graph: atom count, species, bonds untouched (canonical
        # comparison - an undirected edge has no defined orientation)
        assert len(relaxed) == len(mof5)
        assert relaxed.symbols == mof5.symbols
        assert {frozenset(edge) for edge in relaxed.graph.edges()} == {
            frozenset(edge) for edge in mof5.graph.edges()
        }
        assert relaxed.graph.number_of_edges() == mof5.graph.number_of_edges()
        # the input framework is untouched
        assert mof5.energy is None
        # energy recorded per unit cell
        assert isinstance(relaxed.energy, float)
        # UFF4MOF keeps MOF-5 cubic and near the experimental cell
        abc = np.array(relaxed.lattice.abc)
        np.testing.assert_allclose(abc, abc[0], rtol=0.02)
        assert 12.0 < abc[0] < 14.0
        # relaxation is a perturbation, not a rebuild
        moved = np.linalg.norm(relaxed.cart_coords - mof5.cart_coords, axis=1)
        assert moved.max() < 1.5
        # still no overlapping atoms
        assert relaxed.min_contact() > 1.0

    def test_relaxed_framework_exports(self, mof5, tmp_path):
        relaxed = mof5.relax()
        path = relaxed.write_cif(tmp_path / "relaxed.cif")
        assert path.exists()
