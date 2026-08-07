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


def _lammps_interface_available() -> bool:
    try:
        import lammps_interface  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _lammps_interface_available(), reason="lammps-interface not installed"
)
class TestTypeHandOff:
    """Our UFF4MOF table and lammps-interface's are not the same set.

    Both are called "UFF4MOF" and they disagree: 15 of our 227 symbols
    are absent from its 221. Handing one over raises KeyError before a
    single step runs - which is what killed 5 of 25 relaxations in the
    generative sweep, on Co6+2 and N_3+4. The hand-off has to translate
    into the receiver's vocabulary, not just ours.
    """

    def test_every_type_we_assign_is_one_the_backend_knows(self):
        from lammps_interface.uff4mof import UFF4MOF_DATA

        from autografs.relax import _substitution_map

        mapping = _substitution_map("UFF4MOF")
        assert mapping, "the map must not be empty when the backend is present"
        for ours, target in mapping.items():
            if target is None:
                continue
            assert target in UFF4MOF_DATA, f"{ours} -> {target} is still unknown"

    def test_the_known_offenders_are_translated(self):
        from autografs.relax import _substitution_map

        mapping = _substitution_map("UFF4MOF")
        # absent from lammps-interface's table entirely
        assert mapping["Co6+2"] == "Co3+2"
        assert mapping["N_3+4"] == "N_3"
        # a padding-only difference is the SAME type, not a near miss:
        # our bare "K" is its "K_" (identical radius and angle)
        assert mapping["K"] == "K_"
        # a genuine renaming, handled by the explicit alias
        assert mapping["Lr6+3"] == "Lw6+3"

    def test_supported_types_are_left_alone(self):
        from autografs.relax import _substitution_map

        mapping = _substitution_map("UFF4MOF")
        for symbol in ("C_R", "H_", "O_2", "Zn3+2", "N_3"):
            assert mapping[symbol] == symbol

    def test_substitutes_stay_on_the_same_element(self):
        from autografs.relax import _element_of, _substitution_map

        for symbol, target in _substitution_map("UFF4MOF").items():
            if target is None or target == symbol:
                continue
            if symbol in {"Lr6+3"}:  # historical renaming, element differs
                continue
            assert _element_of(target) == _element_of(symbol), (symbol, target)

    def test_plain_uff_needs_far_more_translation(self):
        """UFF has no metal types, so most of ours must be substituted."""
        from autografs.relax import _substitution_map

        mapping = _substitution_map("UFF")
        translated = sum(1 for k, v in mapping.items() if v != k)
        assert translated > 50

    def test_an_unknown_force_field_disables_translation(self):
        """No opinion about Dreiding: leave lammps-interface to type."""
        from autografs.relax import _substitution_map

        assert _substitution_map("Dreiding") == {}


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


class TestCollapseGuard:
    """A minimiser converging on a degenerate cell is not a success.

    Measured need: relaxing a deliberately expanded start reported
    success while returning atoms 0.00 A apart, and over 210 real
    candidates four came back collapsed. Silently handing those back is
    worse than failing.
    """

    def _framework(self, spacing, cell=12.0):
        import networkx

        from autografs.framework import Framework

        graph = networkx.Graph(cell=np.diag([cell, cell, cell]))
        graph.add_node(0, symbol="C", coord=np.zeros(3), tag=0, ufftype="C_3")
        graph.add_node(
            1, symbol="C", coord=np.array([spacing, 0.0, 0.0]), tag=1, ufftype="C_3"
        )
        graph.add_edge(0, 1, bond_order=1.0)
        return Framework(graph, name="probe")

    def test_coincident_atoms_are_refused(self):
        from autografs.relax import _reject_collapse

        before = self._framework(1.5)
        after = self._framework(0.1)
        with pytest.raises(RelaxationError, match="collapsed"):
            _reject_collapse(before, after)

    def test_a_vanished_cell_is_refused(self):
        from autografs.relax import _reject_collapse

        before = self._framework(1.5, cell=12.0)
        # same spacing, a tenth of the volume
        after = self._framework(1.5, cell=5.0)
        with pytest.raises(RelaxationError, match="cell volume"):
            _reject_collapse(before, after)

    def test_an_ordinary_contraction_passes(self):
        """UFF contracts 2-15% by volume; that must not trip the guard."""
        from autografs.relax import _reject_collapse

        before = self._framework(1.5, cell=12.0)
        after = self._framework(1.4, cell=11.4)
        _reject_collapse(before, after)


class TestBuildRelaxedRetry:
    """The retry lives on Autografs, because Framework cannot rebuild.

    A built framework keeps no record of the blueprint and units it came
    from, so `relax` has nothing to re-derive an open build from.
    """

    def _mofgen(self):
        from autografs import Autografs

        mofgen = Autografs(topofile=FIXTURE_PATH)
        topology = mofgen.topologies["pcu"]
        mappings = {}
        for slot in topology.mappings:
            arms = len(slot.atoms.indices_from_symbol("X"))
            mappings[slot] = {6: "Zn_mof5_octahedral", 2: "Benzene_linear"}[arms]
        return mofgen, topology, mappings

    def test_a_first_time_success_is_returned_unmarked(self, monkeypatch):
        from autografs.framework import Framework

        mofgen, topology, mappings = self._mofgen()
        monkeypatch.setattr(Framework, "relax", lambda self, **kw: self)
        result = mofgen.build_relaxed(topology, mappings)
        assert "relax_retry_scale" not in result.graph.graph

    def test_a_failure_retries_from_an_open_build(self, monkeypatch):
        from autografs.framework import Framework

        mofgen, topology, mappings = self._mofgen()
        calls: list[float] = []

        def fake_relax(self, **kwargs):
            # the open build has a visibly larger cell than the faithful
            # one; fail the first (faithful) attempt, accept the retry
            a = float(np.linalg.norm(np.asarray(self.graph.graph["cell"], float)[0]))
            calls.append(a)
            if len(calls) == 1:
                raise RelaxationError("Bond atoms 1 2 missing on proc 0")
            return self

        monkeypatch.setattr(Framework, "relax", fake_relax)
        result = mofgen.build_relaxed(topology, mappings, retry_scales=(2.0,))
        assert result.graph.graph["relax_retry_scale"] == 2.0
        assert len(calls) == 2
        # the retry really was built open, not the same structure again
        assert calls[1] > calls[0]

    def test_every_scale_failing_re_raises(self, monkeypatch):
        from autografs.framework import Framework

        mofgen, topology, mappings = self._mofgen()

        def always_fail(self, **kwargs):
            raise RelaxationError("no minimum here")

        monkeypatch.setattr(Framework, "relax", always_fail)
        with pytest.raises(RelaxationError, match="no minimum here"):
            mofgen.build_relaxed(topology, mappings, retry_scales=(1.5, 2.0))
