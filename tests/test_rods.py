"""
Tests for rod canonicalization and harvest dedup (autografs.rods).

Rod identity is exercised three ways: through full deconstruction of
the synthetic Zn-O pillar rod MOF (shared with test_deconstruct),
through hand-built helical rods where the screw operation is known
exactly (4-fold helix: chemical repeat = L/4, screw angle 90, and the
enantiomer must NOT match — proper flips preserve helicity), and
end-to-end through harvest (1x and 2x supercell deconstructions of
the same rod dedupe into one family; a metal swap does not).
"""

import math

import numpy as np
import pytest
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from autografs.deconstruct import RodUnit
from autografs.rods import RodRepeat, canonical_rod, merge_rod

from .test_deconstruct import _helical_rod_structure, _rod_pillar_structure


@pytest.fixture(scope="module")
def mofgen():
    import os

    from autografs import Autografs

    fixture = os.path.join(os.path.dirname(__file__), "data", "topologies_fixture.json")
    return Autografs(topofile=fixture)


def _pillar_repeat(mofgen, n_repeats=1, metal="Zn"):
    structure = _rod_pillar_structure(n_repeats)
    if metal != "Zn":
        structure = Structure(
            structure.lattice,
            [metal if s.specie.symbol == "Zn" else s.specie.symbol for s in structure],
            structure.frac_coords,
        )
    result = mofgen.deconstruct(structure)
    assert len(result.rod_units) == 1
    return canonical_rod(result.structure, result.rod_units[0])


def _helix(sense=1, length=8.0, order=4, rho=1.5, cell=25.0):
    """A hand-built helical rod: C atoms on a screw, N marking phase.

    Returns (structure, rod_unit). Each chemical repeat holds one CN
    pair; the crystallographic repeat is ``order`` chemical ones.
    """
    species = []
    coords = []
    step = length / order
    for j in range(order):
        theta = sense * math.radians(90.0) * j
        x = 10.0 + rho * math.cos(theta)
        y = 10.0 + rho * math.sin(theta)
        species += ["C", "N"]
        coords += [
            [x, y, j * step],
            [
                10.0 + (rho + 1.2) * math.cos(theta),
                10.0 + (rho + 1.2) * math.sin(theta),
                j * step,
            ],
        ]
    structure = Structure(
        Lattice.tetragonal(cell, length),
        species,
        coords,
        coords_are_cartesian=True,
    )
    rod = RodUnit(
        atom_indices=list(range(len(species))),
        axis=np.array([0.0, 0.0, 1.0]),
        repeat_length=length,
        generator=(0, 0, 1),
        poe_indices=[i for i in range(len(species)) if species[i] == "N"],
        n_connections=order,
    )
    return structure, rod


class TestPillarCanonicalization:
    def test_single_repeat(self, mofgen):
        repeat = _pillar_repeat(mofgen)
        assert repeat.formula == "OZn"
        assert repeat.repeat_length == pytest.approx(3.9)
        assert repeat.screw_order == 1
        assert repeat.screw_angle == pytest.approx(0.0)
        assert repeat.n_connections == 4

    def test_supercell_reduces_to_chemical_repeat(self, mofgen):
        # pitfall 3: a 2x cell has crystallographic repeat 7.8 but the
        # same chemical repeat; canonicalization must reduce it
        repeat = _pillar_repeat(mofgen, n_repeats=2)
        assert repeat.formula == "OZn"
        assert repeat.repeat_length == pytest.approx(3.9)
        assert repeat.n_connections == 4

    def test_supercell_matches_single(self, mofgen):
        assert _pillar_repeat(mofgen).matches(_pillar_repeat(mofgen, n_repeats=2))

    def test_metal_swap_does_not_match(self, mofgen):
        assert not _pillar_repeat(mofgen).matches(_pillar_repeat(mofgen, metal="Mg"))


class TestHelicalRods:
    def test_screw_detected(self):
        structure, rod = _helix()
        repeat = canonical_rod(structure, rod)
        assert repeat.screw_order == 4
        assert repeat.repeat_length == pytest.approx(2.0)
        assert abs(repeat.screw_angle) == pytest.approx(90.0)
        assert repeat.formula == "CN"
        assert repeat.n_connections == 1

    def test_rotated_translated_copy_matches(self):
        structure, rod = _helix()
        base = canonical_rod(structure, rod)
        # a rigid copy: rotate the whole helix about its axis and
        # slide it axially - the embedding freedoms identity ignores
        rotation = math.radians(35.0)
        cosr, sinr = math.cos(rotation), math.sin(rotation)
        shifted = []
        for site in structure:
            x, y, z = site.coords
            dx, dy = x - 10.0, y - 10.0
            shifted.append(
                [10.0 + cosr * dx - sinr * dy, 10.0 + sinr * dx + cosr * dy, z + 0.7]
            )
        moved = Structure(
            structure.lattice,
            [site.specie.symbol for site in structure],
            shifted,
            coords_are_cartesian=True,
        )
        assert base.matches(canonical_rod(moved, rod))

    def test_enantiomer_does_not_match(self):
        # proper flips preserve helicity: the mirror-image screw is a
        # different (chiral) building unit
        right = canonical_rod(*_helix(sense=1))
        left = canonical_rod(*_helix(sense=-1))
        assert not right.matches(left)
        # ...but each matches itself, including through the flip path
        assert right.matches(canonical_rod(*_helix(sense=1)))
        assert left.matches(canonical_rod(*_helix(sense=-1)))


class TestMergeRod:
    def test_same_rod_reuses_name(self, mofgen):
        library: dict[str, RodRepeat] = {}
        first = merge_rod(library, _pillar_repeat(mofgen), "rod_OZn")
        second = merge_rod(library, _pillar_repeat(mofgen, n_repeats=2), "rod_OZn")
        assert first == second == "rod_OZn"
        assert len(library) == 1

    def test_different_rod_gets_suffix(self):
        library: dict[str, RodRepeat] = {}
        merge_rod(library, canonical_rod(*_helix(sense=1)), "rod_CN")
        name = merge_rod(library, canonical_rod(*_helix(sense=-1)), "rod_CN")
        assert name == "rod_CN_2"
        assert len(library) == 2


class TestHarvestRods:
    def test_rod_families_deduped_with_provenance(self, mofgen):
        result = mofgen.harvest(
            [
                _rod_pillar_structure(1),
                _rod_pillar_structure(2),
            ]
        )
        assert result.n_processed == 2
        assert list(result.rods) == ["rod_OZn"]
        assert len(result.rod_provenance["rod_OZn"]) == 2
        # the finite linker (pyrazine) still harvests normally
        assert any(kind == "linker" for kind in result.kinds.values())
        assert "rod family" in result.report()

    def test_metal_variants_stay_distinct(self, mofgen):
        zinc = _rod_pillar_structure(1)
        magnesium = Structure(
            zinc.lattice,
            ["Mg" if s.specie.symbol == "Zn" else s.specie.symbol for s in zinc],
            zinc.frac_coords,
        )
        result = mofgen.harvest([zinc, magnesium])
        assert sorted(result.rods) == ["rod_MgO", "rod_OZn"]


class TestPoeNetsInLibrary:
    def test_rod_poe_nets_ship_in_the_bundled_library(self):
        from pathlib import Path

        import autografs
        from autografs.topology_io import load_topologies

        bundled = Path(autografs.__file__).parent / "data" / "topologies.json.gz"
        library = load_topologies(bundled)
        for net in ("etb", "sra", "bnn"):
            assert net in library


class TestRodFragment:
    def test_pillar_arms(self, mofgen):
        from autografs.rods import rod_fragment

        result = mofgen.deconstruct(_rod_pillar_structure())
        fragment = rod_fragment(result.structure, result.rod_units[0])
        assert len(fragment.arms) == 4
        assert fragment.positions.shape == (2, 3)
        # the pyrazine arms leave the Zn perpendicular to the rod axis
        # (local frame: axis = +z)
        for row, vector in fragment.arms:
            assert fragment.repeat.symbols[row] == "Zn"
            assert abs(vector[2]) < 1e-6
            assert np.linalg.norm(vector[:2]) > 0.5
        # template geometry: Zn and O separated by the chain spacing
        dz = abs(fragment.positions[0, 2] - fragment.positions[1, 2])
        assert dz == pytest.approx(1.95, abs=1e-6)

    def test_supercell_fragment_reduces_arms(self, mofgen):
        from autografs.rods import rod_fragment

        result = mofgen.deconstruct(_rod_pillar_structure(2))
        fragment = rod_fragment(result.structure, result.rod_units[0])
        # 8 cuts per crystallographic repeat reduce to 4 per chemical
        assert len(fragment.arms) == 4
        assert fragment.repeat.repeat_length == pytest.approx(3.9)

    def test_manual_rod_without_cut_vectors_has_no_arms(self):
        from autografs.rods import rod_fragment

        structure, rod = _helix()
        fragment = rod_fragment(structure, rod)
        assert fragment.arms == []
        assert fragment.repeat.screw_order == 4


class TestRodSerialization:
    def test_round_trip(self, mofgen, tmp_path):
        from autografs.rods import load_rods, rod_fragment, save_rods

        result = mofgen.deconstruct(_rod_pillar_structure())
        fragment = rod_fragment(result.structure, result.rod_units[0], name="rod_OZn")
        path = save_rods({"rod_OZn": fragment}, tmp_path / "rods.json.gz")
        loaded = load_rods(path)
        assert list(loaded) == ["rod_OZn"]
        restored = loaded["rod_OZn"]
        assert restored.matches(fragment)
        np.testing.assert_allclose(restored.positions, fragment.positions, atol=1e-7)
        assert len(restored.arms) == len(fragment.arms)
        for (row_a, vec_a), (row_b, vec_b) in zip(
            restored.arms, fragment.arms, strict=True
        ):
            assert row_a == row_b
            np.testing.assert_allclose(vec_a, vec_b, atol=1e-7)

    def test_version_gate(self, tmp_path):
        import json

        from autografs.rods import load_rods

        bad = tmp_path / "rods.json"
        bad.write_text(json.dumps({"format_version": 99, "rods": {}}))
        with pytest.raises(ValueError, match="format version"):
            load_rods(bad)

    def test_harvest_write_rods(self, mofgen, tmp_path):
        from autografs.rods import load_rods

        result = mofgen.harvest([_rod_pillar_structure(1)])
        path = result.write_rods(tmp_path / "rods.json")
        loaded = load_rods(path)
        assert list(loaded) == ["rod_OZn"]
        assert len(loaded["rod_OZn"].arms) == 4


class TestAxialRuns:
    def test_pcu_has_three_straight_runs(self, mofgen):
        from autografs.net import axial_runs

        runs = axial_runs(mofgen.topologies["pcu"])
        assert len(runs) == 3
        assert sorted(r.direction for r in runs) == [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
        ]
        for run in runs:
            # one node slot + one axial edge-center slot per period
            assert len(run.slots) == 2
            assert run.period == pytest.approx(1.0)

    def test_zigzag_nets_have_none(self, mofgen):
        from autografs.net import axial_runs

        for name in ("dia", "hcb", "srs"):
            assert axial_runs(mofgen.topologies[name]) == []

    def test_runs_are_deduplicated(self, mofgen):
        from autografs.net import axial_runs

        runs = axial_runs(mofgen.topologies["sql"])
        keys = {(r.direction, r.slots) for r in runs}
        assert len(keys) == len(runs) == 2


class TestRodForwardBuild:
    """Rod Stage C3: forward building of straight rod frameworks."""

    def _harvest(self, mofgen, n_repeats=1):
        result = mofgen.harvest([_rod_pillar_structure(n_repeats)])
        rod = result.rods["rod_OZn"]
        linker = result.fragments[
            next(n for n, k in result.kinds.items() if k == "linker")
        ]
        return rod, linker

    def test_pillar_round_trip(self, mofgen):
        # the acceptance criterion: harvest the pillar, rebuild it from
        # its RodFragment + pyrazine, deconstruct, recover the same net
        # and the same rod identity
        from autografs.rod_build import build_rod_framework

        rod, linker = self._harvest(mofgen)
        built = build_rod_framework(mofgen.topologies["pcu"], rod, linker)
        assert built.min_contact() > 1.0
        # two chemical repeats along c, one linker per lateral slot per
        # repeat: Zn2 O2 + 4 pyrazine (C4H4N2)
        assert built.symbols.count("Zn") == 2
        assert built.symbols.count("O") == 2
        assert built.symbols.count("N") == 8
        # Framework invariants
        assert sorted(built.graph) == list(range(len(built)))
        assert all(d["tag"] == 0 for _, d in built.graph.nodes(data=True))

        result = mofgen.deconstruct(built.structure)
        assert result.net_candidates == ["pcu"]
        assert len(result.rod_units) == 1
        rebuilt = canonical_rod(result.structure, result.rod_units[0])
        assert rod.repeat.matches(rebuilt)

    def test_supercell_harvest_builds_identically(self, mofgen):
        # a rod canonicalized from a 2-repeat cell (screw_order 2,
        # angle 0) still builds - its template tiles by translation
        from autografs.rod_build import build_rod_framework

        rod1, linker = self._harvest(mofgen, 1)
        rod2, _ = self._harvest(mofgen, 2)
        assert rod2.repeat.screw_order == 2
        assert abs(rod2.repeat.screw_angle) < 1.0
        one = build_rod_framework(mofgen.topologies["pcu"], rod1, linker)
        two = build_rod_framework(mofgen.topologies["pcu"], rod2, linker)
        assert len(one) == len(two)
        assert one.symbols == two.symbols

    def test_build_rod_entry_point(self, mofgen):
        rod, linker = self._harvest(mofgen)
        built = mofgen.build_rod(mofgen.topologies["pcu"], rod, linker)
        assert built.min_contact() > 1.0
        assert built.name == "pcu_rod_OZn"

    def test_bonds_survive_serialization_round_trip(self, mofgen, tmp_path):
        # a rod saved and reloaded still builds (bonds persist)
        from autografs.rod_build import build_rod_framework
        from autografs.rods import load_rods, save_rods

        rod, linker = self._harvest(mofgen)
        path = save_rods({"rod_OZn": rod}, tmp_path / "rods.json.gz")
        reloaded = load_rods(path)["rod_OZn"]
        assert reloaded.bonds == rod.bonds
        built = build_rod_framework(mofgen.topologies["pcu"], reloaded, linker)
        assert built.min_contact() > 1.0

    def test_helical_rod_rejected(self, mofgen):
        from autografs.rod_build import build_rod_framework
        from autografs.rods import rod_fragment

        structure, unit = _helix()
        helix = rod_fragment(structure, unit)
        assert abs(helix.repeat.screw_angle) > 1.0
        _, linker = self._harvest(mofgen)
        with pytest.raises(Exception, match="straight"):
            build_rod_framework(mofgen.topologies["pcu"], helix, linker)

    def test_no_axial_run_rejected(self, mofgen):
        from autografs.rod_build import build_rod_framework

        rod, linker = self._harvest(mofgen)
        # dia has no straight axial run
        with pytest.raises(Exception, match="axial slot run"):
            build_rod_framework(mofgen.topologies["dia"], rod, linker)

    def test_non_ditopic_linker_rejected(self, mofgen):
        from autografs.rod_build import build_rod_framework

        rod, _ = self._harvest(mofgen)
        node = mofgen.sbu["Zn_mof5_octahedral"]
        with pytest.raises(Exception, match="ditopic"):
            build_rod_framework(mofgen.topologies["pcu"], rod, node)


class TestRodEditingGuards:
    """Rod Stage C4: tag/anchor-based edits refuse rod frameworks."""

    @pytest.fixture(scope="class")
    def rod_framework(self, mofgen):
        from autografs.rod_build import build_rod_framework

        result = mofgen.harvest([_rod_pillar_structure(1)])
        rod = result.rods["rod_OZn"]
        linker = result.fragments[
            next(n for n, k in result.kinds.items() if k == "linker")
        ]
        return build_rod_framework(mofgen.topologies["pcu"], rod, linker)

    def test_is_rod_flag(self, rod_framework, mofgen):
        assert rod_framework.is_rod is True
        # a normal finite build is not a rod
        pcu = mofgen.topologies["pcu"]
        mappings = {}
        for key in pcu.mappings:
            conn = len(key.atoms.indices_from_symbol("X"))
            mappings[key] = {6: "Zn_mof5_octahedral", 2: "Benzene_linear"}[conn]
        assert mofgen.build(pcu, mappings=mappings).is_rod is False

    def test_defects_refused(self, rod_framework):
        with pytest.raises(ValueError, match="rod framework"):
            rod_framework.defects(fraction=0.1, seed=1)

    def test_rotate_refused(self, rod_framework):
        with pytest.raises(ValueError, match="rod framework"):
            rod_framework.rotate(0, 0.5)

    def test_flip_refused(self, rod_framework):
        with pytest.raises(ValueError, match="rod framework"):
            rod_framework.flip(0)

    def test_functionalize_refused(self, rod_framework):
        with pytest.raises(ValueError, match="rod framework"):
            rod_framework.functionalize(5, "CH3")

    def test_supercell_allowed_and_marker_preserved(self, rod_framework):
        # supercell is a valid graph op on a rod (it extends the rod);
        # the marker must follow so the guards still apply afterwards
        sc = rod_framework.supercell((1, 1, 2))
        assert len(sc) == 2 * len(rod_framework)
        assert sc.min_contact() > 1.0
        assert sc.is_rod is True
        with pytest.raises(ValueError, match="rod framework"):
            sc.defects(fraction=0.1, seed=1)

    def test_marker_survives_save_load(self, rod_framework, tmp_path):
        from autografs.framework import Framework

        path = rod_framework.save(tmp_path / "rod.json.gz")
        loaded = Framework.load(path)
        assert loaded.is_rod is True
        with pytest.raises(ValueError, match="rod framework"):
            loaded.flip(0)


class TestHelicalRodFixture:
    """The synthetic 2_1-screw rod (MOF-74 class): full deconstruct path."""

    def _canonical(self, mofgen, n_repeats=2):
        result = mofgen.deconstruct(_helical_rod_structure(n_repeats))
        assert len(result.rod_units) == 1
        return canonical_rod(result.structure, result.rod_units[0])

    def test_screw_recovered(self, mofgen):
        rep = self._canonical(mofgen)
        assert rep.screw_order == 2
        assert abs(rep.screw_angle) == pytest.approx(180.0, abs=1.0)
        assert rep.formula == "OZn"

    def test_supercell_dedupes_with_base(self, mofgen):
        # a 4-repeat cell is a supercell of the 2-repeat helical rod;
        # both canonicalize to the same chemical repeat and match
        base = self._canonical(mofgen, 2)
        big = self._canonical(mofgen, 4)
        assert big.screw_order == 4
        assert base.matches(big)

    def test_forward_build_refuses_helical(self, mofgen):
        # Stage C3 builds straight rods only; a real helical rod must
        # be rejected with a clear reason, not mis-built
        from autografs.rod_build import build_rod_framework

        result = mofgen.harvest([_helical_rod_structure()])
        rod = result.rods["rod_OZn"]
        linker = mofgen.sbu["Benzene_linear"]
        with pytest.raises(Exception, match="straight"):
            build_rod_framework(mofgen.topologies["pcu"], rod, linker)

    def test_harvest_keeps_helical_and_straight_apart(self, mofgen):
        # the straight pillar and the helical screw rod are different
        # building units even though both are -Zn-O- (OZn)
        result = mofgen.harvest([_rod_pillar_structure(1), _helical_rod_structure()])
        assert sorted(result.rods) == ["rod_OZn", "rod_OZn_2"]
        straight, helical = (
            result.rods["rod_OZn"],
            result.rods["rod_OZn_2"],
        )
        # whichever is which, one is straight and one is the 2_1 screw
        angles = sorted(abs(r.repeat.screw_angle) for r in (straight, helical))
        assert angles[0] == pytest.approx(0.0, abs=1.0)
        assert angles[1] == pytest.approx(180.0, abs=1.0)
        assert not straight.repeat.matches(helical.repeat)
