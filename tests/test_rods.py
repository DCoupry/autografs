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


def _unc_helical_rod(unc, scale=4.0):
    """A RodFragment hand-matched to unc's 4_1 run, plus a ditopic linker.

    A -Zn-O- helix (screw order 4, +90 deg): one Zn per chemical repeat
    on a cylinder about the run axis, an inner bridging O to the next
    repeat (kept clear of the outward linkers so it clusters with the
    metal, not the organic), and two lateral arms per Zn pointing at
    unc's non-run slots. Deriving the geometry from unc's own run makes
    the build close exactly. Returns (rod, linker).
    """
    from pymatgen.core.structure import Molecule

    from autografs.fragment import Fragment
    from autografs.net import (
        _topology_slot_images,
        helical_runs,
        topology_quotient_edges,
    )
    from autografs.rods import RodFragment

    cell = unc.cell.matrix
    run = next(r for r in helical_runs(unc) if r.screw_order == 4)
    axis_index = int(np.argmax(np.abs(run.direction)))
    axis_hat = cell[axis_index] / np.linalg.norm(cell[axis_index])
    counts = [len(s.atoms.indices_from_symbol("X")) for s in unc.slots]
    nodes = [s for s in run.slots if counts[s] > 2]

    images = _topology_slot_images(unc)
    centers = np.array(
        [np.asarray(s.atoms.cart_coords).mean(axis=0) for s in unc.slots]
    )
    wrapped = centers - np.array([images[i] for i in range(len(centers))]) @ cell
    steps: dict[int, list] = {i: [] for i in range(len(centers))}
    for (a, b, v), _ in topology_quotient_edges(unc).items():
        for u, w, sg in ((a, b, 1), (b, a, -1)):
            off = sg * np.asarray(v, float)
            disp = wrapped[w] + off @ cell - wrapped[u]
            if np.linalg.norm(disp) > 1e-9:
                steps[u].append((w, disp))
    pos = {run.slots[0]: wrapped[run.slots[0]]}
    cur = run.slots[0]
    for nxt in run.slots[1:]:
        pos[nxt] = pos[cur] + next(d for w, d in steps[cur] if w == nxt)
        cur = nxt

    n0 = nodes[0]
    axis_pt = np.array(run.axis_point)
    rel0 = (pos[n0] - axis_pt) - ((pos[n0] - axis_pt) @ axis_hat) * axis_hat
    radius = np.linalg.norm(rel0)
    e1 = rel0 / radius
    e2 = np.cross(axis_hat, e1)
    basis = np.array([e1, e2, axis_hat])
    run_set = set(run.slots)
    lateral = np.array([d for w, d in steps[n0] if w not in run_set])
    arm_local = lateral @ basis.T  # (2, 3): radial, tangential, axial

    chemical = float(run.period / 4) * scale
    r_zn = radius * scale
    r_o = r_zn * 0.45  # inner bridge, clear of the outward linkers
    arms = arm_local * scale
    half = np.radians(run.screw_angle) / 2.0
    positions = np.array(
        [
            [r_zn, 0.0, 0.0],
            [r_o * np.cos(half), r_o * np.sin(half), chemical / 2],
        ]
    )
    repeat = RodRepeat(
        symbols=["Zn", "O"],
        axial=np.array([0.0, chemical / 2]),
        radial=np.array([r_zn, r_o]),
        angular=np.array([0.0, half]),
        repeat_length=chemical,
        screw_order=4,
        screw_angle=float(run.screw_angle),
        n_connections=2,
    )
    rod = RodFragment(
        repeat=repeat,
        positions=positions,
        arms=[(0, arms[0]), (0, arms[1])],
        bonds=[(0, 1, 0), (1, 0, 1)],
        name="rod_unc4",
    )
    linker = Fragment(
        atoms=Molecule(
            ["X", "C", "C", "X"],
            [[-1.9, 0, 0], [-0.7, 0, 0], [0.7, 0, 0], [1.9, 0, 0]],
            site_properties={"tags": [1, 0, 0, 2]},
        ),
        name="cc",
    )
    return rod, linker


def _etb_helical_rod(etb, scale=6.0):
    """A RodFragment matched to etb's 3_1 metal-oxo helix, plus a linker.

    etb (MOF-74's net) is a cross-linked multi-rod net: six interleaved
    3_1 helices per cell, three of each hand (etb is centrosymmetric),
    joined by ditopic linkers that bridge helices of *opposite* hand.
    One rod species fills all six - the enantiomer is placed on the
    opposite-hand runs. The rod is derived from a +120 deg helix; each
    3-connected node carries a single lateral arm. Returns (rod, linker).
    """
    from pymatgen.core.structure import Molecule

    from autografs.fragment import Fragment
    from autografs.net import (
        _topology_slot_images,
        helical_runs,
        topology_quotient_edges,
    )
    from autografs.rods import RodFragment

    cell = etb.cell.matrix
    axis_hat = cell[2] / np.linalg.norm(cell[2])
    counts = [len(s.atoms.indices_from_symbol("X")) for s in etb.slots]
    plus = next(
        r for r in helical_runs(etb) if r.screw_order == 3 and r.screw_angle > 0
    )
    images = _topology_slot_images(etb)
    centers = np.array(
        [np.asarray(s.atoms.cart_coords).mean(axis=0) for s in etb.slots]
    )
    wrapped = centers - np.array([images[i] for i in range(len(centers))]) @ cell
    steps: dict[int, list] = {i: [] for i in range(len(centers))}
    for (a, b, v), _ in topology_quotient_edges(etb).items():
        for u, w, sg in ((a, b, 1), (b, a, -1)):
            off = sg * np.asarray(v, float)
            disp = wrapped[w] + off @ cell - wrapped[u]
            if np.linalg.norm(disp) > 1e-9:
                steps[u].append((w, disp))
    pos = {plus.slots[0]: wrapped[plus.slots[0]]}
    cur = plus.slots[0]
    for nxt in plus.slots[1:]:
        pos[nxt] = pos[cur] + next(d for w, d in steps[cur] if w == nxt)
        cur = nxt

    n0 = next(s for s in plus.slots if counts[s] > 2)
    axis_pt = np.array(plus.axis_point)
    rel0 = (pos[n0] - axis_pt) - ((pos[n0] - axis_pt) @ axis_hat) * axis_hat
    radius = np.linalg.norm(rel0)
    e1 = rel0 / radius
    e2 = np.cross(axis_hat, e1)
    basis = np.array([e1, e2, axis_hat])
    lateral = np.array([d for w, d in steps[n0] if w not in set(plus.slots)])
    arm_dir = (lateral[0] @ basis.T) / np.linalg.norm(lateral[0])

    chemical = float(plus.period / 3) * scale
    r_zn = radius * scale
    r_o = r_zn * 0.45  # inner bridge, clear of the outward linkers
    half = np.radians(plus.screw_angle) / 2.0
    positions = np.array(
        [
            [r_zn, 0.0, 0.0],
            [r_o * np.cos(half), r_o * np.sin(half), chemical / 2],
        ]
    )
    repeat = RodRepeat(
        symbols=["Zn", "O"],
        axial=np.array([0.0, chemical / 2]),
        radial=np.array([r_zn, r_o]),
        angular=np.array([0.0, half]),
        repeat_length=chemical,
        screw_order=3,
        screw_angle=float(plus.screw_angle),
        n_connections=1,
    )
    rod = RodFragment(
        repeat=repeat,
        positions=positions,
        arms=[(0, arm_dir * 1.2)],
        bonds=[(0, 1, 0), (1, 0, 1)],
        name="rod_etb",
    )
    linker = Fragment(
        atoms=Molecule(
            ["X", "C", "C", "X"],
            [[-1.8, 0, 0], [-0.7, 0, 0], [0.7, 0, 0], [1.8, 0, 0]],
            site_properties={"tags": [1, 0, 0, 2]},
        ),
        name="cc",
    )
    return rod, linker


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


@pytest.fixture(scope="module")
def bundled_topologies():
    """The full shipped topology library (etb lives here, not in the
    reduced test fixture)."""
    from pathlib import Path

    import autografs
    from autografs.topology_io import load_topologies

    bundled = Path(autografs.__file__).parent / "data" / "topologies.json.gz"
    return load_topologies(bundled)


class TestHelicalRuns:
    """Rod Stage C: detecting the spiralling slot runs a helical rod
    (MOF-74 / etb class) occupies, where straight axial_runs finds none.
    The library nets are the oracles: etb carries a 3_1 metal-oxo screw
    and srs is the textbook 4_1 chiral net."""

    def test_etb_has_the_mof74_screw(self, bundled_topologies):
        from autografs.net import axial_runs, helical_runs

        etb = bundled_topologies["etb"]
        # etb's channels are helical: nothing straight, everything screw
        assert axial_runs(etb) == []
        runs = helical_runs(etb)
        assert runs
        # a rod axis is a single cell direction (etb's short c)
        assert all(sorted(map(abs, r.direction)) == [0, 0, 1] for r in runs)
        # the distinct screw is 3_1: three node slots per period, +-120
        distinct = {(r.screw_order, round(abs(r.screw_angle))) for r in runs}
        assert distinct == {(3, 120)}
        for run in runs:
            assert -180.0 < run.screw_angle <= 180.0
            # screw_order node slots (>2 X) sit on the run
            nodes = [
                s
                for s in run.slots
                if len(etb.slots[s].atoms.indices_from_symbol("X")) > 2
            ]
            assert len(nodes) == run.screw_order

    def test_srs_is_the_four_one_screw(self, mofgen):
        from autografs.net import helical_runs

        runs = helical_runs(mofgen.topologies["srs"])
        assert runs
        # srs (the (10,3)-a net) is chiral: a single 4_1 handedness
        distinct = {(r.screw_order, round(r.screw_angle)) for r in runs}
        assert distinct == {(4, 90)}

    def test_straight_nets_have_no_helical_runs(self, mofgen):
        from autografs.net import helical_runs

        # pcu/dia/sql are straight or zig-zag, never screw
        for name in ("pcu", "dia", "sql"):
            assert helical_runs(mofgen.topologies[name]) == []

    def test_layer_nets_are_skipped(self, mofgen):
        from autografs.net import helical_runs

        # a 2D net's c is frozen slab padding; an in-plane 2_1 zig-zag
        # is not a 3D channel a rod could occupy
        hcb = mofgen.topologies["hcb"]
        assert hcb.is_2d
        assert helical_runs(hcb) == []

    def test_screw_order_matches_a_harvested_helical_rod(
        self, mofgen, bundled_topologies
    ):
        # the blueprint-side screw is the mirror of a rod's own: the
        # synthetic 2_1 pillar is order 2 / 180, and no straight run
        # carries it - only a helical one would
        from autografs.net import helical_runs

        result = mofgen.harvest([_helical_rod_structure()])
        rod = next(iter(result.rods.values()))
        assert rod.repeat.screw_order == 2
        assert abs(abs(rod.repeat.screw_angle) - 180.0) < 5.0
        # etb's order-3 screw would not host an order-2 rod; a matching
        # host needs (screw_order, |screw_angle|) agreement
        etb_screws = {
            (r.screw_order, round(abs(r.screw_angle)))
            for r in helical_runs(bundled_topologies["etb"])
        }
        assert (rod.repeat.screw_order, 180) not in etb_screws


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

    def test_rod_without_bonds_rejected(self, mofgen):
        # a hand-built rod carrying no internal bond graph cannot be
        # built (there is nothing to wire the continuation from)
        from autografs.rod_build import build_rod_framework
        from autografs.rods import rod_fragment

        structure, unit = _helix()  # RodUnit with no internal_bonds
        helix = rod_fragment(structure, unit)
        assert helix.bonds == []
        _, linker = self._harvest(mofgen)
        with pytest.raises(Exception, match="internal bond graph"):
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


class TestRodVerifyNet:
    """Rod net verification against the blueprint's PoE form (#158).

    A rod build records its continuation as direct point-of-extension
    edges and is an n_repeats supercell, so it is verified by a
    cell-choice-invariant signature comparison rather than the exact
    multiset the finite pipeline uses."""

    def _harvest(self, mofgen, n=1):
        result = mofgen.harvest([_rod_pillar_structure(n)])
        rod = result.rods["rod_OZn"]
        linker = result.fragments[
            next(name for name, k in result.kinds.items() if k == "linker")
        ]
        return rod, linker

    def test_correct_build_verifies(self, mofgen):
        from autografs.rod_build import build_rod_framework

        rod, linker = self._harvest(mofgen)
        built = build_rod_framework(mofgen.topologies["pcu"], rod, linker)
        # explicit and as a build-time gate: neither raises
        built.verify_net(mofgen.topologies["pcu"])
        build_rod_framework(mofgen.topologies["pcu"], rod, linker, verify_net=True)

    def test_supercell_rod_still_verifies(self, mofgen):
        # a rod canonicalized from a 2x cell (screw_order 2) builds an
        # n_repeats supercell; the gcd fold still matches the blueprint
        from autografs.rod_build import build_rod_framework

        rod2, linker = self._harvest(mofgen, 2)
        built = build_rod_framework(mofgen.topologies["pcu"], rod2, linker)
        built.verify_net(mofgen.topologies["pcu"])

    def test_dropped_inter_unit_bond_raises(self, mofgen):
        from autografs.exceptions import NetMismatchError
        from autografs.framework import Framework
        from autografs.rod_build import build_rod_framework

        rod, linker = self._harvest(mofgen)
        built = build_rod_framework(mofgen.topologies["pcu"], rod, linker)
        # mis-wire: drop one rod-linker bond; the coordination
        # sequences shift and no run's PoE form matches
        graph = built.graph.copy()
        u, v = next(
            (a, b)
            for a, b in graph.edges()
            if graph.nodes[a]["sbu"] != graph.nodes[b]["sbu"]
        )
        graph.remove_edge(u, v)
        with pytest.raises(NetMismatchError):
            Framework(graph, name="miswired").verify_net(mofgen.topologies["pcu"])

    def test_verify_against_runless_topology_raises(self, mofgen):
        from autografs.exceptions import NetMismatchError
        from autografs.rod_build import build_rod_framework

        rod, linker = self._harvest(mofgen)
        built = build_rod_framework(mofgen.topologies["pcu"], rod, linker)
        # dia has neither a straight nor a helical run to verify against
        with pytest.raises(NetMismatchError):
            built.verify_net(mofgen.topologies["dia"])

    def test_blueprint_poe_form_contracts_edge_centers(self, mofgen):
        # the PoE expansion turns pcu's node+axial-edge-center run into a
        # direct node self-loop continuation (one vertex, not two)
        from autografs.net import (
            axial_runs,
            topology_quotient_edges,
            topology_rod_quotient_edges,
        )

        pcu = mofgen.topologies["pcu"]
        run = axial_runs(pcu)[0]
        poe = topology_rod_quotient_edges(pcu, run)
        # a self-loop on the node slot carrying the run's generator
        node = next(
            s for s in run.slots if len(pcu.slots[s].atoms.indices_from_symbol("X")) > 2
        )
        assert any(a == b == node for a, b, _ in poe)
        # the axial edge-center slot is gone (contracted away)
        ec = next(s for s in run.slots if s != node)
        assert not any(ec in (a, b) for a, b, _ in poe)
        # and it is a strict reduction of the full quotient
        assert len(poe) < len(topology_quotient_edges(pcu))


class TestGeneralHelicalBuild:
    """Rod Stage C: forward building of a general (non-180 deg) helical
    rod on a spiralling blueprint. unc is a 4_1 single-helix net; the
    rod's four chemical repeats fill its four node slots by the screw."""

    def test_four_one_rod_builds_and_round_trips(self, bundled_topologies):
        from autografs.deconstruct import deconstruct
        from autografs.rod_build import build_rod_framework

        unc = bundled_topologies["unc"]
        rod, linker = _unc_helical_rod(unc)
        built = build_rod_framework(unc, rod, linker)
        # -Zn-O- helix (4 repeats) + one X-C-C-X linker per lateral slot
        assert built.symbols.count("Zn") == 4
        assert built.symbols.count("O") == 4
        assert built.symbols.count("C") == 8
        # a genuine helix: the default contact gate passes unrelaxed
        assert built.min_contact() > 1.0
        assert sorted(built.graph) == list(range(len(built)))
        # it realizes unc's net (PoE-form quotient), and round-trips to
        # the same 4_1 rod identity
        built.verify_net(unc)
        result = deconstruct(built.structure, topologies=bundled_topologies)
        assert result.net_candidates == ["unc"]
        assert len(result.rod_units) == 1
        rebuilt = canonical_rod(result.structure, result.rod_units[0])
        assert rebuilt.screw_order == 4
        assert abs(rebuilt.screw_angle - 90.0) < 5.0
        assert rod.repeat.matches(rebuilt)

    def test_the_build_is_actually_helical(self, bundled_topologies):
        # the four Zn sit at four distinct azimuths about the axis (a
        # 4_1 screw), not stacked straight above one another
        from autografs.rod_build import build_rod_framework

        unc = bundled_topologies["unc"]
        rod, linker = _unc_helical_rod(unc)
        built = build_rod_framework(unc, rod, linker)
        cell = built.cell
        axis = cell[2] / np.linalg.norm(cell[2])
        zn = np.array(
            [
                built.graph.nodes[n]["coord"]
                for n, d in built.graph.nodes(data=True)
                if d["symbol"] == "Zn"
            ]
        )
        perp = zn - np.outer(zn @ axis, axis)
        center = perp.mean(axis=0)
        radial = perp - center
        e1 = radial[0] / np.linalg.norm(radial[0])
        e2 = np.cross(axis, e1)
        azimuths = np.sort(np.degrees(np.arctan2(radial @ e2, radial @ e1)) % 360)
        gaps = np.diff(np.concatenate([azimuths, [azimuths[0] + 360]]))
        # four Zn evenly spread ~90 deg apart round the axis
        assert np.allclose(gaps, 90.0, atol=15.0)

    def test_rod_screw_must_match_a_run(self, bundled_topologies):
        # a 3_1 rod has no matching run on unc (only 4_1 / 2_1) -> rejected
        from autografs.rod_build import build_rod_framework

        unc = bundled_topologies["unc"]
        rod, linker = _unc_helical_rod(unc)
        rod.repeat.screw_order = 3
        rod.repeat.screw_angle = 120.0
        with pytest.raises(Exception, match="helical|run"):
            build_rod_framework(unc, rod, linker)


class TestCrossLinkedMultiRodBuild:
    """Rod Stage C: cross-linked multi-rod nets. etb (MOF-74's net) is
    six interleaved 3_1 helices per cell, three of each hand, joined by
    ditopic linkers that bridge helices of opposite hand - one rod per
    helix, with the enantiomer placed on the opposite-hand runs."""

    def test_etb_builds_six_cross_linked_rods(self, bundled_topologies):
        from autografs.rod_build import build_rod_framework

        etb = bundled_topologies["etb"]
        rod, linker = _etb_helical_rod(etb)
        # the synthetic fixture crowds etb's small cell (as the other
        # helical fixtures do); relax the contact gate - relax() cleans
        # it - and check the topology, which is what matters
        built = build_rod_framework(etb, rod, linker, min_distance=None)
        # six -Zn-O- rods (3 repeats each) + nine X-C-C-X cross-linkers
        assert built.symbols.count("Zn") == 18
        assert built.symbols.count("O") == 18
        assert built.symbols.count("C") == 18
        assert built.is_rod
        assert sorted(built.graph) == list(range(len(built)))
        # 18 rod-repeat slots (6 rods x 3) + 9 linker slots
        assert len(set(built.slots)) == 27
        # it realizes etb's cross-linked net (all six runs' PoE form)
        built.verify_net(etb)

    def test_etb_build_identifies_as_etb(self, bundled_topologies):
        from autografs.net import framework_quotient_edges, identify_net
        from autografs.rod_build import build_rod_framework

        etb = bundled_topologies["etb"]
        rod, linker = _etb_helical_rod(etb)
        built = build_rod_framework(etb, rod, linker, min_distance=None)
        # 18 three-connected nodes + 9 two-connected linkers: etb
        matches = identify_net(framework_quotient_edges(built), bundled_topologies)
        assert matches == ["etb"]

    def test_etb_places_both_handednesses(self, bundled_topologies):
        # etb is centrosymmetric: the six helices come in both hands. The
        # placement reflects the rod onto the opposite-hand runs, so the
        # built Zn helices split into +120 and -120 screws.
        from autografs.rod_build import _RodBuild, _select_runs

        etb = bundled_topologies["etb"]
        rod, linker = _etb_helical_rod(etb)
        build = _RodBuild(etb, rod, linker, _select_runs(etb, rod, True))
        assert build.n_rods == 6
        reflected = [spec["reflect"] for spec in build.rod_specs]
        # three helices of each hand -> three rods reflected, three not
        assert reflected.count(True) == 3
        assert reflected.count(False) == 3


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

    def test_forward_build_applies_the_screw(self, mofgen):
        # forward building places n_repeats = screw_order copies, each
        # rotated by n*screw about the axis, so the built framework is a
        # genuine 2_1 helix that round-trips to the same rod identity.
        # min_distance is relaxed: the synthetic fixture's two tilted
        # pyrazines crowd on the small pcu cell (a real helical MOF is
        # roomier; relax() cleans the geometry) - the topology is right.
        from autografs.rod_build import build_rod_framework

        result = mofgen.harvest([_helical_rod_structure()])
        rod = result.rods["rod_OZn"]
        linker = result.fragments[
            next(n for n, k in result.kinds.items() if k == "linker")
        ]
        built = build_rod_framework(
            mofgen.topologies["pcu"], rod, linker, min_distance=None
        )
        # two chemical repeats along c (screw order 2), Zn2 O2
        assert built.symbols.count("Zn") == 2
        assert built.symbols.count("O") == 2
        # the O atoms spiral: their transverse positions are on opposite
        # sides of the axis (a 180 degree screw), not stacked
        o_coords = built.cart_coords[
            [i for i, s in enumerate(built.symbols) if s == "O"]
        ]
        axis = built.cell[2] / np.linalg.norm(built.cell[2])
        perp = o_coords - np.outer(o_coords @ axis, axis)
        perp -= perp.mean(axis=0)
        assert float(perp[0] @ perp[1]) < 0  # opposite transverse sides

        deconstructed = mofgen.deconstruct(built.structure)
        assert deconstructed.net_candidates == ["pcu"]
        rebuilt = canonical_rod(deconstructed.structure, deconstructed.rod_units[0])
        assert rebuilt.screw_order == 2
        assert abs(rebuilt.screw_angle) == pytest.approx(180.0, abs=1.0)
        assert rod.repeat.matches(rebuilt)

    def test_straight_still_builds_cleanly(self, mofgen):
        # regression: the screw machinery leaves straight rods (screw 0
        # -> pure translation) untouched - they pass the default gate
        from autografs.rod_build import build_rod_framework

        result = mofgen.harvest([_rod_pillar_structure(1)])
        rod = result.rods["rod_OZn"]
        linker = result.fragments[
            next(n for n, k in result.kinds.items() if k == "linker")
        ]
        built = build_rod_framework(mofgen.topologies["pcu"], rod, linker)
        assert built.min_contact() > 1.0

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


class TestForwardUnwrap:
    """The monotonic-forward unwrap in _local_positions underpins
    helical bond recording (rod Stage C / #158)."""

    def test_helix_lays_out_monotonically(self, mofgen):
        # a 2_1 screw rod's atoms unwrap in increasing axial order,
        # not folded back next to the anchor by min-image
        import numpy as np

        from autografs.rods import _local_positions

        result = mofgen.deconstruct(_helical_rod_structure())
        rod = result.rod_units[0]
        pos = _local_positions(
            result.structure, rod.atom_indices, rod.atom_indices[0], rod.internal_bonds
        )
        z = np.sort(pos[:, 2])
        assert np.allclose(z, [0.0, 1.95, 3.9, 5.85], atol=0.05)

    def test_helical_fragment_has_clean_bonds(self, mofgen):
        # the payoff: the helical rod's template bond graph is the
        # clean -Zn-O- chain, so the fragment is build-ready
        frag = self._fragment(mofgen)
        assert len(frag.bonds) == 2
        assert sorted(m for _, _, m in frag.bonds) == [-1, 0]

    def test_no_bond_graph_falls_back(self, mofgen):
        # without internal_bonds the unwrap reverts to min-image
        # (thin-rod behaviour) and still produces a usable frame
        import numpy as np

        from autografs.rods import _local_positions

        result = mofgen.deconstruct(_rod_pillar_structure())
        rod = result.rod_units[0]
        with_bonds = _local_positions(
            result.structure, rod.atom_indices, rod.atom_indices[0], rod.internal_bonds
        )
        without = _local_positions(
            result.structure, rod.atom_indices, rod.atom_indices[0], None
        )
        # the straight pillar is thin: both routes agree
        assert np.allclose(with_bonds, without, atol=1e-6)

    def _fragment(self, mofgen):
        from autografs.rods import rod_fragment

        result = mofgen.deconstruct(_helical_rod_structure())
        return rod_fragment(result.structure, result.rod_units[0])
