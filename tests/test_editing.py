"""
Tests for post-build editing: rotation/flipping of placed SBUs,
supercells, statistical defects, and framework functionalization.

Built on a real MOF-5 (pcu) build from the committed topology fixture,
so every operation is exercised on a genuine periodic bond graph with
boundary-crossing bonds.
"""

import os

import numpy as np
import pytest

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "topologies_fixture.json"
)


@pytest.fixture(scope="module")
def mofgen():
    from autografs import Autografs

    return Autografs(topofile=FIXTURE_PATH)


@pytest.fixture(scope="module")
def mof5(mofgen):
    """One shared MOF-5 build; tests must not mutate it."""
    topology = mofgen.topologies["pcu"]
    mappings = {}
    for key in topology.mappings:
        conn = len(key.atoms.indices_from_symbol("X"))
        mappings[key] = {6: "Zn_mof5_octahedral", 2: "Benzene_linear"}[conn]
    return mofgen.build(topology, mappings=mappings, refine_cell=True, max_rmsd=0.5)


def linker_slot(framework):
    return next(s for s, name in framework.slots.items() if name == "Benzene_linear")


def node_slot(framework):
    return next(
        s for s, name in framework.slots.items() if name == "Zn_mof5_octahedral"
    )


def anchor_nodes(framework, slot):
    return [
        n
        for n, d in framework.graph.nodes(data=True)
        if d["slot"] == slot and d["tag"] > 0
    ]


class TestProvenance:
    """Built graphs record which placed SBU every atom belongs to."""

    def test_every_node_has_slot_and_sbu(self, mof5):
        for _, data in mof5.graph.nodes(data=True):
            assert isinstance(data["slot"], int)
            assert data["sbu"] in ("Zn_mof5_octahedral", "Benzene_linear")

    def test_slots_property(self, mof5):
        # pcu: one 6-c node + three 2-c edges per cell
        assert sorted(mof5.slots) == [0, 1, 2, 3]
        assert sorted(mof5.slots.values()) == [
            "Benzene_linear",
            "Benzene_linear",
            "Benzene_linear",
            "Zn_mof5_octahedral",
        ]

    def test_slot_atom_counts_match_sbus(self, mofgen, mof5):
        by_slot = {}
        for _, data in mof5.graph.nodes(data=True):
            by_slot[data["slot"]] = by_slot.get(data["slot"], 0) + 1
        for slot, name in mof5.slots.items():
            sbu = mofgen.sbu[name]
            expected = len(sbu.atoms) - len(sbu.atoms.indices_from_symbol("X"))
            assert by_slot[slot] == expected

    def test_stack_offsets_second_layer_slots(self, mofgen):
        topology = mofgen.topologies["hcb"]
        mappings = {}
        for key in topology.mappings:
            conn = len(key.atoms.indices_from_symbol("X"))
            mappings[key] = {3: "Boroxine_triangle", 2: "Benzene_linear"}[conn]
        layer = mofgen.build(topology, mappings=mappings, max_rmsd=0.5)
        stacked = layer.stack(mode="AB")
        assert len(stacked.slots) == 2 * len(layer.slots)
        assert len(set(stacked.slots)) == 2 * len(layer.slots)


class TestRotate:
    def test_preserves_bond_graph_and_anchors(self, mof5):
        slot = linker_slot(mof5)
        rotated = mof5.rotate(slot, np.pi / 6)
        assert set(rotated.graph.edges()) == set(mof5.graph.edges())
        for anchor in anchor_nodes(mof5, slot):
            assert np.allclose(
                rotated.graph.nodes[anchor]["coord"],
                mof5.graph.nodes[anchor]["coord"],
            )
        # the ring atoms actually moved
        assert np.abs(rotated.cart_coords - mof5.cart_coords).max() > 0.5

    def test_only_the_slot_moves(self, mof5):
        slot = linker_slot(mof5)
        rotated = mof5.rotate(slot, np.pi / 4)
        for node, data in mof5.graph.nodes(data=True):
            if data["slot"] != slot:
                assert np.allclose(rotated.graph.nodes[node]["coord"], data["coord"])

    def test_full_turn_is_identity(self, mof5):
        slot = linker_slot(mof5)
        rotated = mof5.rotate(slot, 2 * np.pi)
        assert np.allclose(rotated.cart_coords, mof5.cart_coords, atol=1e-9)

    def test_input_framework_unchanged(self, mof5):
        before = mof5.cart_coords.copy()
        mof5.rotate(linker_slot(mof5), 1.0)
        assert np.array_equal(mof5.cart_coords, before)

    def test_rejects_non_2c_sbu(self, mof5):
        with pytest.raises(ValueError, match="connection points"):
            mof5.rotate(node_slot(mof5), 1.0)

    def test_rejects_unknown_slot(self, mof5):
        with pytest.raises(ValueError, match="No placed SBU"):
            mof5.rotate(999, 1.0)


class TestFlip:
    def test_involution(self, mof5):
        slot = linker_slot(mof5)
        flipped = mof5.flip(slot)
        assert set(flipped.graph.edges()) == set(mof5.graph.edges())
        back = flipped.flip(slot)
        assert np.allclose(back.cart_coords, mof5.cart_coords, atol=1e-9)

    def test_anchors_fixed(self, mof5):
        slot = linker_slot(mof5)
        flipped = mof5.flip(slot)
        for anchor in anchor_nodes(mof5, slot):
            assert np.allclose(
                flipped.graph.nodes[anchor]["coord"],
                mof5.graph.nodes[anchor]["coord"],
            )

    def test_rejects_non_coplanar_anchors(self, mof5):
        # the octahedral node's six anchors span 3D
        with pytest.raises(ValueError, match="not coplanar"):
            mof5.flip(node_slot(mof5))


class TestSupercell:
    def test_counts_and_cell(self, mof5):
        supercell = mof5.supercell(2)
        assert len(supercell) == 8 * len(mof5)
        assert supercell.graph.number_of_edges() == 8 * mof5.graph.number_of_edges()
        assert len(supercell.slots) == 8 * len(mof5.slots)
        assert np.allclose(supercell.cell, 2.0 * mof5.cell)

    def test_matches_pymatgen_supercell(self, mof5):
        from pymatgen.analysis.structure_matcher import StructureMatcher

        supercell = mof5.supercell((2, 1, 1))
        reference = mof5.structure.copy()
        reference.make_supercell([2, 1, 1])
        assert StructureMatcher(primitive_cell=False).fit(
            reference, supercell.structure
        )

    def test_geometry_is_exact(self, mof5):
        # replication changes no local geometry, so the closest
        # non-bonded contact is identical - this fails if any
        # boundary-crossing bond were mapped to the wrong image
        supercell = mof5.supercell(2)
        assert supercell.min_contact() == pytest.approx(mof5.min_contact(), abs=1e-8)

    def test_anisotropic_multipliers(self, mof5):
        supercell = mof5.supercell((1, 1, 2))
        assert len(supercell) == 2 * len(mof5)
        assert supercell.graph.number_of_edges() == 2 * mof5.graph.number_of_edges()

    def test_tags_stay_pairwise(self, mof5):
        supercell = mof5.supercell(2)
        counts = {}
        for _, data in supercell.graph.nodes(data=True):
            if data["tag"] > 0:
                counts[data["tag"]] = counts.get(data["tag"], 0) + 1
        assert counts and set(counts.values()) == {2}

    def test_rejects_bad_multipliers(self, mof5):
        with pytest.raises(ValueError, match="positive ints"):
            mof5.supercell(0)
        with pytest.raises(ValueError, match="positive ints"):
            mof5.supercell((2, 2, -1))


@pytest.fixture(scope="module")
def supercell(mof5):
    return mof5.supercell(2)


class TestDefects:
    def test_fraction_removes_and_caps(self, mofgen, supercell):
        defective = supercell.defects(fraction=0.25, sbu="Benzene_linear", seed=42)
        n_linkers = sum(1 for v in supercell.slots.values() if v == "Benzene_linear")
        removed = round(0.25 * n_linkers)
        assert removed == 6
        linker = mofgen.sbu["Benzene_linear"]
        linker_atoms = len(linker.atoms) - len(linker.atoms.indices_from_symbol("X"))
        # each removed 2-c linker leaves two dangling anchors -> two H caps
        assert len(defective) == len(supercell) - removed * linker_atoms + 2 * removed
        assert len(defective.slots) == len(supercell.slots) - removed
        # caps sit one covalent bond from their anchor: still no overlap
        assert defective.min_contact() > 1.5

    def test_seed_determinism(self, supercell):
        one = supercell.defects(fraction=0.25, sbu="Benzene_linear", seed=7)
        two = supercell.defects(fraction=0.25, sbu="Benzene_linear", seed=7)
        assert np.allclose(one.cart_coords, two.cart_coords)
        other = supercell.defects(fraction=0.25, sbu="Benzene_linear", seed=8)
        assert len(other) == len(one)
        assert not np.allclose(other.cart_coords, one.cart_coords)

    def test_explicit_slots(self, mofgen, supercell):
        slot = linker_slot(supercell)
        defective = supercell.defects(slots=[slot])
        assert slot not in defective.slots
        linker = mofgen.sbu["Benzene_linear"]
        linker_atoms = len(linker.atoms) - len(linker.atoms.indices_from_symbol("X"))
        linker_h = sum(1 for s in linker.atoms if s.specie.symbol == "H")
        assert len(defective) == len(supercell) - linker_atoms + 2

        def n_hydrogens(framework):
            return sum(
                1 for _, d in framework.graph.nodes(data=True) if d["symbol"] == "H"
            )

        # the linker's own hydrogens leave with it; two caps come back
        assert n_hydrogens(defective) == n_hydrogens(supercell) - linker_h + 2

    def test_cap_none_leaves_open_sites(self, supercell):
        slot = linker_slot(supercell)
        capped = supercell.defects(slots=[slot], cap="H")
        open_site = supercell.defects(slots=[slot], cap=None)
        assert len(capped) == len(open_site) + 2
        # the surviving ex-partners lost a bond and their tag
        assert open_site.graph.number_of_edges() < supercell.graph.number_of_edges()

    def test_unpaired_anchor_tags_cleared(self, supercell):
        defective = supercell.defects(slots=[linker_slot(supercell)], cap="H")
        counts = {}
        for _, data in defective.graph.nodes(data=True):
            if data["tag"] > 0:
                counts[data["tag"]] = counts.get(data["tag"], 0) + 1
        assert set(counts.values()) == {2}

    def test_node_ids_contiguous(self, supercell):
        defective = supercell.defects(fraction=0.25, sbu="Benzene_linear", seed=1)
        assert sorted(defective.graph) == list(range(len(defective)))

    def test_input_validation(self, supercell):
        with pytest.raises(ValueError, match="exactly one"):
            supercell.defects()
        with pytest.raises(ValueError, match="exactly one"):
            supercell.defects(fraction=0.1, slots=[0])
        with pytest.raises(ValueError, match="within"):
            supercell.defects(fraction=1.5)
        with pytest.raises(ValueError, match="No placed SBU named"):
            supercell.defects(fraction=0.5, sbu="Unobtainium")
        with pytest.raises(ValueError, match="Unknown slot"):
            supercell.defects(slots=[99999])
        with pytest.raises(ValueError, match="every placed SBU"):
            supercell.defects(fraction=1.0)


class TestFunctionalize:
    def test_sites_are_terminal_hydrogens(self, mof5):
        sites = mof5.functionalizable_sites()
        # 3 linkers x 4 aromatic H
        assert len(sites) == 12
        for site in sites:
            data = mof5.graph.nodes[site]
            assert data["symbol"] == "H"
            assert data["tag"] == 0
            assert mof5.graph.degree(site) == 1

    def test_sbu_filter(self, mof5):
        assert mof5.functionalizable_sites(sbu="Benzene_linear") == (
            mof5.functionalizable_sites()
        )
        assert mof5.functionalizable_sites(sbu="Zn_mof5_octahedral") == []

    def test_single_site_amine(self, mof5):
        site = mof5.functionalizable_sites()[0]
        parent = next(iter(mof5.graph.neighbors(site)))
        func = mof5.functionalize(site, "amine")
        assert func.structure.composition["N"] == 1
        # H count: -1 replaced +2 from NH2
        assert func.structure.composition["H"] == (mof5.structure.composition["H"] + 1)
        assert len(func) == len(mof5) + 2
        assert sorted(func.graph.edges()) and sorted(func.graph) == list(
            range(len(func))
        )
        # the nitrogen bonds to the old parent carbon at a sane length
        nitrogen = next(n for n, d in func.graph.nodes(data=True) if d["symbol"] == "N")
        assert func.graph.degree(nitrogen) == 3
        parent_coord = mof5.graph.nodes[parent]["coord"]
        distance = np.linalg.norm(func.graph.nodes[nitrogen]["coord"] - parent_coord)
        assert 1.2 < distance < 1.6

    def test_multi_site(self, mof5):
        sites = mof5.functionalizable_sites()[:4]
        func = mof5.functionalize(sites, "methyl")
        assert func.structure.composition["C"] == (mof5.structure.composition["C"] + 4)
        assert func.structure.composition["H"] == (
            mof5.structure.composition["H"] + 2 * 4
        )

    def test_custom_group_molecule(self, mof5):
        from pymatgen.core.structure import Molecule

        # a hydroxyl written by hand: X marks the attachment point
        group = Molecule(
            ["X", "O", "H"],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.3, 0.9, 0.0]],
        )
        site = mof5.functionalizable_sites()[0]
        func = mof5.functionalize(site, group)
        assert func.structure.composition["O"] == (mof5.structure.composition["O"] + 1)

    def test_provenance_inherited(self, mof5):
        site = mof5.functionalizable_sites()[0]
        parent_slot = mof5.graph.nodes[next(iter(mof5.graph.neighbors(site)))]["slot"]
        func = mof5.functionalize(site, "amine")
        nitrogen = next(n for n, d in func.graph.nodes(data=True) if d["symbol"] == "N")
        assert func.graph.nodes[nitrogen]["slot"] == parent_slot
        assert func.slots == mof5.slots

    def test_rejects_bad_sites(self, mof5):
        # a non-terminal atom: a carbon that is not itself an anchor
        carbon = next(
            n
            for n, d in mof5.graph.nodes(data=True)
            if d["symbol"] == "C" and d["tag"] == 0 and mof5.graph.degree(n) > 1
        )
        with pytest.raises(ValueError, match="terminal"):
            mof5.functionalize(carbon, "amine")
        # an anchor atom
        anchor = anchor_nodes(mof5, linker_slot(mof5))[0]
        with pytest.raises(ValueError, match="connection point"):
            mof5.functionalize(anchor, "amine")
        with pytest.raises(ValueError, match="No atom"):
            mof5.functionalize(10**6, "amine")

    def test_rejects_bad_groups(self, mof5):
        from pymatgen.core.structure import Molecule

        site = mof5.functionalizable_sites()[0]
        with pytest.raises(ValueError, match="Unknown functional group"):
            mof5.functionalize(site, "unobtainide")
        no_dummy = Molecule(["O", "H"], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="exactly one dummy"):
            mof5.functionalize(site, no_dummy)


class TestCombinedWorkflow:
    def test_supercell_defect_functionalize_chain(self, mof5):
        """The v2 workflow: supercell -> defects -> decorate -> export."""
        edited = mof5.supercell((2, 1, 1)).defects(
            fraction=0.2, sbu="Benzene_linear", seed=3
        )
        edited = edited.rotate(linker_slot(edited), np.pi / 3)
        site = edited.functionalizable_sites()[0]
        final = edited.functionalize(site, "nitro")
        # the result is still a valid, exportable framework
        assert final.structure.composition["N"] == 1
        assert sorted(final.graph) == list(range(len(final)))
        assert final.min_contact() > 1.0
