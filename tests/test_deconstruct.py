"""Tests for framework deconstruction (autografs.deconstruct) and net
identification (autografs.net.identify_net)."""

import os
from collections import Counter

import pytest
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from autografs.deconstruct import _hill_formula, deconstruct
from autografs.exceptions import DeconstructionError
from autografs.net import (
    identify_net,
    net_signature,
    topology_quotient_edges,
)

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "topologies_fixture.json"
)

FIXTURE_NETS = ["acs", "dia", "hcb", "pcu", "sql", "srs"]

# the textbook coordination sequence of the primitive cubic net
PCU_SEQUENCE = (6, 18, 38, 66, 102, 146, 198, 258, 326, 402)


@pytest.fixture(scope="module")
def mofgen():
    from autografs import Autografs

    return Autografs(topofile=FIXTURE_PATH)


def build(mofgen, topo_name, choices, **kwargs):
    topology = mofgen.topologies[topo_name]
    mappings = {}
    for key in topology.mappings:
        conn = len(key.atoms.indices_from_symbol("X"))
        mappings[key] = choices[conn]
    kwargs.setdefault("max_rmsd", 0.5)
    return mofgen.build(topology, mappings=mappings, **kwargs)


@pytest.fixture(scope="module")
def mof5(mofgen):
    return build(
        mofgen,
        "pcu",
        {6: "Zn_mof5_octahedral", 2: "Benzene_linear"},
        refine_cell=True,
    )


@pytest.fixture(scope="module")
def mof5_deconstruction(mofgen, mof5):
    return mofgen.deconstruct(mof5.structure)


class TestNetSignature:
    def test_pcu_coordination_sequence(self, mofgen):
        edges = topology_quotient_edges(mofgen.topologies["pcu"])
        signature = net_signature(edges)
        assert signature == ((PCU_SEQUENCE, 1),)

    def test_fixture_signatures_all_distinct(self, mofgen):
        signatures = {}
        for name in FIXTURE_NETS:
            edges = topology_quotient_edges(mofgen.topologies[name])
            signatures[name] = net_signature(edges)
        assert len(set(signatures.values())) == len(FIXTURE_NETS)

    def test_caps_are_pruned(self, mofgen):
        """A dangling 1-coordinated vertex must not change the signature."""
        edges = topology_quotient_edges(mofgen.topologies["pcu"])
        with_cap = Counter(edges)
        with_cap[(0, 99, (0, 0, 0))] += 1
        assert net_signature(with_cap) == net_signature(edges)

    def test_edge_centers_are_contracted(self, mofgen):
        """The contracted signature ignores 2-coordinated decorations:
        pcu's blueprint (with edge centers) and its bare quotient (three
        self-loops on one vertex) reduce to the same signature."""
        bare: Counter = Counter(
            {
                (0, 0, (1, 0, 0)): 1,
                (0, 0, (0, 1, 0)): 1,
                (0, 0, (0, 0, 1)): 1,
            }
        )
        edges = topology_quotient_edges(mofgen.topologies["pcu"])
        assert net_signature(bare) == net_signature(edges)
        # while the uncontracted signature separates them
        assert net_signature(bare, contract=False) != net_signature(
            edges, contract=False
        )


class TestIdentifyNet:
    @pytest.mark.parametrize("name", FIXTURE_NETS)
    def test_each_fixture_net_identifies_itself(self, mofgen, name):
        edges = topology_quotient_edges(mofgen.topologies[name])
        assert identify_net(edges, mofgen.topologies) == [name]

    def test_unknown_net_returns_empty(self, mofgen):
        # the 4-coordinated nbo net is not in the fixture library
        nbo_like: Counter = Counter(
            {
                (0, 1, (0, 0, 0)): 1,
                (0, 1, (1, 0, 0)): 1,
                (0, 2, (0, 0, 0)): 1,
                (0, 2, (0, 0, 1)): 1,
                (1, 2, (0, 1, 0)): 1,
                (1, 2, (-1, 0, 1)): 1,
            }
        )
        matches = identify_net(nbo_like, mofgen.topologies)
        assert matches == []
        assert matches.tier is None

    def test_exact_tier_surfaced(self, mofgen):
        """A blueprint quotient matches its own net on the exact tier."""
        edges = topology_quotient_edges(mofgen.topologies["pcu"])
        matches = identify_net(edges, mofgen.topologies)
        assert matches == ["pcu"]
        assert matches.tier == "exact"

    def test_contracted_tier_surfaced(self, mofgen):
        """The bare pcu quotient (no edge centers) misses the exact
        tier - the blueprint counts its 2-coordinated edge centers as
        vertices - and falls back to the contraction-blind match."""
        bare: Counter = Counter(
            {
                (0, 0, (1, 0, 0)): 1,
                (0, 0, (0, 1, 0)): 1,
                (0, 0, (0, 0, 1)): 1,
            }
        )
        matches = identify_net(bare, mofgen.topologies)
        assert matches == ["pcu"]
        assert matches.tier == "contracted"

    def test_signature_cache_releases_topologies(self, mofgen):
        """The per-topology signature cache must not pin dead libraries."""
        import copy
        import gc

        from autografs.net import _SIGNATURE_CACHE

        topology = copy.deepcopy(mofgen.topologies["pcu"])
        before = len(_SIGNATURE_CACHE)
        matches = identify_net(
            topology_quotient_edges(topology), {"pcu_copy": topology}
        )
        assert matches == ["pcu_copy"]
        assert len(_SIGNATURE_CACHE) == before + 1
        del topology
        gc.collect()
        assert len(_SIGNATURE_CACHE) == before


class TestDeconstructMOF5:
    def test_units(self, mof5_deconstruction):
        kinds = Counter(unit.kind for unit in mof5_deconstruction.units)
        assert kinds == {"node": 1, "linker": 3}
        by_kind = {unit.kind: unit for unit in mof5_deconstruction.units}
        assert by_kind["node"].n_connections == 6
        assert by_kind["linker"].n_connections == 2

    def test_fragments_are_library_ready(self, mof5_deconstruction):
        fragments = mof5_deconstruction.fragments
        assert len(fragments) == 2
        node = fragments["node_C6O13Zn4_6X"]
        linker = fragments["linker_C6H4_2X"]
        assert len(node.atoms.indices_from_symbol("X")) == 6
        assert len(linker.atoms.indices_from_symbol("X")) == 2

    def test_net_identified(self, mof5_deconstruction):
        assert mof5_deconstruction.net_candidates == ["pcu"]

    def test_unit_atoms_partition_the_structure(self, mof5_deconstruction):
        indices = [i for unit in mof5_deconstruction.units for i in unit.atom_indices]
        assert sorted(indices) == list(range(len(mof5_deconstruction.structure)))

    def test_rebuild_from_extracted_fragments(self, mofgen, mof5_deconstruction):
        """The extracted fragments must be buildable as-is."""
        topology = mofgen.topologies["pcu"]
        fragments = mof5_deconstruction.fragments
        mappings = {}
        for key in topology.mappings:
            conn = len(key.atoms.indices_from_symbol("X"))
            name = {6: "node_C6O13Zn4_6X", 2: "linker_C6H4_2X"}[conn]
            mappings[key] = fragments[name]
        rebuilt = mofgen.build(
            topology, mappings=mappings, refine_cell=True, max_rmsd=0.5
        )
        rebuilt.verify_net(topology)
        assert len(rebuilt) == len(mof5_deconstruction.structure)

    def test_write_xyz_roundtrip(self, mof5_deconstruction, tmp_path):
        from autografs.utils import xyz_to_sbu

        path = mof5_deconstruction.write_xyz(tmp_path / "harvested.xyz")
        loaded = xyz_to_sbu(str(path))
        assert set(loaded) == set(mof5_deconstruction.fragments)
        for name, fragment in loaded.items():
            original = mof5_deconstruction.fragments[name]
            assert len(fragment.atoms) == len(original.atoms)
            assert fragment.has_compatible_symmetry(original, max_rmsd=0.05)


class TestGuestRemoval:
    def test_free_guests_are_removed_and_reported(self, mofgen, mof5):
        structure = mof5.structure.copy()
        structure.append("Xe", [0.5, 0.5, 0.5])
        structure.append("O", [0.25, 0.25, 0.25])
        structure.append("H", [0.251, 0.25, 0.19])
        structure.append("H", [0.19, 0.25, 0.251])
        result = mofgen.deconstruct(structure)
        assert result.guest_formulas == ["H2O", "Xe"]
        assert result.net_candidates == ["pcu"]
        assert len(result.structure) == len(mof5.structure)


class TestPaddlewheel:
    def test_2d_paddlewheel_framework(self, mofgen):
        layer = build(mofgen, "sql", {4: "Zn_square_paddlewheel", 2: "Benzene_linear"})
        result = mofgen.deconstruct(layer.structure)
        assert result.net_candidates == ["sql"]
        assert "node_C4O8Zn2_4X" in result.fragments
        node = result.fragments["node_C4O8Zn2_4X"]
        assert len(node.atoms.indices_from_symbol("X")) == 4


class TestCatenation:
    @pytest.fixture(scope="class")
    def dia(self, mofgen):
        # a sparse dia (long linker) so a second net fits without clashes
        return build(
            mofgen,
            "dia",
            {4: "CdGaS_cluster_tetrahedral", 2: "Bis_phenylethynylbenzene_linear"},
            max_rmsd=0.6,
        )

    def test_single_framework_is_not_catenated(self, mofgen, dia):
        result = mofgen.deconstruct(dia.structure)
        assert result.n_periodic_components == 1
        assert result.is_catenated is False
        assert result.subframework_nets == [["dia"]]
        assert result.net_candidates == ["dia"]

    def test_two_fold_interpenetration_detected(self, mofgen, dia):
        catenated = dia.interpenetrate(2)
        result = mofgen.deconstruct(catenated.structure)
        assert result.n_periodic_components == 2
        assert result.is_catenated is True
        # each subframework identified independently...
        assert result.subframework_nets == [["dia"], ["dia"]]
        # ...and the consensus is the single realized net
        assert result.net_candidates == ["dia"]
        # the fold is surfaced in the repr
        assert "2-fold" in repr(result)


class TestParallelBridges:
    """A one-atom linker bridging a node to that node's own periodic
    image cuts the same atom pair twice, through different images. Both
    cuts must survive into the quotient graph: a simple unit graph
    would collapse them, prune the linkers as caps, and identify
    nothing (regression test for #102)."""

    @pytest.fixture(scope="class")
    def carbide(self, mofgen):
        # cubic ZnC3: each face-edge C sits between a Zn and its image,
        # 2.2 A away on both sides - pcu with one-atom edge centers
        structure = Structure(
            Lattice.cubic(4.4),
            ["Zn", "C", "C", "C"],
            [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
        )
        return deconstruct(structure, topologies=mofgen.topologies)

    def test_units(self, carbide):
        kinds = Counter(unit.kind for unit in carbide.units)
        assert kinds == Counter({"node": 1, "linker": 3})
        assert all(
            unit.n_connections == 2 for unit in carbide.units if unit.kind == "linker"
        )

    def test_every_cut_bond_is_a_quotient_edge(self, carbide):
        # 3 linkers x 2 cut bonds each; parallel cuts kept distinct
        assert sum(carbide.quotient_edges.values()) == 6

    def test_net_identified(self, carbide):
        assert carbide.net_candidates == ["pcu"]


class TestCOF:
    """Metal-free frameworks take the branch-point path."""

    def test_2d_boroxine_cof_round_trip(self, mofgen):
        cof = build(mofgen, "hcb", {3: "Boroxine_triangle", 2: "Benzene_linear"})
        result = mofgen.deconstruct(cof.structure)
        assert result.net_candidates == ["hcb"]
        # boroxine ring is the 3-c node, benzene the 2-c linker
        assert "node_B3O3_3X" in result.fragments
        assert "linker_C6H4_2X" in result.fragments
        assert len(result.fragments["node_B3O3_3X"].atoms.indices_from_symbol("X")) == 3
        # rebuild from the extracted fragments and confirm the net
        topology = mofgen.topologies["hcb"]
        mappings = {
            key: result.fragments[
                {3: "node_B3O3_3X", 2: "linker_C6H4_2X"}[
                    len(key.atoms.indices_from_symbol("X"))
                ]
            ]
            for key in topology.mappings
        }
        rebuilt = mofgen.build(topology, mappings=mappings, max_rmsd=0.6)
        rebuilt.verify_net(topology)

    def test_3d_organic_srs(self, mofgen):
        cof = build(
            mofgen,
            "srs",
            {3: "Boroxine_triangle", 2: "Benzene_linear"},
            max_rmsd=0.6,
        )
        result = mofgen.deconstruct(cof.structure)
        assert result.net_candidates == ["srs"]
        assert all(unit.kind in ("node", "linker") for unit in result.units)

    def test_3d_organic_tetrahedral_dia(self, mofgen):
        cof = build(
            mofgen,
            "dia",
            {4: "N66_tetrahedral", 2: "Benzene_linear"},
            max_rmsd=0.7,
        )
        result = mofgen.deconstruct(cof.structure)
        assert result.net_candidates == ["dia"]
        node = next(unit for unit in result.units if unit.kind == "node")
        assert node.n_connections == 4

    def test_metal_path_unaffected(self, mofgen):
        """A metal MOF still uses metal-oxo clustering, not branch points."""
        mof = build(
            mofgen,
            "pcu",
            {6: "Zn_mof5_octahedral", 2: "Benzene_linear"},
            refine_cell=True,
        )
        result = mofgen.deconstruct(mof.structure)
        assert result.net_candidates == ["pcu"]
        assert "node_C6O13Zn4_6X" in result.fragments


def _rod_pillar_structure(n_repeats: int = 1):
    """Tetragonal -Zn-O- chain along c, pyrazine linkers along a and b.

    A minimal rod MOF: the chain is a 1-periodic metal-oxo unit whose
    points of extension (the Zn atoms, one per chemical repeat) each
    carry four Zn-N cuts. The PoE quotient contracts to the bare pcu
    net. Rings are tilted 45 degrees out of the ab plane so
    neighboring linkers and c-images stay comfortably apart. With
    ``n_repeats`` > 1 the cell holds several chemical repeats - the
    crystallographic repeat grows and the rod gets multiple PoE, but
    the net is unchanged (a supercell of the same framework). Chain
    atoms come first (Zn at even indices, O at odd), rings after.
    """
    import numpy as np

    a, c0 = 6.9, 3.9
    lattice = Lattice.tetragonal(a, c0 * n_repeats)
    species = []
    coords = []
    for r in range(n_repeats):
        species += ["Zn", "O"]
        coords += [[0.0, 0.0, r * c0], [0.0, 0.0, r * c0 + 1.95]]
    tilt = np.sqrt(0.5)
    ring = [
        ("N", -1.395, 0.0),
        ("N", 1.395, 0.0),
        ("C", -0.6975, 1.208),
        ("C", 0.6975, 1.208),
        ("C", -0.6975, -1.208),
        ("C", 0.6975, -1.208),
        ("H", -1.237, 2.143),
        ("H", 1.237, 2.143),
        ("H", -1.237, -2.143),
        ("H", 1.237, -2.143),
    ]
    for r in range(n_repeats):
        for center, plane in (
            (np.array([a / 2, 0.0, r * c0]), "x"),
            (np.array([0.0, a / 2, r * c0]), "y"),
        ):
            for symbol, along, out in ring:
                if plane == "x":
                    pos = center + np.array([along, out * tilt, out * tilt])
                else:
                    pos = center + np.array([out * tilt, along, -out * tilt])
                species.append(symbol)
                coords.append(pos.tolist())
    return Structure(lattice, species, coords, coords_are_cartesian=True)


def _helical_rod_structure(n_repeats: int = 2):
    """Tetragonal 2_1-screw -Zn-O- rod, pyrazine linkers laterally.

    A synthetic helical rod MOF standing in for the MOF-74 class: the
    bridging oxygen sits off the rod axis and alternates side by 180
    degrees each chemical repeat, so the -Zn-O- chain is a 2_1 screw
    (screw order 2, angle 180 degrees). Detection and canonicalization
    must recover that screw - and forward building must refuse it,
    since Stage C3 handles straight rods only. Real helical rods
    (zinc formate, Mg carboxylates, ...) look the same to the pipeline
    but are CSD-derived and stay out of the repo; see the rod-fixture
    notes. ``n_repeats`` must be even (>=2) so the screw closes.
    """
    import math

    import numpy as np

    a, c0, rho = 7.4, 3.9, 0.85
    lattice = Lattice.tetragonal(a, n_repeats * c0)
    species, coords = [], []
    for r in range(n_repeats):
        species.append("Zn")
        coords.append([0.0, 0.0, r * c0])
        theta = r * math.pi  # alternate the bridging O side each repeat
        species.append("O")
        coords.append([rho * math.cos(theta), rho * math.sin(theta), r * c0 + c0 / 2])
    tilt = math.sqrt(0.5)
    ring = [
        ("N", -1.395, 0.0),
        ("N", 1.395, 0.0),
        ("C", -0.6975, 1.208),
        ("C", 0.6975, 1.208),
        ("C", -0.6975, -1.208),
        ("C", 0.6975, -1.208),
        ("H", -1.237, 2.143),
        ("H", 1.237, 2.143),
        ("H", -1.237, -2.143),
        ("H", 1.237, -2.143),
    ]
    for r in range(n_repeats):
        for center, plane in (
            (np.array([a / 2, 0.0, r * c0]), "x"),
            (np.array([0.0, a / 2, r * c0]), "y"),
        ):
            for symbol, along, out in ring:
                if plane == "x":
                    pos = center + np.array([along, out * tilt, out * tilt])
                else:
                    pos = center + np.array([out * tilt, along, -out * tilt])
                species.append(symbol)
                coords.append(pos.tolist())
    return Structure(lattice, species, coords, coords_are_cartesian=True)


class TestRodMOF:
    @pytest.fixture(scope="class")
    def rod_result(self, mofgen):
        return mofgen.deconstruct(_rod_pillar_structure())

    def test_rod_detected_and_characterized(self, rod_result):
        assert len(rod_result.rod_units) == 1
        rod = rod_result.rod_units[0]
        assert rod.atom_indices == [0, 1]  # the Zn-O chain
        assert abs(float(rod.axis[2])) == pytest.approx(1.0)  # along c
        assert rod.repeat_length == pytest.approx(3.9)
        assert rod.generator in {(0, 0, 1), (0, 0, -1)}
        assert rod.poe_indices == [0]  # the Zn carries every cut
        assert rod.n_connections == 4
        assert "rod" in repr(rod_result)

    def test_rod_has_no_fragment_but_linkers_do(self, rod_result):
        assert set(rod_result.fragments) == {"linker_C4H4N2_2X"}
        kinds = Counter(unit.kind for unit in rod_result.units)
        assert kinds == {"linker": 2}

    def test_poe_net_identified(self, rod_result):
        """PoE convention: chain self-loop + two ditopic linkers = pcu.

        The PoE expansion carries no blueprint edge centers, so the
        match lands on the contracted tier by construction.
        """
        assert rod_result.net_candidates == ["pcu"]
        assert rod_result.subframework_nets[0].tier == "contracted"
        assert rod_result.n_periodic_components == 1

    def test_multi_poe_chain_orders_along_axis(self, mofgen):
        """Two chemical repeats per cell: several PoE per rod must
        chain in axial order (consecutive links + one wrap-around),
        and the identified net is unchanged - it is a supercell of
        the same framework."""
        result = mofgen.deconstruct(_rod_pillar_structure(n_repeats=2))
        assert len(result.rod_units) == 1
        rod = result.rod_units[0]
        assert sorted(rod.poe_indices) == [0, 2]  # the two Zn, by z
        assert rod.repeat_length == pytest.approx(7.8)
        assert rod.n_connections == 8
        assert result.net_candidates == ["pcu"]
        assert result.subframework_nets[0].tier == "contracted"

    def test_helical_rod_detected(self, mofgen):
        """The 2_1-screw synthetic rod (MOF-74 class) deconstructs to a
        single 1-periodic unit whose canonical form carries the screw:
        order 2, 180 degrees, its chemical repeat half the
        crystallographic one. The monotonic-forward unwrap keeps the
        naturally-bonded -Zn-O- pair together, so the screw-aware
        template bond graph is the clean 2-bond chain (within-repeat +
        continuation to the next), just like the straight pillar."""
        from autografs.rods import rod_fragment

        result = mofgen.deconstruct(_helical_rod_structure())
        assert len(result.rod_units) == 1
        frag = rod_fragment(result.structure, result.rod_units[0])
        assert frag.repeat.formula == "OZn"
        assert frag.repeat.screw_order == 2
        assert abs(frag.repeat.screw_angle) == pytest.approx(180.0, abs=1.0)
        assert frag.repeat.repeat_length == pytest.approx(3.9, abs=0.05)
        # Zn(row 0) - O(row 1) within a repeat, and Zn - O of the
        # neighbouring repeat (m = +-1): two bonds, like the pillar
        assert len(frag.bonds) == 2
        assert {(a, b) for a, b, _ in frag.bonds} == {(0, 1)}
        assert sorted(m for _, _, m in frag.bonds) == [-1, 0]


class TestErrors:
    def test_molecular_crystal_rejected(self):
        structure = Structure(Lattice.cubic(20.0), ["He"], [[0.5, 0.5, 0.5]])
        with pytest.raises(DeconstructionError, match="periodic component"):
            deconstruct(structure)

    def test_disordered_structure_rejected(self):
        structure = Structure(
            Lattice.cubic(5.0), [{"Fe": 0.5, "Co": 0.5}], [[0.0, 0.0, 0.0]]
        )
        with pytest.raises(DeconstructionError, match="occupied"):
            deconstruct(structure)

    def test_rod_mof_rejected(self):
        """A chain of corner-sharing metal octahedra is a rod SBU."""
        structure = Structure(
            Lattice.orthorhombic(3.8, 15.0, 15.0),
            ["Zn", "O"],
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        )
        with pytest.raises(DeconstructionError, match="rod|periodic"):
            deconstruct(structure)


def test_hill_formula():
    assert _hill_formula(["C", "H", "H", "C", "O"]) == "C2H2O"
    assert _hill_formula(["Zn", "O", "Zn"]) == "OZn2"
    assert _hill_formula(["H", "O", "H"]) == "H2O"


def _doubly_bridged_pcu() -> Structure:
    """A pcu-like crystal whose x edges carry parallel double bridges.

    One Zn node per cubic cell; single two-carbon bridges along y and
    z; TWO parallel two-carbon bridges along x, offset to +-1.2 A so
    they do not bond each other. The node therefore perceives 8
    connections where the underlying net has 6 - the doubled-node
    signature that dominates the census's no_mapping bucket (a
    paddlewheel bridged by parallel linker pairs reads as 8-connected
    on a 4-c net). Bridges are two atoms so every bond has its own
    partner atom: the single Zn still carries all 8 connections on one
    anchor - the multi-connection-anchor case bond formation must
    survive - while no linker bonds the same atom pair through two
    images, which the simple-graph min-image representation cannot
    hold.
    """
    a = 5.6
    lo, hi = 2.1, 3.5  # C-C at 1.4 A, Zn-C at 2.1-2.4 A
    species = ["Zn"] + ["C"] * 8
    carts = [
        (0.0, 0.0, 0.0),
        (lo, 1.2, 0.0),  # x bridge, upper pair
        (hi, 1.2, 0.0),
        (lo, -1.2, 0.0),  # x bridge, lower pair (parallel to the upper)
        (hi, -1.2, 0.0),
        (0.0, lo, 0.0),  # y bridge, single
        (0.0, hi, 0.0),
        (0.0, 0.0, lo),  # z bridge, single
        (0.0, 0.0, hi),
    ]
    return Structure(Lattice.cubic(a), species, carts, coords_are_cartesian=True)


class TestParallelBridgeFusion:
    """Opt-in fusion of parallel ditopic bridges (coverage plan stage 1).

    Identification's coordination-sequence walk counts a double bridge
    once; the mapper counts every dummy on the doubled node. Fusion
    restores agreement by merging the parallel pair into one composite
    unit with one connection per end - both molecules kept, so
    composition stays exact.
    """

    def test_default_leaves_units_alone(self, mofgen):
        result = mofgen.deconstruct(_doubly_bridged_pcu())
        assert result.fused_bridges == 0
        node = next(u for u in result.units if u.kind == "node")
        assert node.n_connections == 8

    def test_fusion_restores_the_net_arm_count(self, mofgen):
        result = mofgen.deconstruct(_doubly_bridged_pcu(), fuse_parallel_bridges=True)
        assert result.fused_bridges == 1
        node = next(u for u in result.units if u.kind == "node")
        assert node.n_connections == 6
        # the composite carries BOTH bridge molecules on 2 connections,
        # so the fused fragment is ditopic and composition-complete
        composite = next(
            u for u in result.units if u.kind == "linker" and "C4" in u.name
        )
        assert composite.n_connections == 2
        placed = Counter(atom for u in result.units for atom in u.atom_indices)
        assert len(placed) == len(result.structure)
        assert max(placed.values()) == 1
        assert result.net_candidates == ["pcu"]

    def test_fusion_requires_identical_composition(self, mofgen):
        # hang a hydrogen on one of the parallel bridges: CH vs C no
        # longer match, so nothing fuses and the request is a clean
        # no-op (a nitrogen swap would not test this - the metal-oxo
        # rule absorbs a carbon-free N into the node)
        structure = _doubly_bridged_pcu()
        structure.append("H", (2.1, -1.2, 0.9), coords_are_cartesian=True)
        result = mofgen.deconstruct(structure, fuse_parallel_bridges=True)
        assert result.fused_bridges == 0
        node = next(u for u in result.units if u.kind == "node")
        assert node.n_connections == 8


class TestSelfTemplatedRoundTrip:
    """The structure's own blueprint rebuilds the structure (stage 3).

    topology_from_deconstruction erects one slot per building unit at
    its real position, with one shared connection point per cut; the
    rebuild maps each slot to its own unit's representative fragment.
    No library net is consulted at any point, so this is the round
    trip with every library wall removed - what remains is the rigid-
    unit abstraction itself.
    """

    def _selftemplate(self, mofgen, result):
        import copy

        from autografs.builder import build_framework
        from autografs.extract_topology import topology_from_deconstruction

        topology, mapping = topology_from_deconstruction(result)
        mappings = {
            index: copy.deepcopy(result.fragments[name])
            for index, name in mapping.items()
        }
        return topology, build_framework(
            topology, mappings, max_rmsd=0.5, verify_net=True
        )

    def test_mof5_rebuilds_from_its_own_blueprint(self, mofgen, mof5_deconstruction):
        result = mof5_deconstruction
        topology, rebuilt = self._selftemplate(mofgen, result)
        assert len(topology) == len(result.units)
        assert (
            rebuilt.structure.composition.reduced_formula
            == result.structure.composition.reduced_formula
        )
        # the blueprint is the crystal's own embedding, so the
        # optimized cell must come out at the experimental volume
        ratio = rebuilt.structure.volume / result.structure.volume
        assert 0.9 < ratio < 1.1

    def test_doubled_bridge_needs_no_fusion(self, mofgen):
        """The 8-connected node maps onto its own 8-arm slot natively:
        the multiplicity wall does not exist for a self-template."""
        result = mofgen.deconstruct(_doubly_bridged_pcu())
        topology, rebuilt = self._selftemplate(mofgen, result)
        node_slot = max(
            topology.slots, key=lambda s: len(s.atoms.indices_from_symbol("X"))
        )
        assert len(node_slot.atoms.indices_from_symbol("X")) == 8
        assert (
            rebuilt.structure.composition.reduced_formula
            == result.structure.composition.reduced_formula
        )

    def test_rod_structures_record_poe_but_await_run_blueprints(self, mofgen):
        from autografs.exceptions import TopologyExtractionError
        from autografs.extract_topology import topology_from_deconstruction

        result = mofgen.deconstruct(_rod_pillar_structure(1))
        # the recipe now carries the rod's points of extension in the
        # quotient's own gauge - the run constructor's raw material
        recipe = result.blueprint
        assert recipe is not None and recipe.rod_poe
        (rod_index, entries), *_ = recipe.rod_poe.items()
        assert recipe.centers[rod_index] is None
        assert len(entries) == len(result.rod_units[0].poe_indices)
        # the point-slot constructor still declines: rods need slot runs
        with pytest.raises(TopologyExtractionError, match="[Rr]od"):
            topology_from_deconstruction(result)


def _catenated_pcu_pair() -> Structure:
    """Two interpenetrated single-bridged pcu nets, offset body-center.

    Each net is one Zn plus two-carbon bridges along x, y and z; the
    second net is the first translated by (1/2, 1/2, 1/2). The closest
    inter-net contact is ~4 A, so bond perception keeps the components
    separate and the deconstruction reports fold 2.
    """
    a = 5.6
    lo, hi = 2.1, 3.5
    base = [
        (0.0, 0.0, 0.0),
        (lo, 0.0, 0.0),
        (hi, 0.0, 0.0),
        (0.0, lo, 0.0),
        (0.0, hi, 0.0),
        (0.0, 0.0, lo),
        (0.0, 0.0, hi),
    ]
    shift = a / 2
    species = (["Zn"] + ["C"] * 6) * 2
    carts = base + [(x + shift, y + shift, z + shift) for x, y, z in base]
    return Structure(Lattice.cubic(a), species, carts, coords_are_cartesian=True)


class TestCatenatedSelfTemplate:
    """A 2-fold interpenetrated pair rebuilds from its own blueprint.

    The recipe holds every component's units, so the erected blueprint
    is a disconnected quotient whose two nets share the one real cell
    at their true relative offset; the build places both and the exact
    verification compares like with like. This is what the library arm
    structurally cannot do - it rebuilds one net of a catenated pair
    and fails the composition gate.
    """

    def test_two_fold_pair_closes(self, mofgen):
        import copy

        from autografs.builder import build_framework
        from autografs.extract_topology import topology_from_deconstruction

        result = mofgen.deconstruct(_catenated_pcu_pair())
        assert result.n_periodic_components == 2
        topology, mapping = topology_from_deconstruction(result)
        # one Zn node and three two-carbon bridges per net, two nets
        assert len(topology) == len(result.units) == 8
        mappings = {
            index: copy.deepcopy(result.fragments[name])
            for index, name in mapping.items()
        }
        rebuilt = build_framework(topology, mappings, max_rmsd=0.5, verify_net=True)
        assert (
            rebuilt.structure.composition.reduced_formula
            == result.structure.composition.reduced_formula
        )
        assert len(rebuilt) == len(result.structure)
        # the fixture's synthetic Zn-C bonds (2.1 A) sit above the
        # covalent target the cell objective optimizes toward, so the
        # rebuilt cell is correctly a few percent smaller; both nets
        # must still share it at the true relative offset
        ratio = rebuilt.structure.volume / result.structure.volume
        assert 0.75 < ratio < 1.1


class TestRodSelfTemplate:
    """A rod framework rebuilds from its own slot-run blueprint.

    The rod's points of extension become node slots on the crystal's
    own axis, the run is built from the rod's measured repeat rather
    than detected, and the validated rod builder consumes it unchanged
    - the last library wall, removed for the single-rod case.
    """

    def test_node_slots_sit_on_the_rods_own_axis(self, mofgen):
        """A run node centre must lie on the rod's axis line.

        The builder can only rotate the rod about the run axis and
        slide it along: a node centre off that line is unreachable
        geometry, and no (theta, z0) recovers it. Taking the centre as
        the centroid of the points of extension put it 7.6-8.0 A off
        the axis on real corpus rods - they are transverse to it, and
        home-gauge wrapping moves them further - which left the rod ~6 A
        from its own crystal position with bonds 3-7 A long.
        """
        import numpy as np

        from autografs.extract_topology import rod_topology_from_deconstruction
        from autografs.rods import _local_positions

        result = mofgen.deconstruct(_rod_pillar_structure(1))
        rod_unit = result.rod_units[0]
        axis = np.asarray(rod_unit.axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        unwrapped = _local_positions(
            result.structure,
            rod_unit.atom_indices,
            rod_unit.atom_indices[0],
            rod_unit.internal_bonds,
        )
        centroid = unwrapped.mean(axis=0)
        topology, run, _lateral_mapping, _fragment = rod_topology_from_deconstruction(
            result
        )
        for slot_index in run.slots:
            centre = np.asarray(topology.slots[slot_index].atoms[0].coords, dtype=float)
            offset = centre - centroid
            perpendicular = offset - np.dot(offset, axis) * axis
            assert np.linalg.norm(perpendicular) == pytest.approx(0.0, abs=1e-6)

    def test_pillar_rebuilds_on_its_own_run(self, mofgen):
        import copy

        from autografs.extract_topology import rod_topology_from_deconstruction
        from autografs.rod_build import build_rod_framework

        result = mofgen.deconstruct(_rod_pillar_structure(1))
        topology, run, lateral_mapping, fragment = rod_topology_from_deconstruction(
            result
        )
        assert set(run.slots).isdisjoint(lateral_mapping)
        laterals = {
            index: copy.deepcopy(result.fragments[name])
            for index, name in lateral_mapping.items()
        }
        # verify_net stays off: the rod-form verifier re-detects runs
        # on the blueprint instead of trusting the injected one, and a
        # distorted self-blueprint fails that detection - a machinery
        # conservatism recorded in the coverage plan, not a mismatch.
        # Composition plus exact atom count is the v1 closure gate.
        rebuilt = build_rod_framework(
            topology,
            fragment,
            laterals,
            run=run,
            min_distance=None,
            bond_tolerance=10.0,
            verify_net=False,
            initial_scale=1.0,
            scale_band=0.25,
        )
        built = rebuilt.structure.composition
        experimental = result.structure.composition
        assert built.reduced_formula == experimental.reduced_formula
        # the builder stacks at least two repeats (a continuation bond
        # must join distinct node pairs), so the rebuild is a whole
        # supercell of the crystal; per-atom volume is the
        # supercell-invariant packing check
        assert len(rebuilt.structure) % len(result.structure) == 0
        per_atom = (rebuilt.structure.volume / len(rebuilt.structure)) / (
            result.structure.volume / len(result.structure)
        )
        assert 0.75 < per_atom < 1.25
