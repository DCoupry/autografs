"""Pre-build assessment and post-build validation.

These exist because "build returned a Framework" was the only signal a
user had, and it is a weak one: measured over 1895 builds, the stock
defaults return 1105 structures that do not close and 521 that overlap.
``build(..., strict=True)`` refuses them and ``Framework.validate()``
explains them.
"""

import networkx
import numpy as np
import pytest

from autografs import Autografs
from autografs.exceptions import AlignmentError, OverlapError
from autografs.validation import (
    STRICT_BOND_TOLERANCE,
    STRICT_MIN_DISTANCE,
    assess_build,
    edge_scales,
    free_molecule_count,
    inter_unit_bond_deviations,
    scale_spread,
    validate_framework,
)

FIXTURE_PATH = "tests/data/topologies_fixture.json"


def _atom(symbol: str, coord) -> dict:
    """A node carrying every attribute Framework's views require."""
    return {
        "symbol": symbol,
        "coord": np.asarray(coord, dtype=float),
        "slot": 999,
        "sbu": "guest",
        "tag": 0,
        "ufftype": f"{symbol}_3",
    }


@pytest.fixture(scope="module")
def mofgen():
    return Autografs(topofile=FIXTURE_PATH)


def _mof5_mappings(topology):
    mappings = {}
    for slot_type in topology.mappings:
        arms = len(slot_type.atoms.indices_from_symbol("X"))
        mappings[slot_type] = {6: "Zn_mof5_octahedral", 2: "Benzene_linear"}[arms]
    return mappings


@pytest.fixture(scope="module")
def mof5(mofgen):
    topology = mofgen.topologies["pcu"]
    return mofgen.build(topology, _mof5_mappings(topology), max_rmsd=0.5)


class TestValidate:
    def test_a_good_build_passes_every_check(self, mof5):
        report = mof5.validate()
        assert report.ok, str(report)
        assert not report.failures
        assert "OK" in str(report)

    def test_checks_carry_their_numbers(self, mof5):
        """A boolean would hide whether a miss was near or hopeless."""
        report = mof5.validate()
        contact = report["contact"]
        assert contact.passed
        assert contact.value == pytest.approx(mof5.min_contact())
        assert contact.threshold == STRICT_MIN_DISTANCE
        closure = report["closure"]
        assert closure.threshold == STRICT_BOND_TOLERANCE
        assert closure.value is not None

    def test_descriptors_are_reported_whatever_the_verdict(self, mof5):
        descriptors = mof5.validate().descriptors
        assert descriptors["n_atoms"] == len(mof5.structure)
        assert descriptors["n_free_molecules"] == 0
        assert descriptors["n_components"] == 1
        assert descriptors["n_distinct_sbus"] == 2

    def test_a_tightened_threshold_fails_the_report(self, mof5):
        report = mof5.validate(min_distance=99.0)
        assert not report.ok
        assert [check.name for check in report.failures] == ["contact"]
        assert "contact" in str(report)

    def test_skipped_checks_are_absent_not_passed(self, mof5):
        report = mof5.validate(bond_tolerance=None, min_distance=None)
        names = {check.name for check in report.checks}
        assert "closure" not in names
        assert "contact" not in names

    def test_unmeasurable_closure_is_a_failure_not_a_pass(self, mof5):
        """A framework with no measurable inter-unit bond is unknown.

        Passing on a measurement that does not exist is how a bare node
        net scores as a closed framework.
        """
        bare = mof5.copy() if hasattr(mof5, "copy") else mof5
        graph = networkx.Graph(cell=np.asarray(bare.graph.graph["cell"], float))
        # one lone atom: no edges at all, so nothing to measure
        graph.add_node(0, **_atom("Zn", np.zeros(3)))
        from autografs.framework import Framework

        lonely = Framework(graph, name="bare")
        report = validate_framework(lonely, min_distance=None)
        closure = report["closure"]
        assert not closure.passed
        assert closure.value is None
        assert "no inter-unit bond" in closure.detail


class TestFreeMolecules:
    def test_a_connected_framework_has_none(self, mof5):
        components, free = free_molecule_count(mof5)
        assert (components, free) == (1, 0)

    def test_a_floating_molecule_is_detected(self, mof5):
        """A periodic net plus loose debris is not a framework.

        Two gra builds passed closure and contact this way, carrying a
        free Ni2 dimer and free CO2 fragments; only lammps-interface
        noticed, and only by refusing to run.
        """
        graph = mof5.graph.copy()
        cell = np.asarray(graph.graph["cell"], float)
        # a 0-periodic pair sitting in the middle of the cell
        base = max(graph.nodes()) + 1
        centre = 0.5 * cell.sum(axis=0)
        graph.add_node(base, **_atom("O", centre))
        graph.add_node(base + 1, **_atom("O", centre + np.array([1.2, 0.0, 0.0])))
        graph.add_edge(base, base + 1, bond_order=1.0)
        from autografs.framework import Framework

        polluted = Framework(graph, name="polluted")
        components, free = free_molecule_count(polluted)
        assert components == 2
        assert free == 1
        report = validate_framework(polluted, bond_tolerance=None, min_distance=None)
        assert not report.ok
        assert report["connectivity"].value == 1.0

    def test_interpenetration_is_not_a_defect(self, mof5):
        """Several PERIODIC components are legitimate catenation."""
        graph = mof5.graph.copy()
        cell = np.asarray(graph.graph["cell"], float)
        offset = 0.5 * cell.sum(axis=0)
        base = max(graph.nodes()) + 1
        mapping = {}
        for node in list(mof5.graph.nodes()):
            mapping[node] = base + node
            data = dict(mof5.graph.nodes[node])
            data["coord"] = np.asarray(data["coord"], float) + offset
            data["slot"] = data.get("slot", 0) + 1000
            graph.add_node(base + node, **data)
        for node_a, node_b, data in mof5.graph.edges(data=True):
            graph.add_edge(mapping[node_a], mapping[node_b], **data)
        from autografs.framework import Framework

        catenated = Framework(graph, name="catenated")
        components, free = free_molecule_count(catenated)
        assert components == 2
        assert free == 0


class TestStrictBuild:
    def test_strict_accepts_a_good_build(self, mofgen):
        topology = mofgen.topologies["pcu"]
        mof = mofgen.build(topology, _mof5_mappings(topology), strict=True)
        assert mof.validate().ok

    def test_strict_does_not_override_an_explicit_threshold(self, mofgen):
        """An explicit bond_tolerance means that value, not the preset."""
        topology = mofgen.topologies["pcu"]
        with pytest.raises(AlignmentError):
            mofgen.build(
                topology,
                _mof5_mappings(topology),
                strict=True,
                bond_tolerance=1e-9,
            )

    def test_strict_refuses_an_overlapping_build(self, mofgen):
        topology = mofgen.topologies["pcu"]
        with pytest.raises(OverlapError):
            mofgen.build(
                topology,
                _mof5_mappings(topology),
                strict=True,
                min_distance=99.0,
            )

    def test_defaults_are_unchanged_by_the_feature(self, mofgen):
        """The whole point of opt-in: existing calls must not move."""
        topology = mofgen.topologies["pcu"]
        loose = mofgen.build(topology, _mof5_mappings(topology))
        assert loose.validate(min_distance=None, bond_tolerance=None).ok
        # a build that fails a strict check is still RETURNED by default
        strict_report = loose.validate(min_distance=99.0)
        assert not strict_report.ok


class TestAssess:
    def test_a_pinned_net_with_fitting_units_is_high(self, mofgen):
        topology = mofgen.topologies["pcu"]
        report = mofgen.assess(topology, _mof5_mappings(topology))
        assert report.confidence == "high"
        assert report.feasible
        assert report.n_free == 0
        assert all(slot.fits for slot in report.slots)

    def test_wrong_connectivity_is_infeasible_and_says_so(self, mofgen):
        """The user should learn this without spending a build."""
        topology = mofgen.topologies["pcu"]
        mappings = {}
        for slot_type in topology.mappings:
            arms = len(slot_type.atoms.indices_from_symbol("X"))
            # deliberately put a ditopic unit in the 6-connected slot
            mappings[slot_type] = "Benzene_linear" if arms == 2 else "Benzene_linear"
        with pytest.raises(AlignmentError):
            # build refuses it too, but only after doing the work
            mofgen.build(topology, mappings)

    def test_slot_margin_distinguishes_comfortable_from_strained(self, mofgen):
        topology = mofgen.topologies["pcu"]
        report = mofgen.assess(topology, _mof5_mappings(topology))
        node = next(slot for slot in report.slots if slot.slot_arms == 6)
        assert node.rmsd is not None
        assert node.margin == pytest.approx(node.threshold - node.rmsd)
        assert node.margin > 0

    def test_freedom_drives_the_class(self, mofgen):
        """n_free is the descriptor the class turns on."""
        topology = mofgen.topologies["pcu"]
        validated, _ = mofgen._validate_mappings(
            topology=topology, mappings=_mof5_mappings(topology)
        )
        assert assess_build(topology, validated).confidence == "high"

    def test_assess_costs_no_build(self, mofgen):
        """It must not raise on a combination build would reject."""
        topology = mofgen.topologies["pcu"]
        mappings = dict.fromkeys(topology.mappings, "Benzene_linear")
        report = mofgen.assess(topology, mappings)
        assert report.confidence == "infeasible"
        assert not report.feasible
        assert "connectivity" in str(report)


class TestStericBalance:
    """The packing predictor, and the selection fix it enables.

    Clashing builds overlap between *bonded* slots (measured: 344 of 364
    sub-1.2 A pairs on ith-d), i.e. the cell that closes a bond is too
    small for the units sitting on it. Each edge wants its own scale
    ``(sbu arms) / (slot arms)`` and the cell has one, so the spread of
    those requirements is what predicts the clash.
    """

    def test_identical_proportions_need_one_scale(self, mofgen):
        topology = mofgen.topologies["pcu"]
        validated, _ = mofgen._validate_mappings(
            topology=topology, mappings=_mof5_mappings(topology)
        )
        scales = edge_scales(topology, validated)
        assert scales.size > 1
        spread = scale_spread(scales)
        assert spread is not None
        assert spread == pytest.approx(1.0, abs=0.35)

    def test_a_longer_linker_raises_the_edge_scale(self, mofgen):
        """Swapping in a longer ditopic unit must move the requirement.

        The sieve cannot see this: for a 2-connected slot compatibility
        is vacuous, since any two arms seen from their own centroid are
        antiparallel, so length never enters.
        """
        topology = mofgen.topologies["pcu"]
        short = _mof5_mappings(topology)
        long = dict(short)
        edge_type = next(
            k for k in topology.mappings if len(k.atoms.indices_from_symbol("X")) == 2
        )
        long[edge_type] = "Bis_phenylethynylbenzene_linear"
        # both are "compatible" with the same slot
        assert edge_type.has_compatible_symmetry(mofgen.sbu["Benzene_linear"])
        assert edge_type.has_compatible_symmetry(
            mofgen.sbu["Bis_phenylethynylbenzene_linear"]
        )
        scales = {}
        for label, mapping in (("short", short), ("long", long)):
            validated, _ = mofgen._validate_mappings(
                topology=topology, mappings=mapping
            )
            scales[label] = float(np.median(edge_scales(topology, validated)))
        assert scales["long"] > scales["short"]

    def test_spread_reaches_the_assessment(self, mofgen):
        topology = mofgen.topologies["pcu"]
        report = mofgen.assess(topology, _mof5_mappings(topology))
        assert report.scale_spread is not None
        assert report.scale_spread == pytest.approx(1.0, abs=0.35)

    def test_suggest_mappings_is_buildable_and_balanced(self, mofgen):
        topology = mofgen.topologies["pcu"]
        mappings, spread = mofgen.suggest_mappings(topology)
        assert mappings, "pcu is coverable by the shipped library"
        assert set(mappings) == set(topology.mappings)
        assert spread is not None
        # a well-balanced choice, not a certified-optimal one: the search
        # is over a family of common target ratios, and it is deliberately
        # NOT pushed to the spread's floor (doing so was measured to make
        # the realized structures worse). So the claim is that the result
        # is balanced and buildable, and the population-level gain is
        # documented on balance_mappings rather than asserted per net.
        assert spread < 1.1
        mofgen.build(topology, mappings, max_rmsd=0.5)

    def test_suggestion_is_balanced_across_the_fixture_nets(self, mofgen):
        """Every coverable fixture net should get a balanced proposal."""
        seen = 0
        for name in ("pcu", "dia", "srs", "hcb", "sql"):
            try:
                topology = mofgen.topologies[name]
            except KeyError:
                continue
            mappings, spread = mofgen.suggest_mappings(topology)
            if not mappings:
                continue
            seen += 1
            assert spread is None or spread < 1.5, (name, spread)
        assert seen >= 3

    def test_suggest_reports_nothing_when_a_slot_is_uncoverable(self, mofgen):
        topology = mofgen.topologies["pcu"]
        mappings, spread = mofgen.suggest_mappings(topology, subset=["Benzene_linear"])
        assert mappings == {}
        assert spread is None


class TestBondDeviations:
    def test_internal_bonds_are_excluded(self, mof5):
        deviations = inter_unit_bond_deviations(mof5)
        assert deviations.size
        assert deviations.size < mof5.graph.number_of_edges()

    def test_a_good_build_closes(self, mof5):
        assert float(inter_unit_bond_deviations(mof5).max()) < STRICT_BOND_TOLERANCE
