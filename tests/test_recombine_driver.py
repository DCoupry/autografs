"""Smoke tests for the finite recombination driver (scripts/generative).

The driver is a script, not a package module; it is imported via a path
insertion so CI catches breakage of the library APIs it leans on.

The ordering test is a real regression guard rather than a formality.
``list_building_units`` keys its result in SBU-iteration order while
``topology.mappings`` has its own, and the first version of this driver
zipped a sampled choice against the latter. Every build then landed its
units on the wrong slot types, and the builder's honest complaint
("has 2 connection points but slot 3 needs 4") was indistinguishable
from a chemistry result: a 40-net sweep reported 74/80 failures that
were entirely the driver's.
"""

import os
import sys

import pytest

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts", "generative")
FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "topologies_fixture.json"
)


def _load(name):
    sys.path.insert(0, SCRIPTS)
    try:
        return __import__(name)
    finally:
        sys.path.pop(0)


@pytest.fixture(scope="module")
def recombine():
    return _load("recombine")


@pytest.fixture(scope="module")
def mofgen():
    from autografs import Autografs

    return Autografs(topofile=FIXTURE_PATH)


@pytest.fixture(scope="module")
def subset():
    """A few shipped SBUs spanning the arities the fixture nets need."""
    return [
        "Benzene_linear",
        "Zn_mof5_octahedral",
        "Benzene_triangle",
        "Benzene_rectangle",
    ]


class TestCoverableNets:
    def test_plans_are_arity_consistent(self, recombine, mofgen, subset):
        """Every offered SBU must fit the slot type its position names.

        This is the invariant the ordering bug violated: it is checked
        against the blueprint's OWN ordering, so it fails if
        ``slot_order`` is ever dropped or re-derived from the wrong dict.
        """
        plans = recombine.coverable_nets(mofgen, subset, verbose=False)
        assert plans, "the fixture nets should be coverable by these SBUs"
        for plan in plans:
            topology = mofgen.topologies[plan["net"]]
            ordering = list(topology.mappings)
            assert len(plan["slot_order"]) == len(plan["options"])
            for position, names in zip(
                plan["slot_order"], plan["options"], strict=True
            ):
                wanted = recombine._slot_arity(ordering[position])
                for name in names:
                    offered = len(mofgen.sbu[name].atoms.indices_from_symbol("X"))
                    assert offered == wanted, (
                        f"{plan['net']}: {name} has {offered} connection points "
                        f"at a position needing {wanted}"
                    )

    def test_only_restricts_to_named_nets(self, recombine, mofgen, subset):
        plans = recombine.coverable_nets(mofgen, subset, only=["pcu"], verbose=False)
        assert [plan["net"] for plan in plans] == ["pcu"]

    def test_empty_2c_leaves_the_two_connected_slots_out(
        self, recombine, mofgen, subset
    ):
        plans = recombine.coverable_nets(
            mofgen, subset, only=["pcu"], empty_2c=True, verbose=False
        )
        assert len(plans) == 1
        plan = plans[0]
        assert plan["empty_slot_types"], "pcu carries EDGE_CENTER 2-c slots"
        ordering = list(mofgen.topologies["pcu"].mappings)
        for position in plan["empty_slot_types"]:
            assert recombine._slot_arity(ordering[position]) == 2
        # an emptied slot type is never also offered a unit
        assert not set(plan["empty_slot_types"]) & set(plan["slot_order"])


class TestAttempt:
    def test_a_coverable_plan_builds_and_is_measured(self, recombine, mofgen, subset):
        plan = recombine.coverable_nets(mofgen, subset, only=["pcu"], verbose=False)[0]
        import numpy as np

        choice = recombine.sample_combinations(plan, 1, np.random.default_rng(0))[0]
        record = recombine.attempt(
            mofgen, plan["net"], choice, plan["slot_order"], max_rmsd=0.5
        )
        assert record["outcome"] == "built", record.get("error")
        # the measured quantities the funnel is built on
        assert record["n_atoms"] > 0
        assert record["bond_residual"]["max"] >= 0.0
        assert "closed" in record and "clash_free" in record
        # the fingerprint must be recorded even though no `realized` set
        # was passed: workers are never given one, and gating the
        # fingerprint on it made a freshly-harvested sweep report
        # novelty as unmeasured
        assert record["fingerprint"].startswith("pcu:")
        assert "novel" not in record

    def test_novelty_is_decided_against_the_realized_set(
        self, recombine, mofgen, subset
    ):
        plan = recombine.coverable_nets(mofgen, subset, only=["pcu"], verbose=False)[0]
        import numpy as np

        choice = recombine.sample_combinations(plan, 1, np.random.default_rng(0))[0]
        seen = recombine.attempt(
            mofgen, "pcu", choice, plan["slot_order"], max_rmsd=0.5, realized=set()
        )
        assert seen["novel"] is True
        known = recombine.attempt(
            mofgen,
            "pcu",
            choice,
            plan["slot_order"],
            max_rmsd=0.5,
            realized={seen["fingerprint"]},
        )
        assert known["novel"] is False

    def test_emptying_the_edges_drops_their_atoms(self, recombine, mofgen, subset):
        """Emptying pcu's edge centers must change what gets built.

        Both builds use the same 6-connected node, so the only
        difference is whether a linker sits in every edge - and the
        emptied build must therefore be strictly smaller.
        """
        import numpy as np

        ordering = list(mofgen.topologies["pcu"].mappings)
        built = {}
        for label, empty_2c in (("filled", False), ("empty", True)):
            plan = recombine.coverable_nets(
                mofgen, subset, only=["pcu"], empty_2c=empty_2c, verbose=False
            )[0]
            choice = recombine.sample_combinations(plan, 1, np.random.default_rng(0))[0]
            # pin the node so the two arms differ only in the edges
            choice = tuple(
                "Zn_mof5_octahedral"
                if recombine._slot_arity(ordering[position]) == 6
                else name
                for position, name in zip(plan["slot_order"], choice, strict=True)
            )
            record = recombine.attempt(
                mofgen,
                "pcu",
                choice,
                plan["slot_order"],
                max_rmsd=0.5,
                empty_slot_types=plan["empty_slot_types"],
            )
            assert record["outcome"] == "built", (label, record.get("error"))
            built[label] = record
        assert built["empty"]["n_atoms"] < built["filled"]["n_atoms"]


class TestSummary:
    def test_freedom_bins_separate_pinned_from_free(self, recombine):
        assert recombine._freedom_bin(0) == "0 (pinned)"
        assert recombine._freedom_bin(1) == "1"
        assert recombine._freedom_bin(3) == "2-5"
        assert recombine._freedom_bin(50) == ">5"
        assert recombine._freedom_bin(None) == "unknown"

    def test_novelty_is_none_when_unmeasured(self, recombine):
        """Reusing a vocabulary leaves novelty unknown, not zero."""
        records = [
            {
                "outcome": "built",
                "min_contact": 2.0,
                "bond_residual": {"max": 0.1},
                "closed": True,
                "clash_free": True,
                "usable": True,
                "n_free": 0,
            }
        ]
        summary = recombine.summarize(records)
        assert summary["novel_usable"] is None
        assert summary["novelty"] is None
        records[0]["novel"] = True
        records[0]["fingerprint"] = "pcu: node + 3xlinker"
        records[0]["net"] = "pcu"
        summary = recombine.summarize(records)
        assert summary["novel_usable"] == 1
        # a generation and a round trip must never be pooled
        assert summary["novelty"]["usable_novel"] == 1
        assert summary["novelty"]["usable_known"] == 0
        assert summary["novelty"]["distinct_usable_novel"] == 1

    def test_a_free_molecule_disqualifies_a_build(self, recombine, mofgen, subset):
        """A periodic net plus loose debris is not a candidate material.

        Emptying 2-connected slots can strand a unit with no bond to
        the framework; two gra builds passed closure and contact while
        carrying a free Ni2 dimer and free CO2 fragments, and only
        lammps-interface noticed. A real build must report zero.
        """
        import numpy as np

        plan = recombine.coverable_nets(mofgen, subset, only=["pcu"], verbose=False)[0]
        choice = recombine.sample_combinations(plan, 1, np.random.default_rng(0))[0]
        record = recombine.attempt(
            mofgen, "pcu", choice, plan["slot_order"], max_rmsd=0.5
        )
        assert record["outcome"] == "built"
        assert record["n_free_molecules"] == 0
        assert record["connected"] is True
        assert record["usable"] == bool(record["closed"] and record["clash_free"])

    def test_a_bare_metal_net_is_not_a_generated_framework(self, recombine):
        """Emptying every edge can leave a bare node net that scores well.

        A few metal atoms with nothing non-bonded inside the contact
        cutoff are "closed and clash-free", and counting them as
        generated materials overstated the empty-slot policy by an
        order of magnitude (54 of 58 usable novel builds).
        """
        bare = {"sbus": ["node_Fe_4X", "node_Fe_4X"]}
        real = {"sbus": ["node_Fe_4X", "linker_C6H4_2X"]}
        assert not recombine.is_genuine_framework(bare)
        assert recombine.is_genuine_framework(real)
        # a linker alone is not a framework either
        assert not recombine.is_genuine_framework({"sbus": ["linker_C6H4_2X"]})

        records = [
            {
                "outcome": "built",
                "min_contact": 3.0,
                "bond_residual": {"max": 0.0},
                "closed": True,
                "clash_free": True,
                "usable": True,
                "novel": True,
                "n_free": 0,
                "net": "kag",
                "fingerprint": "kag: 2xnode_Fe_4X",
                **bare,
            },
            {
                "outcome": "built",
                "min_contact": 2.0,
                "bond_residual": {"max": 0.1},
                "closed": True,
                "clash_free": True,
                "usable": True,
                "novel": True,
                "n_free": 0,
                "net": "pcu",
                "fingerprint": "pcu: node_Fe_4X + linker_C6H4_2X",
                **real,
            },
        ]
        novelty = recombine.summarize(records)["novelty"]
        assert novelty["usable_novel"] == 2
        assert novelty["genuine_usable_novel"] == 1
        assert novelty["degenerate_usable_novel"] == 1
        assert novelty["distinct_nets_genuine"] == 1

    def test_an_unmeasurable_closure_is_not_a_closed_build(self, recombine):
        """bond_residuals returns {} when no inter-unit bond is measurable.

        Emptying the 2-connected slots makes that reachable on real
        nets, and it crashed the reporter before it was guarded. It must
        count as unknown, never as closed.
        """
        records = [
            {
                "outcome": "built",
                "min_contact": 2.0,
                "bond_residual": {},
                "closure_measured": False,
                "closed": False,
                "clash_free": True,
                "usable": False,
                "n_free": 0,
            }
        ]
        summary = recombine.summarize(records)
        assert summary["unmeasurable_closure"] == 1
        assert summary["closed"] == 0
        assert summary["worst_bond"] is None
        recombine._report(1, 1, dict(records[0], net="pcu", formula="C"))

    def test_pinned_and_free_are_reported_separately(self, recombine):
        """A pooled figure is what this stratification exists to prevent."""
        records = [
            {
                "outcome": "built",
                "min_contact": 2.0,
                "bond_residual": {"max": 0.1},
                "closed": True,
                "clash_free": True,
                "usable": True,
                "n_free": 0,
            },
            {
                "outcome": "built",
                "min_contact": 0.2,
                "bond_residual": {"max": 1.9},
                "closed": False,
                "clash_free": False,
                "usable": False,
                "n_free": 4,
            },
        ]
        by_freedom = recombine.summarize(records)["by_freedom"]
        assert by_freedom["0 (pinned)"]["clash_free"] == 1
        assert by_freedom["2-5"]["clash_free"] == 0


class TestNetFreedom:
    def test_pinned_nets_report_zero(self, recombine, mofgen):
        """pcu is fully pinned - the reason it builds correctly today."""
        assert recombine.net_freedom(mofgen.topologies["pcu"]) == 0
