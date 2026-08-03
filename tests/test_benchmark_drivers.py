"""Smoke tests for the netid and throughput drivers (scripts/benchmarks).

The drivers are scripts, not package modules; they are imported via a
path insertion so CI catches breakage of the library APIs they lean on.
(The round-trip driver has its own smoke test alongside its PR.)
"""

import json
import os
import sys

import pytest

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts", "benchmarks")
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
def mofgen():
    from autografs import Autografs

    return Autografs(topofile=FIXTURE_PATH)


@pytest.fixture(scope="module")
def mof5_cif(mofgen, tmp_path_factory):
    topology = mofgen.topologies["pcu"]
    mappings = {}
    for key in topology.mappings:
        conn = len(key.atoms.indices_from_symbol("X"))
        mappings[key] = {6: "Zn_mof5_octahedral", 2: "Benzene_linear"}[conn]
    mof = mofgen.build(topology, mappings=mappings, max_rmsd=0.5)
    path = tmp_path_factory.mktemp("corpus") / "mof5.cif"
    mof.write_cif(path)
    return path


class TestNetId:
    def test_agreement_scored_by_tier(self, mofgen, mof5_cif):
        netid = _load("netid")
        labels = {"mof5.cif": "pcu"}
        payload = netid.run([mof5_cif], mofgen, labels)
        assert payload["outcomes"] == {"agree": 1}
        assert payload["agreement_rate"] == 1.0
        record = payload["structures"]["mof5.cif"]
        assert record["net"] == ["pcu"]
        assert record["tier"] in ("exact", "contracted")
        assert payload["agreement_by_tier"] == {f"agree_{record['tier']}": 1}

    def test_disagreement_and_missing_label(self, mofgen, mof5_cif):
        netid = _load("netid")
        payload = netid.run([mof5_cif], mofgen, {"mof5.cif": "dia"})
        assert payload["outcomes"] == {"disagree": 1}
        assert payload["agreement_rate"] == 0.0
        payload = netid.run([mof5_cif], mofgen, {})
        assert payload["outcomes"] == {"unlabelled": 1}
        assert payload["agreement_rate"] is None

    def test_labels_may_be_lists(self, mofgen, mof5_cif):
        netid = _load("netid")
        payload = netid.run([mof5_cif], mofgen, {"mof5.cif": ["dia", "pcu"]})
        assert payload["outcomes"] == {"agree": 1}


class TestThroughput:
    def test_timings_are_positive_and_identified(self, mofgen):
        throughput = _load("throughput")
        payload = throughput.run(mofgen, ["pcu"], repeats=2)
        record = payload["topologies"]["pcu"]
        assert record["error"] is None
        assert record["n_atoms"] > 0
        assert record["build_seconds"] > 0
        assert record["identify_seconds"] > 0
        assert record["identified_as"] == ["pcu"]

    def test_unknown_topology_is_data_not_a_raise(self, mofgen):
        throughput = _load("throughput")
        payload = throughput.run(mofgen, ["not_a_net"], repeats=1)
        assert payload["topologies"]["not_a_net"]["error"] == "unknown topology"
        json.dumps(payload, default=str)


class TestMappingGap:
    """The driver that attributes roundtrip's ``no_mapping`` bucket.

    A structure built from a net's own slots must map back onto it, so
    the round trip here is trivially satisfiable -- which is the point:
    if this reports a wall, the predicate or the deconstruction changed
    under us, not the chemistry.
    """

    def test_selfbuilt_structure_maps(self, mofgen, mof5_cif):
        gap = _load("mapping_gap")
        payload = gap.run([mof5_cif], n_jobs=1)
        record = payload["structures"]["mof5.cif"]
        assert record["outcome"] in {"mapped", "geometry", "arm_count"}
        json.dumps(payload, default=str)

    def test_walls_are_attributed_not_pooled(self, mofgen):
        """A slot with no right-sized unit is arm_count, not geometry."""
        gap = _load("mapping_gap")
        topology = mofgen.topologies["pcu"]
        two_connected = next(
            key
            for key in topology.mappings
            if len(key.atoms.indices_from_symbol("X")) == 2
        )
        six_connected = next(
            key
            for key in topology.mappings
            if len(key.atoms.indices_from_symbol("X")) == 6
        )
        # only ditopic units available: the 6-connected slot cannot be
        # filled at any threshold, and that is a connectivity fact
        verdict = gap._slot_verdict(six_connected, [two_connected], 0.35)
        assert verdict["wall"] == "arm_count"
        assert verdict["best_rmsd"] is None
        # a slot against itself is a fit, at any threshold
        assert gap._slot_verdict(two_connected, [two_connected], 0.35)["wall"] is None


class TestSelfTemplate:
    """The driver that rebuilds a crystal from its own blueprint.

    A structure this library built must survive its own self-templated
    round trip: the blueprint is its real embedding and the fragments
    are its own units, so anything short of closed_self means the
    stage-3 machinery, not the chemistry, broke.
    """

    def test_selfbuilt_structure_closes(self, mofgen, mof5_cif):
        st = _load("selftemplate")
        payload = st.run([mof5_cif], topofile=FIXTURE_PATH, n_jobs=1)
        record = payload["structures"]["mof5.cif"]
        assert record["outcome"] == "closed_self"
        assert 0.9 < record["volume_ratio"] < 1.1
        json.dumps(payload, default=str)


class TestUnidentifiedProbe:
    """The driver that attributes roundtrip's ``unidentified`` bucket.

    A structure built from a library net must identify, so the probe on
    it must land in the ``identified`` outcome with a well-formed
    quotient: nonzero vertices, a degree profile, and at least one
    prefilter candidate (itself).
    """

    def test_selfbuilt_structure_identifies(self, mofgen, mof5_cif):
        probe = _load("unidentified_probe")
        payload = probe.run([mof5_cif], topofile=FIXTURE_PATH, n_jobs=1)
        record = payload["structures"]["mof5.cif"]
        assert record["outcome"] == "identified"
        assert record["net"] == ["pcu"]
        assert record["contracted_vertices"] > 0
        assert record["n_prefilter_candidates"] >= 1
        assert not record["sig_empty"]
        json.dumps(payload, default=str)
