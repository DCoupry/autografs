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
