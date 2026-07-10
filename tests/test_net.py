"""Tests for the net verification gate (autografs.net)."""

import os

import networkx
import numpy as np
import pytest

from autografs.exceptions import NetMismatchError
from autografs.framework import Framework

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "topologies_fixture.json"
)


@pytest.fixture(scope="module")
def mofgen():
    from autografs import Autografs

    return Autografs(topofile=FIXTURE_PATH)


def mappings_by_connectivity(topology, choices):
    mappings = {}
    for key in topology.mappings:
        conn = len(key.atoms.indices_from_symbol("X"))
        mappings[key] = choices[conn]
    return mappings


@pytest.fixture(scope="module")
def mof5(mofgen):
    topology = mofgen.topologies["pcu"]
    mappings = mappings_by_connectivity(
        topology, {6: "Zn_mof5_octahedral", 2: "Benzene_linear"}
    )
    return mofgen.build(topology, mappings=mappings, refine_cell=True, max_rmsd=0.5)


class TestVerifyNet:
    def test_as_built_framework_verifies(self, mofgen, mof5):
        mof5.verify_net(mofgen.topologies["pcu"])  # must not raise

    def test_2d_layer_verifies(self, mofgen):
        topology = mofgen.topologies["hcb"]
        mappings = mappings_by_connectivity(
            topology, {3: "Boroxine_triangle", 2: "Benzene_linear"}
        )
        layer = mofgen.build(topology, mappings=mappings, max_rmsd=0.5)
        layer.verify_net(topology)  # must not raise

    def test_build_gate_passes(self, mofgen):
        topology = mofgen.topologies["pcu"]
        mappings = mappings_by_connectivity(
            topology, {6: "Zn_mof5_octahedral", 2: "Benzene_linear"}
        )
        framework = mofgen.build(topology, mappings=mappings, verify_net=True)
        assert len(framework) == 53

    def test_wrong_topology_rejected(self, mofgen, mof5):
        with pytest.raises(NetMismatchError, match="slots"):
            mof5.verify_net(mofgen.topologies["dia"])

    def test_missing_bond_detected(self, mofgen, mof5):
        """Deleting one inter-SBU bond must break verification."""
        broken = networkx.Graph(**mof5.graph.graph)
        broken.add_nodes_from(mof5.graph.nodes(data=True))
        broken.add_edges_from(mof5.graph.edges(data=True))
        inter = next(
            (u, v)
            for u, v in broken.edges()
            if broken.nodes[u]["slot"] != broken.nodes[v]["slot"]
        )
        broken.remove_edge(*inter)
        with pytest.raises(NetMismatchError, match="missing inter-SBU"):
            Framework(broken, name="broken").verify_net(mofgen.topologies["pcu"])

    def test_wrong_image_detected(self, mofgen, mof5):
        """Rewiring a boundary-crossing bond to the in-cell image (the
        classic wrong-image failure) must break verification even
        though node count, edge count, and degrees all stay right."""
        rewired = networkx.Graph(**mof5.graph.graph)
        rewired.add_nodes_from((n, dict(d)) for n, d in mof5.graph.nodes(data=True))
        rewired.add_edges_from(mof5.graph.edges(data=True))
        inv_cell = np.linalg.inv(mof5.cell)
        # find a bond crossing the boundary and pull its far endpoint
        # (and only it) into the home image
        for u, v in rewired.edges():
            if rewired.nodes[u]["slot"] == rewired.nodes[v]["slot"]:
                continue
            delta = np.asarray(rewired.nodes[u]["coord"]) - np.asarray(
                rewired.nodes[v]["coord"]
            )
            shift = np.round(delta @ inv_cell)
            if np.any(shift != 0):
                coord = np.asarray(rewired.nodes[v]["coord"], dtype=float)
                rewired.nodes[v]["coord"] = coord + shift @ mof5.cell
                break
        else:
            pytest.fail("no boundary-crossing inter-SBU bond found")
        with pytest.raises(NetMismatchError):
            Framework(rewired, name="rewired").verify_net(mofgen.topologies["pcu"])

    def test_no_provenance_rejected(self, mofgen):
        graph = networkx.Graph(cell=np.eye(3) * 10)
        graph.add_node(0, symbol="C", coord=np.zeros(3), tag=0, ufftype="C_R")
        bare = Framework(graph, name="bare")
        with pytest.raises(NetMismatchError, match="provenance"):
            bare.verify_net(mofgen.topologies["pcu"])
