"""Round-trip tests for Framework JSON serialization (framework_io)."""

import os

import numpy as np
import pytest

from autografs.framework import Framework

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "topologies_fixture.json"
)


@pytest.fixture(scope="module")
def mofgen():
    from autografs import Autografs

    return Autografs(topofile=FIXTURE_PATH)


@pytest.fixture(scope="module")
def mof5(mofgen):
    topology = mofgen.topologies["pcu"]
    mappings = {}
    for key in topology.mappings:
        conn = len(key.atoms.indices_from_symbol("X"))
        mappings[key] = {6: "Zn_mof5_octahedral", 2: "Benzene_linear"}[conn]
    return mofgen.build(topology, mappings=mappings, refine_cell=True, max_rmsd=0.5)


def assert_same_framework(a: Framework, b: Framework) -> None:
    assert a.name == b.name
    assert a.energy == b.energy
    np.testing.assert_allclose(a.cell, b.cell)
    assert a.symbols == b.symbols
    np.testing.assert_allclose(a.cart_coords, b.cart_coords)
    assert a.mmtypes == b.mmtypes
    assert sorted(a.bonds) == sorted(b.bonds)
    tags_a = [a.graph.nodes[n]["tag"] for n in sorted(a.graph)]
    tags_b = [b.graph.nodes[n]["tag"] for n in sorted(b.graph)]
    assert tags_a == tags_b
    assert a.slots == b.slots


class TestRoundTrip:
    def test_json_round_trip(self, mof5, tmp_path):
        path = mof5.save(tmp_path / "mof5.json")
        loaded = Framework.load(path)
        assert_same_framework(mof5, loaded)

    def test_gzip_round_trip(self, mof5, tmp_path):
        path = mof5.save(tmp_path / "mof5.json.gz")
        loaded = Framework.load(path)
        assert_same_framework(mof5, loaded)

    def test_energy_survives(self, mof5, tmp_path):
        mof5_copy = Framework.load(mof5.save(tmp_path / "a.json"))
        mof5_copy.energy = -123.5
        reloaded = Framework.load(mof5_copy.save(tmp_path / "b.json"))
        assert reloaded.energy == -123.5

    def test_loaded_framework_is_editable(self, mof5, tmp_path):
        """The whole point: provenance survives, so post-build editing
        works on a framework loaded in a later session."""
        loaded = Framework.load(mof5.save(tmp_path / "mof5.json.gz"))
        supercell = loaded.supercell(2)
        assert len(supercell) == 8 * len(loaded)
        linker = next(s for s, name in loaded.slots.items() if name == "Benzene_linear")
        rotated = loaded.rotate(linker, np.pi / 6)
        assert set(rotated.graph.edges()) == set(loaded.graph.edges())
        defective = loaded.supercell(2).defects(
            fraction=0.25, sbu="Benzene_linear", seed=1
        )
        assert len(defective.slots) < 8 * len(loaded.slots)

    def test_stacked_and_edited_frameworks_round_trip(self, mofgen, tmp_path):
        """Layer builds and post-stack graphs serialize too."""
        topology = mofgen.topologies["hcb"]
        mappings = {}
        for key in topology.mappings:
            conn = len(key.atoms.indices_from_symbol("X"))
            mappings[key] = {3: "Boroxine_triangle", 2: "Benzene_linear"}[conn]
        layer = mofgen.build(topology, mappings=mappings, max_rmsd=0.5)
        stacked = layer.stack(mode="AB")
        loaded = Framework.load(stacked.save(tmp_path / "cof.json.gz"))
        assert_same_framework(stacked, loaded)


class TestFormatGuards:
    def test_wrong_version_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text('{"format_version": 999, "framework": {}}')
        with pytest.raises(ValueError, match="format version"):
            Framework.load(bad)

    def test_non_contiguous_ids_rejected(self, mof5):
        import networkx

        from autografs.framework_io import framework_to_dict

        graph = networkx.Graph(cell=np.eye(3) * 10)
        graph.add_node(0, symbol="C", coord=np.zeros(3), tag=0, ufftype="C_R")
        graph.add_node(2, symbol="C", coord=np.ones(3), tag=0, ufftype="C_R")
        broken = Framework(graph, name="broken")
        with pytest.raises(ValueError, match="contiguous"):
            framework_to_dict(broken)
