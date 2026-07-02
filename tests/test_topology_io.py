"""
Tests for the JSON topology serialization in autografs.topology_io.
"""

import numpy as np
import pytest
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Molecule

from autografs import topology_io
from autografs.fragment import Fragment
from autografs.topology import Topology


@pytest.fixture
def binodal_topology():
    """A two-orbit topology with tagged slots, like cgd2pkl produces."""
    lin = Molecule(
        ["C", "X", "X"], [[0, 0, 0], [0.25, 0, 0], [-0.25, 0, 0]]
    )
    lin.add_site_property("tags", [0, 1, 2])
    tri = Molecule(
        ["N", "X", "X", "X"],
        [[0, 0, 0], [0.25, 0, 0], [-0.125, 0.2165, 0], [-0.125, -0.2165, 0]],
    )
    tri.add_site_property("tags", [0, 3, 4, 5])
    slots = [
        Fragment(atoms=lin.copy(), name="slot"),
        Fragment(atoms=lin.copy(), name="slot"),
        Fragment(atoms=tri.copy(), name="slot"),
    ]
    return Topology(
        name="test_net",
        slots=slots,
        cell=Lattice.cubic(1.0),
        equivalence_classes=[0, 0, 1],
        spacegroup_number=221,
    )


class TestRoundtrip:
    """save -> load must reproduce the topology."""

    @pytest.mark.parametrize("filename", ["topos.json", "topos.json.gz"])
    def test_roundtrip(self, binodal_topology, tmp_path, filename):
        path = tmp_path / filename
        topology_io.save_topologies({"test_net": binodal_topology}, path)
        loaded = topology_io.load_topologies(path)

        assert list(loaded) == ["test_net"]
        topo = loaded["test_net"]
        np.testing.assert_array_almost_equal(
            topo.cell.matrix, binodal_topology.cell.matrix
        )
        assert topo.spacegroup_number == 221
        assert len(topo.slots) == 3
        for original, restored in zip(binodal_topology.slots, topo.slots):
            assert restored.pointgroup == original.pointgroup
            assert restored.equivalence_class == original.equivalence_class
            assert (
                restored.atoms.site_properties["tags"]
                == original.atoms.site_properties["tags"]
            )
            np.testing.assert_array_almost_equal(
                restored.atoms.cart_coords, original.atoms.cart_coords
            )
            assert [s.specie.symbol for s in restored.atoms] == [
                s.specie.symbol for s in original.atoms
            ]

    def test_roundtrip_preserves_slot_grouping(self, binodal_topology, tmp_path):
        path = tmp_path / "topos.json"
        topology_io.save_topologies({"test_net": binodal_topology}, path)
        topo = topology_io.load_topologies(path)["test_net"]

        original_groups = sorted(
            sorted(v) for v in binodal_topology.mappings.values()
        )
        restored_groups = sorted(sorted(v) for v in topo.mappings.values())
        assert restored_groups == original_groups == [[0, 1], [2]]

    def test_no_pointgroup_analysis_on_load(self, binodal_topology, tmp_path):
        """Loading must not trigger the expensive symmetry analysis."""
        path = tmp_path / "topos.json"
        topology_io.save_topologies({"test_net": binodal_topology}, path)
        topo = topology_io.load_topologies(path)["test_net"]
        for slot in topo.slots:
            assert slot._symmetry is None
            assert slot.pointgroup  # symbol available regardless


class TestFormatVersion:
    def test_unsupported_version_raises(self, tmp_path):
        path = tmp_path / "topos.json"
        path.write_text('{"format_version": 999, "topologies": {}}')
        with pytest.raises(ValueError, match="format version"):
            topology_io.load_topologies(path)

    def test_missing_version_raises(self, tmp_path):
        path = tmp_path / "topos.json"
        path.write_text('{"topologies": {}}')
        with pytest.raises(ValueError, match="format version"):
            topology_io.load_topologies(path)


class TestBuilderIntegration:
    def test_autografs_loads_json_library(self, binodal_topology, tmp_path):
        """Autografs accepts a JSON topology library via topofile."""
        from autografs import Autografs

        path = tmp_path / "topos.json.gz"
        topology_io.save_topologies({"test_net": binodal_topology}, path)
        mofgen = Autografs(topofile=str(path))
        assert "test_net" in mofgen.topologies
        assert len(mofgen.topologies["test_net"].slots) == 3
