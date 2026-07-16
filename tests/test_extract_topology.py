"""Tests for tetrahedral topology extraction (autografs.extract_topology)."""

import os

import numpy as np
import pytest
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from autografs.exceptions import TopologyExtractionError
from autografs.extract_topology import topology_from_tetrahedral
from autografs.net import identify_net, topology_quotient_edges

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "topologies_fixture.json"
)


@pytest.fixture(scope="module")
def mofgen():
    from autografs import Autografs

    return Autografs(topofile=FIXTURE_PATH)


def _tetrahedral_crystal(topology, t_o: float = 1.61) -> Structure:
    """Idealized 'SiO2' crystal realizing a 4-c net: Si on the node
    slot centers, O on the edge-center slot centers, cell scaled to a
    realistic T-O bond length."""
    inv = np.linalg.inv(topology.cell.matrix)
    t_frac, o_frac = [], []
    for slot in topology.slots:
        # library slots hold only the X arms; their centroid is the
        # slot center (T position for nodes, O position for edges)
        n_arms = len(slot.atoms.indices_from_symbol("X"))
        frac = slot.atoms.cart_coords.mean(axis=0) @ inv
        (t_frac if n_arms == 4 else o_frac).append(frac)
    unscaled = Structure(
        topology.cell,
        ["Si"] * len(t_frac) + ["O"] * len(o_frac),
        t_frac + o_frac,
    )
    current = min(
        neighbor.nn_distance
        for neighbor in unscaled.get_neighbors(
            unscaled[0], r=2.0 * max(unscaled.lattice.abc)
        )
    )
    scale = t_o / current
    return Structure(
        Lattice(topology.cell.matrix * scale),
        unscaled.species,
        unscaled.frac_coords,
    )


@pytest.fixture(scope="module")
def dia_crystal(mofgen):
    return _tetrahedral_crystal(mofgen.topologies["dia"])


class TestExtraction:
    def test_dia_roundtrips_on_the_exact_tier(self, mofgen, dia_crystal):
        """The extracted net must be indistinguishable from the
        CGD-imported entry: same slot structure, same uncontracted
        signature - the exact identification tier, not the fallback."""
        extracted = topology_from_tetrahedral(dia_crystal, "dia-extracted")
        assert len(extracted) == len(mofgen.topologies["dia"])
        matches = identify_net(topology_quotient_edges(extracted), mofgen.topologies)
        assert matches == ["dia"]
        assert matches.tier == "exact"

    def test_crystallography_recovered(self, dia_crystal):
        extracted = topology_from_tetrahedral(dia_crystal, "dia-extracted")
        assert extracted.spacegroup_number == 227  # Fd-3m, diamond
        assert all(slot.equivalence_class is not None for slot in extracted.slots)

    def test_extracted_topology_is_buildable(self, mofgen, dia_crystal):
        """The end goal: the extracted blueprint builds and the result
        verifies against it exactly."""
        extracted = topology_from_tetrahedral(dia_crystal, "dia-extracted")
        by_arms = {
            len(key.atoms.indices_from_symbol("X")): key for key in extracted.mappings
        }
        tetrahedral = sorted(
            name
            for name, fragment in mofgen.sbu.items()
            if fragment.has_compatible_symmetry(by_arms[4])
        )
        assert tetrahedral, "no tetrahedral SBU fits the extracted node slot"
        framework = mofgen.build(
            extracted,
            mappings={by_arms[4]: tetrahedral[0], by_arms[2]: "Benzene_linear"},
            max_rmsd=0.5,
            verify_net=True,
        )
        assert len(framework) > 0

    def test_interrupted_framework_rejected(self, dia_crystal):
        """Removing one bridging oxygen leaves two 3-bridged T atoms -
        the IZA dash-code situation, which has no 4-c net."""
        broken = dia_crystal.copy()
        oxygens = [i for i, site in enumerate(broken) if site.specie.symbol == "O"]
        broken.remove_sites([oxygens[0]])
        with pytest.raises(TopologyExtractionError, match="four oxygen bridges"):
            topology_from_tetrahedral(broken, "broken")

    def test_nonbridging_oxygen_rejected(self):
        """A terminal oxygen (one T neighbor) is not a bridge: an
        isolated SiO4 tetrahedron has four of them."""
        arm = 1.61 / np.sqrt(3.0)
        sio4 = Structure(
            Lattice.cubic(12.0),
            ["Si", "O", "O", "O", "O"],
            [
                [0.0, 0.0, 0.0],
                [arm, arm, arm],
                [-arm, -arm, arm],
                [-arm, arm, -arm],
                [arm, -arm, -arm],
            ],
            coords_are_cartesian=True,
        )
        with pytest.raises(TopologyExtractionError, match="instead of two"):
            topology_from_tetrahedral(sio4, "sio4")

    def test_missing_species_rejected(self):
        all_si = Structure(Lattice.cubic(5.0), ["Si"], [[0.0, 0.0, 0.0]])
        with pytest.raises(TopologyExtractionError, match="T atoms and bridging"):
            topology_from_tetrahedral(all_si, "no-oxygen")
