"""Tests for the LAMMPS data file we write ourselves (lammps_data).

The writer exists because lammps-interface re-derives topology from
geometry and loses structures doing it. Correctness here is checked two
ways: the topology against hand-countable graphs, and the force-field
coefficients against the published UFF values that lammps-interface
independently produces (validated on MOF-5: every bond coefficient
equal to six decimals, total energy within 0.5%).

Only the coefficient tests need the parameter tables; none of them need
a LAMMPS runtime.
"""

import math

import networkx
import numpy as np
import pytest

FIXTURE_PATH = "tests/data/topologies_fixture.json"


def _has_tables() -> bool:
    try:
        import lammps_interface.uff4mof  # noqa: F401

        return True
    except Exception:
        return False


needs_tables = pytest.mark.skipif(
    not _has_tables(), reason="UFF parameter tables (lammps-interface) not installed"
)


def _benzene_like():
    """A 6-ring of resonant carbons, each carrying one hydrogen."""
    graph = networkx.Graph(cell=np.diag([20.0, 20.0, 20.0]))
    radius = 1.39
    for i in range(6):
        angle = 2 * math.pi * i / 6
        graph.add_node(
            i,
            symbol="C",
            coord=np.array([radius * math.cos(angle), radius * math.sin(angle), 0.0]),
            tag=i,
            ufftype="C_R",
        )
        graph.add_node(
            i + 6,
            symbol="H",
            coord=np.array([2.47 * math.cos(angle), 2.47 * math.sin(angle), 0.0]),
            tag=i + 6,
            ufftype="H_",
        )
        graph.add_edge(i, i + 6, bond_order=1.0)
    for i in range(6):
        graph.add_edge(i, (i + 1) % 6, bond_order=1.5)
    return graph


class TestTopology:
    def test_counts_are_hand_checkable(self):
        from autografs.lammps_data import enumerate_topology

        topology = enumerate_topology(_benzene_like())
        # 6 ring bonds + 6 C-H
        assert len(topology["bonds"]) == 12
        # each ring carbon centres 3 angles (C-C-C, C-C-H twice)
        assert len(topology["angles"]) == 18
        # every ring carbon is 3-coordinate and inversion-active
        assert len(topology["impropers"]) == 6

    def test_dihedrals_span_every_bond_with_two_ends(self):
        from autografs.lammps_data import enumerate_topology

        topology = enumerate_topology(_benzene_like())
        # ring bonds carry (2 neighbours each side) = 4 torsions; C-H
        # bonds have a terminal end and carry none through H
        assert len(topology["dihedrals"]) == 6 * 4
        for a, b, c, d in topology["dihedrals"]:
            assert len({a, b, c, d}) == 4

    def test_a_terminal_atom_gets_no_improper(self):
        from autografs.lammps_data import enumerate_topology

        graph = networkx.Graph(cell=np.diag([20.0] * 3))
        graph.add_node(0, symbol="C", coord=np.zeros(3), tag=0, ufftype="C_3")
        graph.add_node(1, symbol="H", coord=np.array([1.1, 0, 0]), tag=1, ufftype="H_")
        graph.add_edge(0, 1, bond_order=1.0)
        topology = enumerate_topology(graph)
        assert topology["impropers"] == []
        assert topology["dihedrals"] == []


class TestBondOrderConvention:
    """UFF wants a discrete order; our graph carries a perceived one."""

    def test_resonant_pair_snaps_to_one_and_a_half(self):
        from autografs.lammps_data import uff_bond_order

        # measured perception on MOF-5's ring: 1.736
        assert uff_bond_order("C_R", "C_R", 1.736) == 1.5

    def test_a_resonant_type_on_a_single_bond_stays_single(self):
        """MOF-5's ring-to-carboxylate C_R-C_R is a single bond.

        Resonance cannot be read from the types alone, which is why the
        perceived order decides: lammps-interface emits two distinct
        C_R-C_R bond types for exactly this reason.
        """
        from autografs.lammps_data import uff_bond_order

        assert uff_bond_order("C_R", "C_R", 1.0) == 1.0

    def test_resonance_can_involve_a_non_resonant_partner(self):
        """Carboxylate C_R-O_2 is resonant though O_2 carries no _R."""
        from autografs.lammps_data import uff_bond_order

        assert uff_bond_order("C_R", "O_2", 1.803) == 1.5

    def test_plain_bonds_round_to_integers(self):
        from autografs.lammps_data import uff_bond_order

        assert uff_bond_order("O_2", "Zn3+2", 1.0) == 1.0
        assert uff_bond_order("C_3", "C_3", 1.05) == 1.0


@needs_tables
class TestCoefficients:
    """Values cross-checked against lammps-interface's own data file."""

    def test_bond_lengths_match_the_published_ones(self):
        from autografs.lammps_data import natural_bond_length, uff_parameters

        table = uff_parameters("UFF4MOF")
        # the four MOF-5 bond types, as lammps-interface emits them
        expected = {
            ("C_R", "C_R", 1.5): 1.379256,
            ("C_R", "O_2", 1.5): 1.269010,
            ("C_R", "H_", 1.0): 1.081418,
            ("O_2", "Zn3+2", 1.0): 1.795426,
        }
        for (ti, tj, order), reference in expected.items():
            got = natural_bond_length(table[ti], table[tj], order)
            assert got == pytest.approx(reference, abs=1e-6), (ti, tj, order)

    def test_bond_force_constants_match(self):
        from autografs.lammps_data import bond_coefficients, uff_parameters

        table = uff_parameters("UFF4MOF")
        force, r0 = bond_coefficients(table["C_R"], table["C_R"], 1.5)
        # lammps-interface: 462.655054 1.379256
        assert force == pytest.approx(462.655054, abs=1e-5)
        assert r0 == pytest.approx(1.379256, abs=1e-6)

    def test_torsion_barrier_is_shared_among_its_torsions(self):
        """UFF states one barrier per bond, divided among the torsions.

        Omitting the division overcounted MOF-5's dihedral energy by
        exactly 4x - benzene's C_R-C_R carries four.
        """
        from autografs.lammps_data import dihedral_coefficients, uff_parameters

        table = uff_parameters("UFF4MOF")
        one = dihedral_coefficients(table["C_R"], table["C_R"], "C_R", "C_R", 1.5, 1)
        four = dihedral_coefficients(table["C_R"], table["C_R"], "C_R", "C_R", 1.5, 4)
        assert one is not None and four is not None
        assert four[0] == pytest.approx(one[0] / 4.0)
        assert four[1:] == one[1:]

    def test_a_terminal_centre_carries_no_torsion(self):
        """H_ encodes no coordination, so no UFF torsion rule applies.

        A metal DOES get one: `Zn3+2` reads as sp3 and pairs with the
        sp2 oxygen under UFF's sp2-sp3 rule. That is the documented
        behaviour, not an oversight.
        """
        from autografs.lammps_data import dihedral_coefficients, uff_parameters

        table = uff_parameters("UFF4MOF")
        assert (
            dihedral_coefficients(table["H_"], table["O_2"], "H_", "O_2", 1.0) is None
        )
        assert (
            dihedral_coefficients(table["Zn3+2"], table["O_2"], "Zn3+2", "O_2", 1.0)
            is not None
        )

    def test_linear_centres_use_the_periodic_angle_style(self):
        from autografs.lammps_data import angle_coefficients, uff_parameters

        table = uff_parameters("UFF4MOF")
        style, _values = angle_coefficients(
            table["C_R"], table["C_R"], table["C_R"], 1.5, 1.5
        )
        # 120 degrees -> periodic form
        assert style == "cosine/periodic"
        style, values = angle_coefficients(
            table["O_2"], table["Zn3+2"], table["O_2"], 1.0, 1.0
        )
        # 109.47 is not one of the special cases
        assert style == "fourier"
        assert len(values) == 4


@needs_tables
class TestWriter:
    def test_mof5_sections_are_complete_and_countable(self, tmp_path):
        from autografs import Autografs
        from autografs.lammps_data import write_lammps_data

        mofgen = Autografs(topofile=FIXTURE_PATH)
        topology = mofgen.topologies["pcu"]
        mappings = {}
        for slot in topology.mappings:
            arms = len(slot.atoms.indices_from_symbol("X"))
            mappings[slot] = {6: "Zn_mof5_octahedral", 2: "Benzene_linear"}[arms]
        framework = mofgen.build(topology, mappings, max_rmsd=0.5)

        data = write_lammps_data(framework, path=str(tmp_path / "data.mof5"))
        assert data.n_atoms == len(framework.structure)
        assert data.n_bonds == framework.graph.number_of_edges()
        # 3 BDC linkers, 8 three-coordinate carbons each
        assert data.n_impropers == 24
        for section in ("Masses", "Pair Coeffs", "Bond Coeffs", "Atoms", "Bonds"):
            assert section in data.text
        assert (tmp_path / "data.mof5").exists()

    def test_atoms_are_written_inside_the_box(self):
        """The graph stores unwrapped coordinates; LAMMPS needs them in."""
        from autografs import Autografs
        from autografs.lammps_data import write_lammps_data

        mofgen = Autografs(topofile=FIXTURE_PATH)
        topology = mofgen.topologies["pcu"]
        mappings = {}
        for slot in topology.mappings:
            arms = len(slot.atoms.indices_from_symbol("X"))
            mappings[slot] = {6: "Zn_mof5_octahedral", 2: "Benzene_linear"}[arms]
        framework = mofgen.build(topology, mappings, max_rmsd=0.5)
        data = write_lammps_data(framework)

        box = [
            float(line.split()[1])
            for line in data.text.splitlines()
            if line.endswith(("xlo xhi", "ylo yhi", "zlo zhi"))
        ]
        section = data.text.split("Atoms # full")[1].strip().splitlines()
        for line in section:
            if not line.strip() or line.startswith(("Bonds", "Angles")):
                break
            parts = line.split()
            for axis, value in enumerate(float(v) for v in parts[4:7]):
                assert -1e-6 <= value <= box[axis] + 1e-6

    def test_a_bond_longer_than_half_the_box_is_refused(self):
        """LAMMPS would resolve it through the wrong image."""
        from autografs.exceptions import RelaxationError
        from autografs.framework import Framework
        from autografs.lammps_data import write_lammps_data

        # a diagonal separation: along one axis the minimum image can
        # never exceed half the box, so the guard needs a bond that is
        # long in all three
        graph = networkx.Graph(cell=np.diag([4.0, 4.0, 4.0]))
        graph.add_node(0, symbol="C", coord=np.zeros(3), tag=0, ufftype="C_3")
        graph.add_node(
            1, symbol="C", coord=np.array([1.9, 1.9, 1.9]), tag=1, ufftype="C_3"
        )
        graph.add_edge(0, 1, bond_order=1.0)
        with pytest.raises(RelaxationError, match="half"):
            write_lammps_data(Framework(graph, name="stretched"))
