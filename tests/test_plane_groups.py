"""Tests for the plane-group operator tables and the 2D CGD pipeline."""

import numpy as np
import pytest
from pymatgen.symmetry.groups import SpaceGroup

from autografs import plane_groups
from autografs.cgd import build_group_lookup, read_cgd_data, topology_from_string

# group order = coset count x centering count
EXPECTED_ORDERS = {
    "p1": 1,
    "p2": 2,
    "pm": 2,
    "pg": 2,
    "cm": 4,
    "p2mm": 4,
    "p2mg": 4,
    "p2gg": 4,
    "c2mm": 8,
    "p4": 4,
    "p4mm": 8,
    "p4gm": 8,
    "p3": 3,
    "p3m1": 6,
    "p31m": 6,
    "p6": 6,
    "p6mm": 12,
}

# plane groups whose rotation axis maps directly onto the c-axis of an
# extruded 3D space group with no setting ambiguity; used to validate
# the hand-written tables against pymatgen's ITA data
EXTRUDED = {
    "p4": "P4",
    "p4mm": "P4mm",
    "p4gm": "P4bm",  # the in-plane g glide extrudes to a b glide (No. 100)
    "p3": "P3",
    "p3m1": "P3m1",
    "p31m": "P31m",
    "p6": "P6",
    "p6mm": "P6mm",
}

HCB_CGD = """CRYSTAL
  NAME hcb
  GROUP p6mm
  CELL 1.73205 1.73205 120.0000
  NODE 1 3  0.33333 0.66667
  EDGE  0.33333 0.66667   0.66667 0.33333
# EDGE_CENTER  0.50000 0.50000
END
"""

SQL_CGD = """CRYSTAL
  NAME sql
  GROUP p4mm
  CELL 1.00000 1.00000 90.0000
  NODE 1 4  0.00000 0.00000
  EDGE  0.00000 0.00000   0.00000 1.00000
# EDGE_CENTER  0.00000 0.50000
END
"""


def _augmented_ops(group: plane_groups.PlaneGroup) -> list[np.ndarray]:
    """All operators (cosets x centerings) as 2x3 augmented matrices."""
    ops = []
    for matrix_rows, translation in group.operators:
        for centering in group.centerings:
            matrix = np.zeros((2, 3))
            matrix[:, :2] = np.asarray(matrix_rows, dtype=float)
            matrix[:, 2] = np.mod(np.asarray(translation) + np.asarray(centering), 1.0)
            ops.append(matrix)
    return ops


def _op_key(matrix: np.ndarray) -> tuple:
    linear = tuple(np.round(matrix[:, :2], 6).ravel())
    translation = tuple(np.round(np.mod(matrix[:, 2] + 1e-9, 1.0), 6))
    return linear + translation


class TestOperatorTables:
    def test_all_17_groups_present(self):
        assert len(plane_groups.PLANE_GROUPS) == 17
        numbers = {g.number for g in plane_groups.PLANE_GROUPS.values()}
        assert numbers == set(range(1, 18))

    @pytest.mark.parametrize("symbol", sorted(plane_groups.PLANE_GROUPS))
    def test_group_axioms(self, symbol):
        """Identity, expected order, closure under composition mod 1."""
        group = plane_groups.PLANE_GROUPS[symbol]
        ops = _augmented_ops(group)
        assert len(ops) == EXPECTED_ORDERS[symbol]
        keys = {_op_key(op) for op in ops}
        # no duplicate operators
        assert len(keys) == len(ops)
        identity = np.hstack([np.eye(2), np.zeros((2, 1))])
        assert _op_key(identity) in keys
        # closure: (M1|t1) o (M2|t2) = (M1 M2 | M1 t2 + t1) mod 1
        for op1 in ops:
            for op2 in ops:
                product = np.zeros((2, 3))
                product[:, :2] = op1[:, :2] @ op2[:, :2]
                product[:, 2] = op1[:, :2] @ op2[:, 2] + op1[:, 2]
                assert _op_key(product) in keys, f"{symbol} not closed"

    @pytest.mark.parametrize("symbol", sorted(EXTRUDED))
    def test_hex_square_tables_match_pymatgen_extrusion(self, symbol):
        """Orbit of a generic point equals the extruded space group's."""
        point = np.array([0.1234, 0.4567])
        ours = plane_groups.expand_orbit(symbol, point)
        theirs = set()
        for op in SpaceGroup(EXTRUDED[symbol]).symmetry_ops:
            image = op.operate([point[0], point[1], 0.0])
            assert abs(image[2] % 1.0) < 1e-9  # extrusion never moves z
            theirs.add(tuple(np.round(np.mod(image[:2], 1.0), 6)))
        assert {tuple(np.round(xy, 6)) for xy in ours} == theirs

    def test_layer_system_families(self):
        assert plane_groups.layer_system(1) == "oblique"
        assert plane_groups.layer_system(2) == "oblique"
        assert plane_groups.layer_system(3) == "rectangular"
        assert plane_groups.layer_system(9) == "rectangular"
        assert plane_groups.layer_system(10) == "square"
        assert plane_groups.layer_system(12) == "square"
        assert plane_groups.layer_system(13) == "hexagonal"
        assert plane_groups.layer_system(17) == "hexagonal"
        with pytest.raises(ValueError):
            plane_groups.layer_system(18)

    def test_special_position_multiplicity(self):
        """Points on symmetry elements collapse to reduced orbits."""
        # origin is the 6-fold axis of p6mm: orbit of size 1
        assert len(plane_groups.expand_orbit("p6mm", [0.0, 0.0])) == 1
        # (1/3, 2/3) is the 3-fold site: orbit of size 2
        assert len(plane_groups.expand_orbit("p6mm", [1 / 3, 2 / 3])) == 2
        # generic point of the centered c2mm: 4 cosets x 2 centerings
        assert len(plane_groups.expand_orbit("c2mm", [0.1234, 0.4567])) == 8


class TestMonoclinicSettingRescue:
    """Nonstandard-setting GROUP symbols generate correct nets.

    RCSR writes some monoclinic nets in alternate cell choices
    (I12/a1, P121/n1, ...) and No. 64 in the pre-2002 glide notation
    (Cmca). Expanding a generic orbit and re-detecting the group with
    spglib guards against the silent wrong-setting trap that once
    produced garbage dia nets.
    """

    @pytest.mark.parametrize(
        "symbol, number, angles",
        [
            ("Cmca", 64, (90.0, 90.0, 90.0)),
            ("I12/a1", 15, (90.0, 100.0, 90.0)),
            ("A12/n1", 15, (90.0, 100.0, 90.0)),
            ("P121/n1", 14, (90.0, 100.0, 90.0)),
            ("I12/m1", 12, (90.0, 100.0, 90.0)),
        ],
    )
    def test_setting_survives_spglib_roundtrip(self, symbol, number, angles):
        from pymatgen.core.lattice import Lattice
        from pymatgen.core.structure import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        from autografs.cgd import normalize_group_symbol

        groupname = normalize_group_symbol(symbol, build_group_lookup())
        lattice = Lattice.from_parameters(6.0, 7.0, 8.0, *angles)
        struct = Structure.from_spacegroup(
            sg=groupname,
            lattice=lattice,
            species=["C"],
            coords=[[0.1234, 0.2345, 0.3456]],
        )
        detected = SpacegroupAnalyzer(struct, symprec=1e-3)
        assert detected.get_space_group_number() == number


class TestPlaneGroupCgdParsing:
    def test_hcb_is_honeycomb(self):
        """hcb must render a honeycomb, not a setting-mangled net."""
        name, struct, number, is_2d = topology_from_string(
            HCB_CGD.split("END")[0].strip().strip("CRYSTAL"),
            build_group_lookup(),
        )
        assert name == "hcb"
        assert number == 17
        assert is_2d
        # 2 nodes + 3 edge centers + 6 edge quarter-point dummies
        symbols = [site.specie.symbol for site in struct]
        assert symbols.count("Li") == 2  # Z = 3 encodes 3-coordination
        assert symbols.count("He") == 3
        assert symbols.count("X") == 6
        assert len(struct) == 11
        # honeycomb coordination: every node has exactly 3 edge centers
        # at half the unit edge length, in the z = 0 plane
        assert np.allclose(struct.cart_coords[:, 2], 0.0)
        for site in struct:
            if site.specie.symbol != "Li":
                continue
            neighbors = struct.get_neighbors(site, r=0.55)
            centers = [n for n in neighbors if n.specie.symbol == "He"]
            assert len(centers) == 3
            assert np.allclose([n.nn_distance for n in centers], 0.5, atol=1e-4)

    def test_sql_is_square_grid(self):
        name, struct, number, is_2d = topology_from_string(
            SQL_CGD.split("END")[0].strip().strip("CRYSTAL"),
            build_group_lookup(),
        )
        assert name == "sql"
        assert number == 11
        assert is_2d
        symbols = [site.specie.symbol for site in struct]
        assert symbols.count("Be") == 1  # Z = 4 encodes 4-coordination
        assert symbols.count("He") == 2
        assert symbols.count("X") == 4
        node = struct[symbols.index("Be")]
        centers = [
            n for n in struct.get_neighbors(node, r=0.55) if n.specie.symbol == "He"
        ]
        assert len(centers) == 4
        assert np.allclose([n.nn_distance for n in centers], 0.5, atol=1e-4)

    def test_read_cgd_data_produces_2d_topologies(self):
        topologies = read_cgd_data(HCB_CGD + SQL_CGD)
        assert set(topologies) == {"hcb", "sql"}
        hcb = topologies["hcb"]
        assert hcb.is_2d
        assert hcb.spacegroup_number == 17
        assert len(hcb) == 5  # 2 nodes + 3 edge centers
        sql = topologies["sql"]
        assert sql.is_2d
        assert sql.spacegroup_number == 11
        assert len(sql) == 3  # 1 node + 2 edge centers
