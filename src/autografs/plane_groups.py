"""
Explicit symmetry operators for the 17 crystallographic plane groups.

2D layer nets (the COF workhorses: hcb, sql, kgm, hxl, ...) are stored
in RCSR with plane-group GROUP symbols (p6mm, c2mm, ...) that no 3D
``SpaceGroup`` lookup can translate. Extruding them into 3D space-group
settings is a trap for the oblique/rectangular families: the standard
3D settings put the unique axis along b while the layer's symmetry
elements are along c, exactly the origin/setting confusion class that
once produced wrong dia nets. This module sidesteps settings entirely:
each plane group is a hand-written table of (2x2 matrix, translation)
operators acting on fractional (x, y), applied directly to expand the
orbits. z is untouched.

Operator tables follow the standard ITA settings (the ones Systre and
RCSR use). The hexagonal/square tables are cross-checked against
pymatgen's extruded space groups in the test suite; all tables are
checked for group closure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure


@dataclass(frozen=True)
class PlaneGroup:
    """One plane group: ITA number and its symmetry operators.

    Attributes
    ----------
    number : int
        ITA plane-group number (1-17).
    operators : tuple
        Coset representatives as ((2x2 matrix rows), (tx, ty)) acting
        on fractional (x, y) coordinates.
    centerings : tuple
        Centering translations; (0, 0) plus (1/2, 1/2) for the
        c-centered groups cm and c2mm.
    """

    number: int
    operators: tuple[
        tuple[tuple[tuple[int, int], tuple[int, int]], tuple[float, float]], ...
    ]
    centerings: tuple[tuple[float, float], ...] = ((0.0, 0.0),)


def _op(matrix_rows, translation=(0.0, 0.0)):
    return (matrix_rows, translation)


# point operators, written as matrix rows: x' = row0 . (x, y) etc.
_E = ((1, 0), (0, 1))  # x, y
_I2 = ((-1, 0), (0, -1))  # -x, -y
_MX = ((-1, 0), (0, 1))  # -x, y
_MY = ((1, 0), (0, -1))  # x, -y
_R4 = ((0, -1), (1, 0))  # -y, x
_R4I = ((0, 1), (-1, 0))  # y, -x
_MD = ((0, 1), (1, 0))  # y, x
_MDI = ((0, -1), (-1, 0))  # -y, -x
# hexagonal axes
_R3 = ((0, -1), (1, -1))  # -y, x-y
_R3I = ((-1, 1), (-1, 0))  # -x+y, -x
_R6 = ((1, -1), (1, 0))  # x-y, x
_R6I = ((0, 1), (-1, 1))  # y, -x+y
_M21 = ((-1, 1), (0, 1))  # -x+y, y
_M22 = ((1, 0), (1, -1))  # x, x-y
_M31 = ((1, -1), (0, -1))  # x-y, -y
_M32 = ((-1, 0), (-1, 1))  # -x, -x+y

_HALF = (0.5, 0.5)
_P3 = (_op(_E), _op(_R3), _op(_R3I))
_P6 = _P3 + (_op(_I2), _op(_R6), _op(_R6I))
_M3M1 = (_op(_MDI), _op(_M21), _op(_M22))  # mirrors of p3m1
_M31M = (_op(_MD), _op(_M31), _op(_M32))  # mirrors of p31m

PLANE_GROUPS: dict[str, PlaneGroup] = {
    "p1": PlaneGroup(1, (_op(_E),)),
    "p2": PlaneGroup(2, (_op(_E), _op(_I2))),
    "pm": PlaneGroup(3, (_op(_E), _op(_MX))),
    "pg": PlaneGroup(4, (_op(_E), _op(_MX, (0.0, 0.5)))),
    "cm": PlaneGroup(5, (_op(_E), _op(_MX)), ((0.0, 0.0), _HALF)),
    "p2mm": PlaneGroup(6, (_op(_E), _op(_I2), _op(_MX), _op(_MY))),
    "p2mg": PlaneGroup(
        7, (_op(_E), _op(_I2), _op(_MX, (0.5, 0.0)), _op(_MY, (0.5, 0.0)))
    ),
    "p2gg": PlaneGroup(8, (_op(_E), _op(_I2), _op(_MX, _HALF), _op(_MY, _HALF))),
    "c2mm": PlaneGroup(9, (_op(_E), _op(_I2), _op(_MX), _op(_MY)), ((0.0, 0.0), _HALF)),
    "p4": PlaneGroup(10, (_op(_E), _op(_I2), _op(_R4), _op(_R4I))),
    "p4mm": PlaneGroup(
        11,
        (
            _op(_E),
            _op(_I2),
            _op(_R4),
            _op(_R4I),
            _op(_MX),
            _op(_MY),
            _op(_MD),
            _op(_MDI),
        ),
    ),
    "p4gm": PlaneGroup(
        12,
        (
            _op(_E),
            _op(_I2),
            _op(_R4),
            _op(_R4I),
            _op(_MX, _HALF),
            _op(_MY, _HALF),
            _op(_MDI, _HALF),
            _op(_MD, _HALF),
        ),
    ),
    "p3": PlaneGroup(13, _P3),
    "p3m1": PlaneGroup(14, _P3 + _M3M1),
    "p31m": PlaneGroup(15, _P3 + _M31M),
    "p6": PlaneGroup(16, _P6),
    "p6mm": PlaneGroup(17, _P6 + _M3M1 + _M31M),
}


def layer_system(plane_group_number: int) -> str:
    """Lattice family of a plane group: the free cell parameters.

    oblique (1-2): a, b, gamma free; rectangular (3-9): a, b free;
    square (10-12) and hexagonal (13-17): a free only.
    """
    if not 1 <= plane_group_number <= 17:
        raise ValueError(f"Not a plane group number: {plane_group_number}")
    if plane_group_number <= 2:
        return "oblique"
    if plane_group_number <= 9:
        return "rectangular"
    if plane_group_number <= 12:
        return "square"
    return "hexagonal"


def expand_orbit(symbol: str, frac_xy: np.ndarray, decimals: int = 8) -> np.ndarray:
    """All images of a fractional (x, y) point under a plane group.

    Parameters
    ----------
    symbol : str
        Plane group symbol, a key of PLANE_GROUPS.
    frac_xy : np.ndarray
        (2,) fractional coordinates in the layer plane.
    decimals : int, optional
        Rounding used to deduplicate images (wrapped into [0, 1)).

    Returns
    -------
    np.ndarray
        (m, 2) unique fractional images, wrapped into [0, 1).
    """
    group = PLANE_GROUPS[symbol]
    frac_xy = np.asarray(frac_xy, dtype=float)
    images = []
    for matrix_rows, translation in group.operators:
        base = np.asarray(matrix_rows, dtype=float) @ frac_xy + translation
        for centering in group.centerings:
            images.append(base + centering)
    wrapped = np.mod(np.round(np.stack(images), decimals), 1.0)
    # rounding after the wrap too: mod can return values within one ulp
    # below 1.0 that round-trip to distinct rows otherwise
    wrapped = np.mod(np.round(wrapped, decimals), 1.0)
    return np.unique(wrapped, axis=0)


def structure_from_plane_group(
    symbol: str,
    lattice: Lattice,
    species: list,
    frac_coords: np.ndarray,
) -> Structure:
    """Expand a 2D net's asymmetric unit with plane-group operators.

    The 2D analogue of ``Structure.from_spacegroup``: operators act on
    fractional (x, y); each site's z coordinate is carried through
    unchanged (the padded-slab convention puts nets at z = 0).

    Parameters
    ----------
    symbol : str
        Plane group symbol, a key of PLANE_GROUPS.
    lattice : Lattice
        The padded 3D lattice of the layer.
    species : list
        One species per input site.
    frac_coords : np.ndarray
        (n, 3) fractional coordinates of the asymmetric sites.

    Returns
    -------
    Structure
        The expanded periodic structure. Coincident images within one
        orbit are already deduplicated; merging near-duplicates across
        different sites is left to the caller (as with
        Structure.from_spacegroup).
    """
    frac_coords = np.asarray(frac_coords, dtype=float)
    all_species = []
    all_coords = []
    for specie, site in zip(species, frac_coords, strict=True):
        for xy in expand_orbit(symbol, site[:2]):
            all_species.append(specie)
            all_coords.append([xy[0], xy[1], site[2]])
    return Structure(lattice, all_species, all_coords)
