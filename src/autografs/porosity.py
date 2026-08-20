"""
Geometric porosity descriptors for built frameworks.

For a generator of porous materials, the first questions about any
output are geometric: how dense is it, how much empty space does it
hold, how big is its largest cavity, and what is the widest probe that
can actually travel through it? This module answers them with a
periodic distance grid - ranking-quality numbers from pure geometry,
not a replacement for Zeo++-grade analysis.

All functions work on the vdW-surface convention: the distance field
d(p) is the distance from point p to the nearest atom's van der Waals
surface (negative inside an atom). A probe of radius r fits at p when
d(p) >= r.

- ``void_fraction``: fraction of the cell where a probe fits.
- ``largest_cavity_diameter`` (LCD): diameter of the largest sphere
  that fits anywhere in the structure.
- ``pore_limiting_diameter`` (PLD): diameter of the largest sphere
  that can percolate through the periodic structure - found by binary
  search over the grid with a wrap-detecting union-find.

Resolution is controlled by ``spacing`` (grid step in Angstrom):
smaller is more accurate and slower; the 0.4 A default resolves
typical MOF pores to a few tenths of an Angstrom.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.analysis.local_env import CovalentRadius
from pymatgen.core.periodic_table import Element
from scipy import ndimage
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from autografs.framework import Framework

__all__ = [
    "void_fraction",
    "largest_cavity_diameter",
    "pore_limiting_diameter",
    "porosity_report",
    "PorosityReport",
]

logger = logging.getLogger(__name__)

DEFAULT_SPACING = 0.4

# elements without a tabulated vdW radius fall back to their covalent
# radius plus this pad (a typical covalent-to-vdW gap)
_VDW_FALLBACK_PAD = 0.8


def _vdw_radius(symbol: str) -> float:
    radius = Element(symbol).van_der_waals_radius
    if radius:
        return float(radius)
    return float(CovalentRadius.radius.get(symbol, 1.2) + _VDW_FALLBACK_PAD)


def _distance_grid(
    framework: Framework, spacing: float
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Distance from each grid point to the nearest vdW surface.

    Returns the (n1*n2*n3,) distance array (negative inside atoms) and
    the grid dimensions. Periodicity is handled by replicating atoms
    into the 27 neighboring images; one KD-tree per element keeps the
    per-element vdW offset exact.
    """
    cell = framework.cell
    a, b, c = framework.lattice.abc
    dims = (
        max(2, math.ceil(a / spacing)),
        max(2, math.ceil(b / spacing)),
        max(2, math.ceil(c / spacing)),
    )
    axes = [(np.arange(n) + 0.5) / n for n in dims]
    frac = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    points = frac @ cell

    atom_frac = framework.cart_coords @ np.linalg.inv(cell) % 1.0
    shifts = np.array(
        [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=float,
    )
    symbols = np.array(framework.symbols)
    distances = np.full(len(points), np.inf)
    for symbol in sorted(set(symbols)):
        element_frac = atom_frac[symbols == symbol]
        images = (element_frac[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
        tree = cKDTree(images @ cell)
        raw, _ = tree.query(points)
        distances = np.minimum(distances, raw - _vdw_radius(symbol))
    return distances, dims


def void_fraction(
    framework: Framework,
    probe_radius: float = 0.0,
    spacing: float = DEFAULT_SPACING,
) -> float:
    """Fraction of the cell volume where a probe sphere fits.

    Parameters
    ----------
    framework : Framework
        The framework to analyze.
    probe_radius : float, optional
        Probe radius in Angstrom; 0.0 (default) gives the geometric
        void fraction outside the vdW surfaces. 1.2 approximates a
        helium probe, 1.86 nitrogen.
    spacing : float, optional
        Grid resolution in Angstrom.

    Returns
    -------
    float
        Accessible-volume fraction, in [0, 1].
    """
    distances, _ = _distance_grid(framework, spacing)
    return _void_fraction_from(distances, probe_radius)


def largest_cavity_diameter(
    framework: Framework, spacing: float = DEFAULT_SPACING
) -> float:
    """Diameter of the largest sphere fitting anywhere in the cell (LCD).

    Returns
    -------
    float
        The LCD in Angstrom; 0.0 when no grid point lies outside the
        vdW surfaces (a dense structure).
    """
    distances, _ = _distance_grid(framework, spacing)
    return _lcd_from(distances)


def pore_limiting_diameter(
    framework: Framework, spacing: float = DEFAULT_SPACING
) -> float:
    """Diameter of the largest sphere that can travel through the net (PLD).

    Binary-searches the probe radius: at each radius, the grid points
    with enough clearance form a graph (periodic 6-neighbor
    connectivity), and a wrap-detecting union-find checks whether any
    open component connects to its own periodic image - the definition
    of a percolating channel, in any direction.

    Returns
    -------
    float
        The PLD in Angstrom; 0.0 when not even a point probe
        percolates (a closed-pore or dense structure).
    """
    distances, dims = _distance_grid(framework, spacing)
    return _pld_from(distances, dims, spacing)


def _void_fraction_from(distances: np.ndarray, probe_radius: float) -> float:
    return float(np.count_nonzero(distances >= probe_radius) / len(distances))


def _lcd_from(distances: np.ndarray) -> float:
    return max(0.0, 2.0 * float(distances.max()))


def _pld_from(
    distances: np.ndarray, dims: tuple[int, int, int], spacing: float
) -> float:
    hi = float(distances.max())
    if hi <= 0.0 or not _percolates(distances >= 0.0, dims):
        return 0.0
    lo = 0.0
    # bisect until the radius bracket is far below the grid resolution
    while hi - lo > spacing / 8.0:
        mid = 0.5 * (lo + hi)
        if _percolates(distances >= mid, dims):
            lo = mid
        else:
            hi = mid
    return 2.0 * lo


@dataclass(frozen=True)
class PorosityReport:
    """The three geometric descriptors, from one shared distance grid."""

    void_fraction: float
    largest_cavity_diameter: float
    pore_limiting_diameter: float


def porosity_report(
    framework: Framework,
    probe_radius: float = 0.0,
    spacing: float = DEFAULT_SPACING,
) -> PorosityReport:
    """All three porosity descriptors of a framework, computed together.

    Building the periodic distance grid dominates the cost of every
    descriptor, and the individual functions each rebuild it - so the
    common "characterize this framework" call pattern pays for it three
    times. This computes the grid once. Values are identical to the
    individual functions at the same ``spacing``.
    """
    distances, dims = _distance_grid(framework, spacing)
    return PorosityReport(
        void_fraction=_void_fraction_from(distances, probe_radius),
        largest_cavity_diameter=_lcd_from(distances),
        pore_limiting_diameter=_pld_from(distances, dims, spacing),
    )


def _percolates(open_mask: np.ndarray, dims: tuple[int, int, int]) -> bool:
    """Whether the open grid cells contain a periodically wrapping path.

    Two-stage quotient construction. ``scipy.ndimage.label`` first
    finds the NON-periodic connected components of the open cells (C
    speed, 6-connectivity - the same neighbor convention the previous
    per-cell Python union-find walked in ~10^5 iterations per call).
    Any two cells of one component are connected with zero net wrap, so
    each component contracts to a single vertex, and only the
    boundary-face edges (last slice -> first slice per axis, wrap +1 on
    that axis) remain. The wrap-tracking union-find then runs over
    those component-level edges - typically tens of vertices instead of
    the full grid. Joining two components that already share a root
    through an edge whose wrap disagrees with the recorded counts means
    a component connects to its own periodic image - a percolating
    channel, in any direction.
    """
    open_grid = open_mask.reshape(dims)
    labels, n_labels = ndimage.label(open_grid)
    if not n_labels:
        return False
    parent = list(range(n_labels + 1))
    wraps: list[tuple[int, int, int]] = [(0, 0, 0)] * (n_labels + 1)

    def find(node: int) -> tuple[int, tuple[int, int, int]]:
        # accumulate the wrap count along the chain to the root
        root = node
        w0 = w1 = w2 = 0
        while parent[root] != root:
            r0, r1, r2 = wraps[root]
            w0, w1, w2 = w0 + r0, w1 + r1, w2 + r2
            root = parent[root]
        # path compression, keeping every wrap count consistent
        current = node
        a0, a1, a2 = w0, w1, w2
        while parent[current] != root:
            following = parent[current]
            s0, s1, s2 = wraps[current]
            parent[current] = root
            wraps[current] = (a0, a1, a2)
            a0, a1, a2 = a0 - s0, a1 - s1, a2 - s2
            current = following
        return root, (w0, w1, w2)

    unit_steps = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    for axis in range(3):
        last = labels[(slice(None),) * axis + (-1,)].ravel()
        first = labels[(slice(None),) * axis + (0,)].ravel()
        both_open = (last > 0) & (first > 0)
        pairs = np.unique(np.stack([last[both_open], first[both_open]], axis=1), axis=0)
        step = unit_steps[axis]
        for a, b in pairs.tolist():
            root_a, (a0, a1, a2) = find(a)
            root_b, (b0, b1, b2) = find(b)
            delta = (a0 + step[0] - b0, a1 + step[1] - b1, a2 + step[2] - b2)
            if root_a == root_b:
                if delta != (0, 0, 0):
                    return True
                continue
            # attach root_b under root_a with the consistent wrap count
            parent[root_b] = root_a
            wraps[root_b] = delta
    return False
