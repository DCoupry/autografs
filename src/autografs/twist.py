"""
Commensurate twist angles and moiré bilayers for 2D frameworks.

An arbitrary twist between two copies of a layer destroys periodicity;
only coincidence-site-lattice (CSL) angles — rotations that map a
sublattice of the layer onto itself — yield a periodic moiré
supercell, and the supercell grows roughly as 1/theta^2 as the angle
shrinks. ``Framework.stack(mode="twisted", angle=...)`` therefore
snaps the requested angle to the nearest commensurate one and reports
exactly what it built.

The search (`commensurate_twists`) is plane-group-general: instead of
hardcoding the hexagonal formula (cos theta = (m^2 + 4mn + n^2) /
(2 (m^2 + mn + n^2))), it enumerates coincidence pairs — two lattice
vectors of (nearly) equal length — and clusters them by rotation
angle; two independent pairs at one angle define the moiré supercell.
On a hexagonal lattice this reproduces the closed-form family exactly;
on square lattices the 2*atan(n/m) family; on rectangular/oblique
lattices exact coincidences are rare and a small ``max_strain``
(strained-commensurate mode, standard in twistronics tooling) admits
approximate ones, with the residual absorbed as an in-plane strain of
the top layer and reported.

The bilayer builder reuses the stacking conventions: van-der-Waals
layers (no interlayer bonds), unwrapped coordinates, tag/slot
provenance offset per copy.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import networkx
import numpy as np

from autografs.exceptions import StackingError

if TYPE_CHECKING:
    from autografs.framework import Framework

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class TwistCandidate:
    """One commensurate (or strained-commensurate) twist angle.

    Attributes
    ----------
    angle : float
        The exact rotation angle in degrees (of the pure-rotation part
        of the top-layer transform), in (0, 180).
    n_cells : int
        Primitive layer cells per layer in the moiré supercell; the
        bilayer holds ``2 * n_cells * n_atoms_per_layer`` atoms.
    supercell : np.ndarray
        (2, 2) integer matrix: rows are the moiré supercell vectors in
        units of the layer's in-plane cell vectors.
    transform : np.ndarray
        (2, 2) exact in-plane map applied to the top layer (row-vector
        convention, ``xy' = xy @ transform``). A pure rotation when
        ``strain`` is 0; otherwise rotation composed with the small
        strain that closes the coincidence.
    strain : float
        Largest principal-stretch deviation from 1 of the transform
        (0 for exactly commensurate angles).
    """

    angle: float
    n_cells: int
    supercell: np.ndarray = field(repr=False)
    transform: np.ndarray = field(repr=False)
    strain: float = 0.0

    def __repr__(self) -> str:
        return (
            f"TwistCandidate(angle={self.angle:.4f}, n_cells={self.n_cells}, "
            f"strain={self.strain:.2e})"
        )


def _coincidence_pairs(
    basis: np.ndarray, max_index: int, max_strain: float
) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    """Ordered pairs of primitive lattice vectors of near-equal length.

    Each pair (v_indices, w_indices, angle) is a candidate coincidence:
    rotating v by ``angle`` (degrees, signed, taken positive) lands on
    w up to a relative length mismatch of at most ``max_strain``.
    """
    indices = [
        (i, j)
        for i in range(-max_index, max_index + 1)
        for j in range(-max_index, max_index + 1)
        if (i, j) != (0, 0) and math.gcd(abs(i), abs(j)) == 1
    ]
    vectors = {ij: np.asarray(ij, dtype=float) @ basis for ij in indices}
    lengths = {ij: float(np.linalg.norm(v)) for ij, v in vectors.items()}
    tolerance = max(max_strain, 1e-9)
    pairs = []
    for v_ij, v in vectors.items():
        for w_ij, w in vectors.items():
            if w_ij == v_ij:
                continue
            if abs(lengths[w_ij] / lengths[v_ij] - 1.0) > tolerance:
                continue
            angle = math.degrees(math.atan2(v[0] * w[1] - v[1] * w[0], float(v @ w)))
            if angle <= 1e-9:
                # the negative-angle twin of every candidate appears
                # with v and w exchanged; keep one chirality
                continue
            pairs.append((v_ij, w_ij, angle))
    return pairs


def _reduce_candidate(
    cluster: list[tuple[tuple[int, int], tuple[int, int], float]],
    basis: np.ndarray,
    max_strain: float,
) -> TwistCandidate | None:
    """Build the minimal supercell candidate from one angle cluster."""
    ordered = sorted(
        cluster, key=lambda p: float(np.linalg.norm(np.asarray(p[0]) @ basis))
    )
    v1_ij, w1_ij, _ = ordered[0]
    second = None
    for v_ij, w_ij, _ in ordered[1:]:
        if v1_ij[0] * v_ij[1] - v1_ij[1] * v_ij[0] != 0:
            second = (v_ij, w_ij)
            break
    if second is None:
        return None
    v2_ij, w2_ij = second
    v_int = np.array([v1_ij, v2_ij])
    w_int = np.array([w1_ij, w2_ij])
    if np.linalg.det(w_int) < 0:
        v_int = v_int[::-1]
        w_int = w_int[::-1]
    v_cart = v_int @ basis
    w_cart = w_int @ basis
    # the exact top-layer map carries the v basis onto the w basis
    transform = np.linalg.solve(v_cart, w_cart)
    u_mat, singulars, vt_mat = np.linalg.svd(transform)
    strain = float(np.abs(singulars - 1.0).max())
    if strain > max(max_strain, 1e-9):
        return None
    rotation = u_mat @ vt_mat
    angle = math.degrees(math.atan2(rotation[0, 1], rotation[0, 0]))
    if angle <= 0:
        return None
    n_cells = int(round(abs(np.linalg.det(w_int))))
    return TwistCandidate(
        angle=angle,
        n_cells=n_cells,
        supercell=w_int,
        transform=transform,
        strain=strain,
    )


def commensurate_twists(
    cell: np.ndarray, max_index: int = 8, max_strain: float = 0.0
) -> list[TwistCandidate]:
    """All commensurate twist angles of a layer lattice, by enumeration.

    Parameters
    ----------
    cell : np.ndarray
        The layer's cell matrix (3x3, slab convention: a and b span
        the plane) or a bare (2, 2) in-plane basis.
    max_index : int, optional
        Coincidence vectors are searched with integer components up to
        this bound, by default 8. Larger values find smaller angles
        (and larger supercells).
    max_strain : float, optional
        Relative length mismatch admitted between the coincidence
        vectors, by default 0 (exactly commensurate only). Non-zero
        values enable the strained-commensurate mode: the mismatch is
        absorbed as an in-plane strain of the top layer.

    Returns
    -------
    list[TwistCandidate]
        Distinct candidates sorted by angle, each with its exact
        angle, supercell, top-layer transform and strain. Angles
        related by the lattice's own point symmetry appear as separate
        entries (e.g. 21.79 and 38.21 degrees on a hexagonal lattice
        are the same moire family mirrored).
    """
    cell = np.asarray(cell, dtype=float)
    basis = cell[:2, :2]
    pairs = _coincidence_pairs(basis, max_index, max_strain)
    if not pairs:
        return []
    # cluster by angle: exact coincidences agree to float precision,
    # strained ones scatter by about the admitted strain
    width = max(1e-6, 2.0 * math.degrees(max_strain))
    pairs.sort(key=lambda p: p[2])
    clusters: list[list] = [[pairs[0]]]
    for pair in pairs[1:]:
        if pair[2] - clusters[-1][-1][2] <= width:
            clusters[-1].append(pair)
        else:
            clusters.append([pair])
    candidates = []
    seen: set[tuple[float, int]] = set()
    for cluster in clusters:
        candidate = _reduce_candidate(cluster, basis, max_strain)
        if candidate is None:
            continue
        key = (round(candidate.angle, 6), candidate.n_cells)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return sorted(candidates, key=lambda c: c.angle)


def _supercell_translations(
    basis: np.ndarray, supercell: np.ndarray
) -> list[np.ndarray]:
    """Cartesian lattice translations of ``basis`` inside ``supercell``.

    ``supercell`` is the (2, 2) cartesian matrix whose rows span the
    moiré cell; the returned translations tile it exactly once.
    """
    inverse = np.linalg.inv(supercell)
    # integer bounding box from the supercell corners in basis units
    corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]) @ supercell
    corner_indices = corners @ np.linalg.inv(basis)
    lo = np.floor(corner_indices.min(axis=0)).astype(int) - 1
    hi = np.ceil(corner_indices.max(axis=0)).astype(int) + 1
    translations = []
    for i in range(lo[0], hi[0] + 1):
        for j in range(lo[1], hi[1] + 1):
            t = np.array([i, j], dtype=float) @ basis
            frac = t @ inverse
            if np.all(frac > -1e-9) and np.all(frac < 1.0 - 1e-9):
                translations.append(t)
    expected = int(round(abs(np.linalg.det(supercell) / np.linalg.det(basis))))
    if len(translations) != expected:
        raise StackingError(
            f"Moiré tiling found {len(translations)} lattice points, "
            f"expected {expected}; the supercell is numerically "
            "degenerate."
        )
    return translations


def twisted_bilayer(
    framework: Framework,
    angle: float,
    interlayer: float,
    angle_tolerance: float = 1.0,
    max_strain: float = 0.0,
    max_index: int = 8,
    max_atoms: int = 20000,
) -> Framework:
    """A commensurate twisted bilayer of a 2D layer framework.

    The requested angle is snapped to the nearest commensurate twist
    of the layer's lattice; the moiré supercell holds one unrotated
    and one rotated copy of the layer, van-der-Waals stacked. The
    exact angle and residual strain are recorded on the result's graph
    (``twist_angle``, ``twist_strain`` attributes) and logged.

    Parameters
    ----------
    framework : Framework
        A flat layer in the padded-slab convention (as any 2D build).
    angle : float
        Requested twist angle in degrees, in (0, 180).
    interlayer : float
        Spacing between the layer planes in Angstrom.
    angle_tolerance : float, optional
        Largest acceptable snap distance in degrees, by default 1.0.
    max_strain : float, optional
        Strained-commensurate tolerance (see ``commensurate_twists``),
        by default 0 — exactly commensurate only.
    max_index : int, optional
        Search bound for coincidence vectors, by default 8.
    max_atoms : int, optional
        Guardrail: refuse supercells with more atoms than this, by
        default 20000. Small angles genuinely need huge cells (the
        moiré period grows as 1/theta); raise the cap deliberately.

    Returns
    -------
    Framework
        The bilayer; ``framework`` is unchanged.

    Raises
    ------
    StackingError
        If no commensurate angle lies within the tolerance, or the
        moiré supercell exceeds ``max_atoms``.
    """
    from autografs.editing import replicated_graph
    from autografs.framework import Framework as FrameworkCls

    candidates = commensurate_twists(
        framework.cell, max_index=max_index, max_strain=max_strain
    )
    if not candidates:
        raise StackingError(
            f"No commensurate twist found for {framework.name!r} with "
            f"max_index={max_index} and max_strain={max_strain}; an "
            "oblique lattice may need a non-zero max_strain."
        )
    matches = [c for c in candidates if abs(c.angle - angle) <= angle_tolerance]
    if not matches:
        nearest = sorted(candidates, key=lambda c: abs(c.angle - angle))[:5]
        listing = ", ".join(f"{c.angle:.3f}°(x{c.n_cells})" for c in nearest)
        raise StackingError(
            f"No commensurate twist within {angle_tolerance}° of "
            f"{angle}° for {framework.name!r}; nearest: {listing}. "
            "Widen angle_tolerance, raise max_index, or allow "
            "max_strain."
        )
    chosen = min(
        matches, key=lambda c: (round(abs(c.angle - angle), 6), c.n_cells, c.strain)
    )
    n_total = 2 * chosen.n_cells * len(framework)
    if n_total > max_atoms:
        raise StackingError(
            f"The {chosen.angle:.3f}° moiré supercell holds {n_total} "
            f"atoms (cap {max_atoms}); raise max_atoms to build it "
            "anyway."
        )

    cell = framework.cell
    basis = cell[:2, :2]
    super2d = chosen.supercell.astype(float) @ basis
    new_cell = np.zeros((3, 3))
    new_cell[:2, :2] = super2d
    new_cell[2, 2] = 2.0 * interlayer

    # bottom layer: unrotated copies tiling the moiré cell
    bottom_shifts = [
        np.array([t[0], t[1], 0.0]) for t in _supercell_translations(basis, super2d)
    ]
    combined = replicated_graph(framework, bottom_shifts, cell=new_cell)

    # top layer: the exact commensurate transform (rotation + any
    # admitted strain) applied in-plane, then tiled and lifted
    top = _transformed_layer(framework, chosen.transform)
    top_basis = basis @ chosen.transform
    top_shifts = [
        np.array([t[0], t[1], interlayer])
        for t in _supercell_translations(top_basis, super2d)
    ]
    top_graph = replicated_graph(top, top_shifts, cell=new_cell)
    _append_graph(combined, top_graph)

    combined.graph["twist_angle"] = chosen.angle
    combined.graph["twist_strain"] = chosen.strain
    result = FrameworkCls(combined, name=f"{framework.name}_twist{chosen.angle:.2f}")
    logger.info(
        f"Built twisted bilayer of {framework.name!r}: {chosen.angle:.4f}° "
        f"(requested {angle}°), {chosen.n_cells} cells/layer, "
        f"{n_total} atoms, strain {chosen.strain:.2e}."
    )
    return result


def _transformed_layer(framework: Framework, transform: np.ndarray) -> Framework:
    """A copy of a layer with an in-plane linear map applied.

    Coordinates and the in-plane cell vectors are mapped (row-vector
    convention); z and the c axis are untouched. Bond lengths are
    preserved exactly for pure rotations and to first order in the
    strain otherwise.
    """
    from autografs.framework import Framework as FrameworkCls

    full = np.eye(3)
    full[:2, :2] = transform
    graph = networkx.Graph(cell=framework.cell @ full)
    for node in sorted(framework.graph):
        copied = dict(framework.graph.nodes[node])
        copied["coord"] = np.asarray(copied["coord"], dtype=float) @ full
        graph.add_node(node, **copied)
    graph.add_edges_from(framework.graph.edges(data=True))
    return FrameworkCls(graph, name=framework.name)


def _append_graph(base: networkx.Graph, extra: networkx.Graph) -> None:
    """Splice ``extra`` into ``base`` preserving uniqueness invariants.

    Node ids, positive anchor tags, and slot ids of ``extra`` are
    offset past ``base``'s maxima — the same invariants
    ``editing.replicated_graph`` maintains within one replication.
    """
    node_base = max(base, default=-1) + 1
    tag_base = max((d["tag"] for _, d in base.nodes(data=True)), default=0)
    slot_base = (
        max((d.get("slot", -1) for _, d in base.nodes(data=True)), default=-1) + 1
    )
    for node, data in extra.nodes(data=True):
        copied = dict(data)
        if copied["tag"] > 0:
            copied["tag"] += tag_base
        if "slot" in copied:
            copied["slot"] += slot_base
        base.add_node(node + node_base, **copied)
    for i, j, data in extra.edges(data=True):
        base.add_edge(i + node_base, j + node_base, **dict(data))
