"""
Canonical rod repeat units: identity and dedup for 1-periodic SBUs.

A rod building unit (Stage A: ``deconstruct.RodUnit``) has no finite
fragment, so ``merge_fragment``'s finite-geometry identity cannot
deduplicate rods across a harvest. Rod identity is the *chemical*
repeat unit modulo everything the crystal embedding chooses freely:
rotation about the rod axis, axial phase, translation — and the
proper flip (a 180-degree rotation about a perpendicular axis), which
preserves helicity, so enantiomeric screw rods remain distinct, in
line with the builder's proper-rotations-only invariant.

``canonical_rod`` turns a detected rod into a :class:`RodRepeat`:

- atoms of one crystallographic repeat are reconstructed locally by
  walking the rod's own bond graph (intra-cell bonds first, so a screw
  rod lays out monotonically along its axis) and expressed in a
  cylindrical frame about the rod axis through their perpendicular
  centroid — the only transverse origin a screw symmetry can have;
- the *chemical* repeat is found by self-matching under a screw
  operation (axial shift + rotation): MOF-74-style rods have a
  crystallographic repeat that is an integer multiple of the chemical
  one, and a 2x supercell deconstruction must dedupe with a 1x one;
- the frame is truncated to one chemical repeat, keeping the screw
  angle as part of the identity.

``RodRepeat.matches`` then compares two rods by searching anchor
pairings for an (axial shift, rotation, optional proper flip) that
superimposes them element-wise within a cartesian tolerance
(Hungarian assignment per element). ``merge_rod`` mirrors
``deconstruct.merge_fragment`` on top of it, and ``harvest`` uses it
to build cross-structure rod families with provenance.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

    from autografs.deconstruct import RodUnit

logger = logging.getLogger(__name__)

# cartesian tolerance (Angstrom) for two rod repeats to be the same
# building unit; looser than exact geometry, tighter than "same shape"
ROD_MATCH_TOLERANCE = 0.3

# a rod whose screw angle stays under this (degrees) is treated as
# straight: its chemical repeats tile by pure translation, so the
# template bond graph is well defined and forward building applies
STRAIGHT_SCREW_TOL = 1.0


@dataclass(eq=False)
class RodRepeat:
    """One chemical repeat of a rod in its canonical cylindrical frame.

    Attributes
    ----------
    symbols : list[str]
        Element symbols of the atoms of one chemical repeat.
    axial : np.ndarray
        Position of each atom along the axis, in [0, repeat_length).
    radial : np.ndarray
        Distance of each atom from the rod axis.
    angular : np.ndarray
        Azimuth of each atom (radians); the global phase is arbitrary,
        matching happens modulo rotation.
    repeat_length : float
        The chemical repeat along the axis (Angstrom); the
        crystallographic repeat is ``screw_order`` times this.
    screw_order : int
        Chemical repeats per crystallographic repeat (1 = no screw).
    screw_angle : float
        Rotation (degrees, in (-180, 180]) accompanying one chemical
        translation; 0 for non-helical rods. Part of the identity —
        rods with different screw angles are different building units.
    n_connections : int
        Cut bonds (points of extension) per chemical repeat.
    """

    symbols: list[str]
    axial: np.ndarray = field(repr=False)
    radial: np.ndarray = field(repr=False)
    angular: np.ndarray = field(repr=False)
    repeat_length: float
    screw_order: int
    screw_angle: float
    n_connections: int

    @property
    def formula(self) -> str:
        """Composition of one chemical repeat, sorted symbols."""
        counts = Counter(self.symbols)
        return "".join(
            f"{symbol}{counts[symbol] if counts[symbol] > 1 else ''}"
            for symbol in sorted(counts)
        )

    def __repr__(self) -> str:
        return (
            f"RodRepeat({self.formula}, repeat={self.repeat_length:.2f} A, "
            f"screw={self.screw_angle:.1f} deg x{self.screw_order}, "
            f"{self.n_connections} connections)"
        )

    def _flipped(self) -> RodRepeat:
        """The rod after a proper 180-degree flip about a transverse
        axis: (theta, z) -> (-theta, -z). Preserves helicity."""
        length = self.repeat_length
        return RodRepeat(
            symbols=self.symbols,
            axial=(-self.axial) % length,
            radial=self.radial,
            angular=-self.angular,
            repeat_length=length,
            screw_order=self.screw_order,
            screw_angle=self.screw_angle,
            n_connections=self.n_connections,
        )

    def _pair_distances(
        self, other: RodRepeat, shift: float, rotation: float
    ) -> np.ndarray | None:
        """Max-matching distances after transforming ``other``.

        ``other`` is shifted axially and rotated, then matched to this
        rod element-by-element (Hungarian). Returns the per-pair
        distances, or None when the element multisets differ.
        """
        if Counter(self.symbols) != Counter(other.symbols):
            return None
        length = self.repeat_length
        z_other = (other.axial + shift) % length
        theta_other = other.angular + rotation
        distances = []
        for element in sorted(set(self.symbols)):
            mine = [i for i, s in enumerate(self.symbols) if s == element]
            theirs = [i for i, s in enumerate(other.symbols) if s == element]
            dz = np.abs(self.axial[mine][:, None] - z_other[theirs][None, :])
            dz = np.minimum(dz, length - dz)
            rho_a = self.radial[mine][:, None]
            rho_b = other.radial[theirs][None, :]
            dtheta = self.angular[mine][:, None] - theta_other[theirs][None, :]
            perp_sq = rho_a**2 + rho_b**2 - 2.0 * rho_a * rho_b * np.cos(dtheta)
            cost = np.sqrt(np.maximum(perp_sq, 0.0) + dz**2)
            rows, cols = linear_sum_assignment(cost)
            distances.extend(cost[rows, cols])
        return np.asarray(distances)

    def matches(self, other: RodRepeat, tolerance: float = ROD_MATCH_TOLERANCE) -> bool:
        """True when ``other`` is the same rod building unit.

        Identity is the chemical repeat modulo axial rotation, axial
        phase, and the proper flip; the screw angle and repeat length
        must agree. Enantiomeric helical rods do NOT match.
        """
        if Counter(self.symbols) != Counter(other.symbols):
            return False
        if abs(self.repeat_length - other.repeat_length) > tolerance:
            return False
        if self.n_connections != other.n_connections:
            return False
        # the screw angle sign is frame-invariant once the axial step
        # is normalized positive: axis reversal negates both theta and
        # z, and re-picking the smallest-positive-step generator
        # restores the original signed angle. Enantiomeric screws
        # (+90 vs -90) therefore differ here - and must, since no
        # proper isometry relates them (helicity is chiral)
        delta = (self.screw_angle - other.screw_angle + 180.0) % 360.0 - 180.0
        if abs(delta) > 5.0:
            return False
        anchor = self._anchor_element()
        a0 = next(i for i, s in enumerate(self.symbols) if s == anchor)
        for candidate in (other, other._flipped()):
            for b in (i for i, s in enumerate(candidate.symbols) if s == anchor):
                shift = float(self.axial[a0] - candidate.axial[b])
                rotation = float(self.angular[a0] - candidate.angular[b])
                distances = self._pair_distances(candidate, shift, rotation)
                if distances is not None and distances.max() <= tolerance:
                    return True
        return False

    def _anchor_element(self) -> str:
        """The rarest element (deterministic tie-break): fewest anchor
        pairings to try."""
        counts = Counter(self.symbols)
        return min(sorted(counts), key=lambda s: counts[s])


def _local_positions(
    structure: Structure,
    indices: list[int],
    anchor_index: int,
    internal_bonds: list[tuple[int, int, tuple[int, int, int]]] | None = None,
) -> np.ndarray:
    """Contiguous cartesian positions of a rod's atoms.

    The rod is laid out by walking its own bond graph from the anchor,
    each atom placed at its bonded neighbour plus the (image-corrected)
    bond vector, so the whole repeat is geometrically connected even
    when it spans more than half a cell. **Intra-cell bonds (zero image
    offset) are followed first**; the periodic-closure bonds (nonzero
    offset along the rod axis, which realize the 1-periodicity) are used
    only to bridge otherwise-disconnected atoms. This lays a screw rod
    out *monotonically along its axis* instead of folding a
    later-repeat atom back next to the anchor - so the chemical-repeat
    reduction keeps naturally-bonded atoms together.

    Falls back to minimum-image around the anchor when no bond graph is
    available (or an atom is unreachable), which recovers the old
    behaviour for thin rods.
    """
    matrix = structure.lattice.matrix
    anchor_coords = np.asarray(structure[anchor_index].coords, dtype=float)

    def _min_image(index: int) -> np.ndarray:
        delta = structure[index].frac_coords - structure[anchor_index].frac_coords
        delta -= np.round(delta)
        return np.asarray(anchor_coords + delta @ matrix)

    if not internal_bonds:
        return np.asarray([_min_image(i) for i in indices])

    # adjacency with image-corrected bond vectors, zero-offset bonds
    # first so the walk prefers them
    zero_adj: dict[int, list[tuple[int, np.ndarray]]] = {i: [] for i in indices}
    wrap_adj: dict[int, list[tuple[int, np.ndarray]]] = {i: [] for i in indices}
    for u, v, off in internal_bonds:
        vec = (
            structure[v].frac_coords
            + np.asarray(off, dtype=float)
            - structure[u].frac_coords
        ) @ matrix
        target = zero_adj if not any(off) else wrap_adj
        target[u].append((v, vec))
        target[v].append((u, -vec))

    placed: dict[int, np.ndarray] = {anchor_index: anchor_coords}
    frontier = [anchor_index]
    # first exhaust the zero-offset subgraph, then bridge with a wrap
    # bond, then exhaust zero-offset again, until every atom is placed
    while len(placed) < len(indices):
        while frontier:
            current = frontier.pop()
            for neighbour, vec in zero_adj.get(current, ()):
                if neighbour not in placed:
                    placed[neighbour] = placed[current] + vec
                    frontier.append(neighbour)
        bridged = False
        for node in list(placed):
            for neighbour, vec in wrap_adj.get(node, ()):
                if neighbour not in placed:
                    placed[neighbour] = placed[node] + vec
                    frontier.append(neighbour)
                    bridged = True
                    break
            if bridged:
                break
        if not bridged:
            break  # disconnected remainder: min-image them below

    return np.asarray([placed.get(i, _min_image(i)) for i in indices])


def _cylindrical_frame(
    positions: np.ndarray, axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(axial, radial, angular, basis) about the centroid axis.

    The axis line runs along ``axis`` through the perpendicular
    centroid of the atoms — the only transverse origin invariant under
    the rod's own screw/rotational symmetry, hence canonical. The
    returned (3, 3) basis rows (e1, e2, axis) are the frame the
    angular coordinates are measured in; vectors attached to the atoms
    (connection arms) transform into the same local frame by
    right-multiplying with its transpose.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    axial = positions @ axis
    perpendicular = positions - np.outer(axial, axis)
    center = perpendicular.mean(axis=0)
    perpendicular = perpendicular - center
    # transverse basis: any pair completing the axis; the angular
    # origin is arbitrary and matching is rotation-modulo anyway
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(seed @ axis)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = seed - (seed @ axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    radial = np.linalg.norm(perpendicular, axis=1)
    angular = np.arctan2(perpendicular @ e2, perpendicular @ e1)
    # atoms on the axis have no azimuth; zero is as good as any and
    # the distance metric ignores it there (rho = 0)
    angular[radial < 1e-9] = 0.0
    return axial, radial, angular, np.array([e1, e2, axis])


def _chemical_reduction(
    symbols: list[str],
    axial: np.ndarray,
    radial: np.ndarray,
    angular: np.ndarray,
    length: float,
    tolerance: float,
) -> tuple[float, int, float]:
    """The smallest screw operation mapping the repeat onto itself.

    Returns (chemical repeat, screw order, screw angle in degrees).
    Candidate axial shifts come from same-element anchor pairs; the
    smallest shift under which the whole atom set self-matches is the
    chemical translation, and its rotation is the screw angle.
    """
    probe = RodRepeat(
        symbols=symbols,
        axial=axial % length,
        radial=radial,
        angular=angular,
        repeat_length=length,
        screw_order=1,
        screw_angle=0.0,
        n_connections=0,
    )
    counts = Counter(symbols)
    anchor = min(sorted(counts), key=lambda s: counts[s])
    anchors = [i for i, s in enumerate(symbols) if s == anchor]
    a0 = anchors[0]
    candidates: list[tuple[float, float]] = []
    for b in anchors:
        if b == a0:
            continue
        shift = float((probe.axial[b] - probe.axial[a0]) % length)
        rotation = float(angular[b] - angular[a0])
        if shift > tolerance:
            candidates.append((shift, rotation))
    for shift, rotation in sorted(candidates):
        distances = probe._pair_distances(probe, shift, rotation)
        if distances is not None and distances.max() <= tolerance:
            order = int(round(length / shift))
            if order < 2 or abs(length / order - shift) > tolerance:
                continue
            angle = float(np.degrees(rotation))
            angle = (angle + 180.0) % 360.0 - 180.0
            return length / order, order, angle
    return length, 1, 0.0


@dataclass(eq=False)
class RodFragment:
    """A buildable rod: the canonical repeat plus connection arms.

    The forward-pipeline counterpart of a finite ``Fragment`` (rod
    Stage C): identity and matching live in ``repeat``; ``positions``
    is the chemical repeat's atom template in the canonical local
    frame (axis = +z through the transverse origin, z in
    [0, repeat_length)); ``arms`` carries the connection geometry —
    one entry per cut bond of the chemical repeat, as (template atom
    row, local vector from that atom to the cut-bond midpoint, i.e.
    where deconstruction would put the dummy).

    Attributes
    ----------
    repeat : RodRepeat
        The canonical chemical repeat (identity, screw operation).
    positions : np.ndarray
        (n, 3) local cartesian coordinates of the template atoms.
    arms : list[tuple[int, np.ndarray]]
        Connection arms, (atom row, (3,) local vector). Empty when
        the source ``RodUnit`` carried no ``cut_vectors``.
    bonds : list[tuple[int, int, int]]
        The rod's internal bond graph as (row a, row b, m): atom a of
        one repeat bonds to atom b of the repeat m steps further along
        the axis (m = 0 within a repeat, m = +-1 the continuation).
        Recorded for screwless rods (screw_order 1) with source
        ``internal_bonds``; empty otherwise — forward building needs
        it, identity does not.
    name : str
        Library name, set by ``merge_rod``/harvest.
    """

    repeat: RodRepeat
    positions: np.ndarray = field(repr=False)
    arms: list[tuple[int, np.ndarray]] = field(repr=False)
    bonds: list[tuple[int, int, int]] = field(repr=False, default_factory=list)
    name: str = "rod"

    @property
    def symbols(self) -> list[str]:
        return self.repeat.symbols

    def matches(
        self,
        other: RodFragment | RodRepeat,
        tolerance: float = ROD_MATCH_TOLERANCE,
    ) -> bool:
        """Same rod building unit (identity delegates to the repeat)."""
        other_repeat = other.repeat if isinstance(other, RodFragment) else other
        return self.repeat.matches(other_repeat, tolerance)

    def __repr__(self) -> str:
        return (
            f"RodFragment({self.repeat.formula}, "
            f"repeat={self.repeat.repeat_length:.2f} A, "
            f"{len(self.arms)} arms)"
        )


def rod_fragment(
    structure: Structure,
    rod: RodUnit,
    tolerance: float = ROD_MATCH_TOLERANCE,
    name: str = "rod",
) -> RodFragment:
    """The buildable canonical fragment of a detected rod unit.

    Parameters
    ----------
    structure : Structure
        The deconstructed structure (``Deconstruction.structure``).
    rod : RodUnit
        A rod from ``Deconstruction.rod_units``. Its ``cut_vectors``
        (when present) become the fragment's connection arms.
    tolerance : float, optional
        Cartesian tolerance for the internal screw self-matching.
    name : str, optional
        Name recorded on the fragment.

    Returns
    -------
    RodFragment
        One chemical repeat (template + arms) in the canonical local
        frame, with the screw operation that generates the full
        crystallographic repeat.
    """
    anchor_index = rod.atom_indices[0]
    positions = _local_positions(
        structure, rod.atom_indices, anchor_index, rod.internal_bonds
    )
    axial, radial, angular, basis = _cylindrical_frame(positions, rod.axis)
    length = float(rod.repeat_length)
    symbols = [structure[i].specie.symbol for i in rod.atom_indices]
    axial = (axial - axial.min()) % length

    chemical, order, screw = _chemical_reduction(
        symbols, axial, radial, angular, length, tolerance
    )
    if order > 1 and rod.n_connections % order != 0:
        # the bare rod atoms have a screw symmetry the cut pattern
        # breaks; the connected building unit is the full
        # crystallographic repeat after all
        chemical, order, screw = length, 1, 0.0
    keep = np.ones(len(symbols), dtype=bool)
    if order > 1:
        # keep one chemical repeat: the atoms in the first axial slab
        keep = axial < chemical - 1e-9
        expected = len(symbols) // order
        if keep.sum() != expected:
            # slab boundary crossed an atom cluster; fall back to the
            # nearest-to-expected selection by axial order
            ranked = np.argsort(axial, kind="stable")[:expected]
            keep = np.zeros(len(symbols), dtype=bool)
            keep[ranked] = True

    # connection arms, rotated into the same local frame as the atoms
    row_of = {site: row for row, site in enumerate(rod.atom_indices)}
    kept_rows = np.flatnonzero(keep)
    remap = {int(old): new for new, old in enumerate(kept_rows)}
    arms: list[tuple[int, np.ndarray]] = []
    for site, vector in rod.cut_vectors:
        row = row_of[site]
        if not keep[row]:
            continue
        arms.append((remap[row], basis @ np.asarray(vector, dtype=float)))
    if rod.cut_vectors and len(arms) != rod.n_connections // order:
        logger.warning(
            f"Rod arm reduction kept {len(arms)} arms but expected "
            f"{rod.n_connections // order}; the cut pattern does not "
            "tile the chemical repeat evenly."
        )

    # internal bond graph in template rows. Repeats are related by a
    # screw (translation + rotation by screw_angle), so the slab
    # correspondence de-screws each atom's azimuth by its slab index
    # before matching it to the chemical-repeat template. This relies
    # on _local_positions laying the rod out monotonically along its
    # axis (intra-cell bonds first) so each atom's slab = floor(axial /
    # chemical) is unambiguous. Works for straight (screw 0) and
    # helical rods alike; identity does not need bonds - forward
    # building does.
    bonds_local: list[tuple[int, int, int]] = []
    if rod.internal_bonds:
        axis_hat = basis[2]
        matrix = structure.lattice.matrix
        screw_rad = np.radians(screw)
        template_axial_all = axial % chemical
        slab_all = np.floor(axial / chemical + 1e-6)
        kept = np.flatnonzero(keep)
        kept_axial = template_axial_all[kept]
        kept_radial = radial[kept]
        kept_angular = angular[kept]

        def _template_row(full_row: int) -> int:
            # nearest kept atom after de-screwing this atom's azimuth by
            # its slab: slab s sits at (rho, theta + s*screw, z + s*chem)
            slab = slab_all[full_row]
            da = np.abs(template_axial_all[full_row] - kept_axial)
            da = np.minimum(da, chemical - da)
            drho = np.abs(radial[full_row] - kept_radial)
            dth = np.abs(
                (angular[full_row] - slab * screw_rad - kept_angular + np.pi)
                % (2.0 * np.pi)
                - np.pi
            )
            return int(np.argmin(da + drho + radial[full_row] * dth))

        seen: set[tuple[int, int, int]] = set()
        for i, j, off in rod.internal_bonds:
            a = _template_row(row_of[i])
            b = _template_row(row_of[j])
            off_axial = float((np.asarray(off, dtype=float) @ matrix) @ axis_hat)
            n_i = round((axial[row_of[i]] - kept_axial[a]) / chemical)
            n_j = round((axial[row_of[j]] + off_axial - kept_axial[b]) / chemical)
            m = int(n_j - n_i)
            key = (a, b, m) if (a, b, m) <= (b, a, -m) else (b, a, -m)
            if key in seen:
                continue
            seen.add(key)
            bonds_local.append(key)

    symbols = [s for s, k in zip(symbols, keep, strict=True) if k]
    axial = axial[keep] % chemical
    radial = radial[keep]
    angular = angular[keep]
    repeat = RodRepeat(
        symbols=symbols,
        axial=axial,
        radial=radial,
        angular=angular,
        repeat_length=chemical,
        screw_order=order,
        screw_angle=screw,
        n_connections=rod.n_connections // order,
    )
    template = np.column_stack(
        [radial * np.cos(angular), radial * np.sin(angular), axial]
    )
    return RodFragment(
        repeat=repeat,
        positions=template,
        arms=arms,
        bonds=bonds_local,
        name=name,
    )


def canonical_rod(
    structure: Structure,
    rod: RodUnit,
    tolerance: float = ROD_MATCH_TOLERANCE,
) -> RodRepeat:
    """The canonical chemical repeat of a detected rod unit.

    Identity-only view of :func:`rod_fragment` (see there); kept as
    the Stage B entry point.
    """
    return rod_fragment(structure, rod, tolerance=tolerance).repeat


def merge_rod(
    library: dict,
    instance: RodFragment | RodRepeat,
    base_name: str,
) -> str:
    """Add a rod to a library under a deduplicated name.

    The counterpart of ``deconstruct.merge_fragment`` for rods: an
    instance matching an existing entry of the same base name reuses
    that name; a genuinely different rod sharing the base name gets a
    numeric suffix. Entries may be ``RodRepeat`` or ``RodFragment``
    (identity always compares the repeats). The library is mutated in
    place; the resolved name is returned.
    """

    def _repeat_of(entry: RodFragment | RodRepeat) -> RodRepeat:
        return entry.repeat if isinstance(entry, RodFragment) else entry

    name = base_name
    suffix = 1
    while name in library:
        if _repeat_of(library[name]).matches(_repeat_of(instance)):
            return name
        suffix += 1
        name = f"{base_name}_{suffix}"
    if isinstance(instance, RodFragment):
        instance.name = name
    library[name] = instance
    return name


ROD_FORMAT_VERSION = 1


def rod_fragment_to_dict(fragment: RodFragment) -> dict:
    """Convert a RodFragment into a JSON-compatible dict."""
    repeat = fragment.repeat
    return {
        "name": fragment.name,
        "symbols": list(repeat.symbols),
        "axial": np.round(repeat.axial, 8).tolist(),
        "radial": np.round(repeat.radial, 8).tolist(),
        "angular": np.round(repeat.angular, 8).tolist(),
        "repeat_length": repeat.repeat_length,
        "screw_order": repeat.screw_order,
        "screw_angle": repeat.screw_angle,
        "n_connections": repeat.n_connections,
        "positions": np.round(fragment.positions, 8).tolist(),
        "arms": [
            [int(row), np.round(np.asarray(vec), 8).tolist()]
            for row, vec in fragment.arms
        ],
        "bonds": [[int(a), int(b), int(m)] for a, b, m in fragment.bonds],
    }


def rod_fragment_from_dict(data: dict) -> RodFragment:
    """Reconstruct a RodFragment from its dict representation."""
    repeat = RodRepeat(
        symbols=list(data["symbols"]),
        axial=np.asarray(data["axial"], dtype=float),
        radial=np.asarray(data["radial"], dtype=float),
        angular=np.asarray(data["angular"], dtype=float),
        repeat_length=float(data["repeat_length"]),
        screw_order=int(data["screw_order"]),
        screw_angle=float(data["screw_angle"]),
        n_connections=int(data["n_connections"]),
    )
    return RodFragment(
        repeat=repeat,
        positions=np.asarray(data["positions"], dtype=float),
        arms=[(int(row), np.asarray(vec, dtype=float)) for row, vec in data["arms"]],
        bonds=[(int(a), int(b), int(m)) for a, b, m in data.get("bonds", [])],
        name=data.get("name", "rod"),
    )


def save_rods(rods: dict[str, RodFragment], path: str | Path) -> Path:
    """Write a rod fragment library to a versioned JSON file.

    Rods cannot join the XYZ SBU format (no finite molecule), so they
    get their own sidecar; gzip-compressed when the path ends in .gz.
    """
    path = Path(path)
    payload = {
        "format_version": ROD_FORMAT_VERSION,
        "rods": {name: rod_fragment_to_dict(rod) for name, rod in rods.items()},
    }
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")))
    else:
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    logger.info(f"Saved {len(rods)} rod fragment(s) to {path}")
    return path


def load_rods(path: str | Path) -> dict[str, RodFragment]:
    """Load a rod fragment library saved with ``save_rods``."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("format_version")
    if version != ROD_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported rod library format version {version!r} in {path}; "
            f"this build of AuToGraFS reads version {ROD_FORMAT_VERSION}."
        )
    return {
        name: rod_fragment_from_dict(data) for name, data in payload["rods"].items()
    }
