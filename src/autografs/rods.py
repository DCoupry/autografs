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

- atoms of one crystallographic repeat are reconstructed locally
  (minimum-image around an anchor atom) and expressed in a
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

import logging
from collections import Counter
from dataclasses import dataclass, field
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
    structure: Structure, indices: list[int], anchor_index: int
) -> np.ndarray:
    """Minimum-image cartesian positions around an anchor atom.

    Rods are thin compared to the cell, so reconstructing every atom
    at its image nearest the anchor recovers a geometrically connected
    copy of the repeat, regardless of how sites were wrapped.
    """
    matrix = structure.lattice.matrix
    anchor_frac = structure[anchor_index].frac_coords
    positions = []
    for index in indices:
        delta = structure[index].frac_coords - anchor_frac
        delta -= np.round(delta)
        positions.append(structure[anchor_index].coords + delta @ matrix)
    return np.asarray(positions)


def _cylindrical_frame(
    positions: np.ndarray, axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(axial, radial, angular) coordinates about the centroid axis.

    The axis line runs along ``axis`` through the perpendicular
    centroid of the atoms — the only transverse origin invariant under
    the rod's own screw/rotational symmetry, hence canonical.
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
    return axial, radial, angular


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


def canonical_rod(
    structure: Structure,
    rod: RodUnit,
    tolerance: float = ROD_MATCH_TOLERANCE,
) -> RodRepeat:
    """The canonical chemical repeat of a detected rod unit.

    Parameters
    ----------
    structure : Structure
        The deconstructed structure (``Deconstruction.structure``).
    rod : RodUnit
        A rod from ``Deconstruction.rod_units``.
    tolerance : float, optional
        Cartesian tolerance for the internal screw self-matching.

    Returns
    -------
    RodRepeat
        One chemical repeat in the canonical cylindrical frame, with
        the screw operation (order and angle) that generates the full
        crystallographic repeat from it.
    """
    anchor_index = rod.atom_indices[0]
    positions = _local_positions(structure, rod.atom_indices, anchor_index)
    axial, radial, angular = _cylindrical_frame(positions, rod.axis)
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
        symbols = [s for s, k in zip(symbols, keep, strict=True) if k]
        axial = axial[keep] % chemical
        radial = radial[keep]
        angular = angular[keep]
    return RodRepeat(
        symbols=symbols,
        axial=axial,
        radial=radial,
        angular=angular,
        repeat_length=chemical,
        screw_order=order,
        screw_angle=screw,
        n_connections=rod.n_connections // order,
    )


def merge_rod(
    library: dict[str, RodRepeat], instance: RodRepeat, base_name: str
) -> str:
    """Add a rod repeat to a library under a deduplicated name.

    The counterpart of ``deconstruct.merge_fragment`` for rods: an
    instance matching an existing entry of the same base name reuses
    that name; a genuinely different rod sharing the base name gets a
    numeric suffix. The library is mutated in place; the resolved name
    is returned.
    """
    name = base_name
    suffix = 1
    while name in library:
        if library[name].matches(instance):
            return name
        suffix += 1
        name = f"{base_name}_{suffix}"
    library[name] = instance
    return name
