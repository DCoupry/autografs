"""
Blueprint symmetry operations, derived from the blueprint itself.

The embedding-relaxation work (#174) needs to displace slot centres
without breaking the net's declared symmetry: one displacement per
crystallographic orbit, restricted to the directions the site's
stabilizer allows, and propagated to the orbit's other slots by the
operations relating them. That needs the space-group *operations*, and
``Topology`` only records the spacegroup *number* - recovering
operations from the number alone would reintroduce a setting/origin
ambiguity against whatever convention the CGD idealization chose.

This module derives the operations numerically from the slot-centre
set instead, so they agree with the stored embedding by construction:

1. candidate rotation parts are the integer matrices preserving the
   cell's metric tensor (the lattice's point group, proper and
   improper - mirrors constrain displacements too);
2. each rotation is kept with every translation that maps the
   slot-centre set onto itself while preserving orbit labels;
3. a site's allowed displacement subspace is the common fixed space of
   its stabilizer's rotation parts;
4. an orbit representative's displacement propagates to each member
   through (the rotation part of) one operation mapping the
   representative onto it.

Everything is exact in the tolerance of the stored coordinates; the
identity operation always survives, so the machinery degrades to
"every orbit fully free" for a P1 blueprint rather than failing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product

import numpy as np

from autografs.topology import Topology

__all__ = [
    "SymmetryOperation",
    "OrbitDisplacements",
    "blueprint_operations",
    "orbit_displacements",
]

logger = logging.getLogger(__name__)

# fractional-coordinate agreement for "the same site": CGD-derived
# blueprints store far more precision than this, and idealized
# coordinates are often exact rationals
_SITE_TOL = 1e-4

# metric-tensor agreement, relative to the largest entry
_METRIC_TOL = 1e-4


@dataclass(frozen=True)
class SymmetryOperation:
    """One space-group operation of a blueprint, in fractional space.

    Maps fractional coordinates as ``x -> rotation @ x + translation``
    (modulo lattice translations). ``rotation`` is an integer matrix in
    the lattice basis; improper operations (det = -1) are included.
    ``slot_image[i]`` is the index of the slot that slot ``i`` lands on.
    """

    rotation: tuple[tuple[int, ...], ...]
    translation: tuple[float, ...]
    slot_image: tuple[int, ...]

    @property
    def rotation_array(self) -> np.ndarray:
        return np.asarray(self.rotation, dtype=float)

    @property
    def is_proper(self) -> bool:
        return bool(np.linalg.det(self.rotation_array) > 0)


def _lattice_point_group(matrix: np.ndarray) -> list[np.ndarray]:
    """Integer matrices W with W^T G W = G for the cell's metric G.

    Entries are searched in {-1, 0, 1}: sufficient for every
    conventional crystallographic setting (a centred or badly skewed
    cell could in principle need +-2, but no CGD blueprint uses one;
    the identity always survives regardless, so the failure mode is
    missing operations, never wrong ones).
    """
    metric = matrix @ matrix.T
    scale = float(np.abs(metric).max())
    candidates = []
    columns = [np.array(c) for c in product((-1, 0, 1), repeat=3) if any(c)]
    # build W column by column, pruning on the diagonal of the metric
    # condition first (a full 3^9 scan is ~20k matrices; this is
    # equivalent but keeps the inner check cheap)
    for w in product(columns, repeat=3):
        candidate = np.array(w, dtype=int).T
        if abs(round(float(np.linalg.det(candidate)))) != 1:
            continue
        transformed = candidate.T @ metric @ candidate
        if np.allclose(transformed, metric, atol=_METRIC_TOL * scale):
            candidates.append(candidate)
    return candidates


def _slot_centers(topology: Topology) -> tuple[np.ndarray, list[int]]:
    """Fractional dummy-centroid of every slot, plus its orbit label."""
    centers = []
    orbits = []
    for slot in topology.slots:
        dummies = [i for i, site in enumerate(slot.atoms) if site.specie.symbol == "X"]
        frac = topology.cell.get_fractional_coords(slot.atoms.cart_coords[dummies])
        centers.append(frac.mean(axis=0))
        orbit = getattr(slot, "equivalence_class", None)
        orbits.append(-1 if orbit is None else int(orbit))
    return np.asarray(centers), orbits


def _match_sites(
    mapped: np.ndarray, centers: np.ndarray, orbits: list[int]
) -> list[int] | None:
    """Match mapped centres onto the centre set, modulo lattice
    translations and preserving orbit labels; None when any site has no
    partner or two sites claim the same one."""
    image: list[int] = []
    used: set[int] = set()
    for index, site in enumerate(mapped):
        delta = centers - site
        delta -= np.round(delta)
        distances = np.abs(delta).max(axis=1)
        partner = int(np.argmin(distances))
        if (
            distances[partner] > _SITE_TOL
            or partner in used
            or orbits[partner] != orbits[index]
        ):
            return None
        used.add(partner)
        image.append(partner)
    return image


def blueprint_operations(topology: Topology) -> list[SymmetryOperation]:
    """Space-group operations of a blueprint, from its own embedding.

    Every operation maps the slot-centre set onto itself (modulo
    lattice translations) with orbit labels preserved, so orbits are
    never merged even when two of them happen to be geometrically
    superimposable. The identity is always present. For a 2D layer
    blueprint the operations are still exact for the stored slab
    embedding (c maps to +-c).

    Parameters
    ----------
    topology : Topology
        The blueprint; not modified.

    Returns
    -------
    list[SymmetryOperation]
        All operations found, identity first. Proper and improper.
    """
    centers, orbits = _slot_centers(topology)
    if not len(centers):
        return []
    operations: list[SymmetryOperation] = []
    seen: set[tuple] = set()
    # candidate translations: whatever takes a reference slot (of the
    # rarest orbit, to minimize candidates) onto a same-orbit slot
    orbit_counts = {orbit: orbits.count(orbit) for orbit in set(orbits)}
    reference = min(range(len(centers)), key=lambda i: (orbit_counts[orbits[i]], i))
    same_orbit = [i for i, orbit in enumerate(orbits) if orbit == orbits[reference]]
    for rotation in _lattice_point_group(topology.cell.matrix):
        rotated = centers @ rotation.T
        for target in same_orbit:
            translation = centers[target] - rotated[reference]
            translation -= np.floor(np.round(translation, 6))
            image = _match_sites(rotated + translation, centers, orbits)
            if image is None:
                continue
            key = (
                tuple(map(tuple, rotation)),
                tuple(np.round(translation, 4)),
            )
            if key in seen:
                continue
            seen.add(key)
            operations.append(
                SymmetryOperation(
                    rotation=tuple(tuple(int(x) for x in row) for row in rotation),
                    translation=tuple(float(x) for x in translation),
                    slot_image=tuple(image),
                )
            )
    # identity first, deterministic order after
    operations.sort(
        key=lambda op: (
            op.rotation != ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            op.rotation,
            op.translation,
        )
    )
    return operations


@dataclass(frozen=True)
class OrbitDisplacements:
    """The symmetry-allowed slot displacements of one blueprint.

    The free parameters are one fractional displacement per orbit,
    restricted to the orbit representative's site-symmetry subspace;
    ``expand`` turns the packed parameter vector into a per-slot
    displacement array with every orbit's members moved consistently.

    Attributes
    ----------
    orbit_of_slot : tuple[int, ...]
        Orbit label per slot index.
    representatives : dict[int, int]
        Orbit label -> representative slot index.
    bases : dict[int, np.ndarray]
        Orbit label -> (n_free, 3) orthonormal basis (fractional space,
        metric-orthonormalized) of the representative's allowed
        subspace. An empty (0, 3) basis means the orbit is pinned.
    propagation : tuple[np.ndarray, ...]
        Per slot: the rotation part taking the representative's
        displacement to this slot's, i.e. d_slot = W @ d_rep.
    """

    orbit_of_slot: tuple[int, ...]
    representatives: dict[int, int]
    bases: dict[int, np.ndarray]
    propagation: tuple[np.ndarray, ...]

    @property
    def n_free(self) -> int:
        """Total free scalar parameters across all orbits."""
        return sum(len(basis) for basis in self.bases.values())

    def expand(self, free: np.ndarray) -> np.ndarray:
        """Per-slot fractional displacements from the packed vector.

        Parameters are consumed orbit by orbit in ascending orbit-label
        order, ``len(basis)`` scalars each.
        """
        free = np.asarray(free, dtype=float)
        if len(free) != self.n_free:
            raise ValueError(
                f"Expected {self.n_free} slot parameters, got {len(free)}."
            )
        rep_displacement: dict[int, np.ndarray] = {}
        cursor = 0
        for orbit in sorted(self.bases):
            basis = self.bases[orbit]
            take = len(basis)
            rep_displacement[orbit] = (
                free[cursor : cursor + take] @ basis if take else np.zeros(3)
            )
            cursor += take
        return np.array(
            [
                self.propagation[slot] @ rep_displacement[self.orbit_of_slot[slot]]
                for slot in range(len(self.orbit_of_slot))
            ]
        )


def _fixed_subspace(rotations: list[np.ndarray], matrix: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the common fixed space of the rotations.

    Solved in cartesian coordinates (fractional rotation parts are
    similar to orthogonal matrices through the cell, and orthonormality
    is only meaningful in the metric), then mapped back to fractional.
    """
    inverse = np.linalg.inv(matrix)
    accumulated = np.zeros((0, 3))
    for rotation in rotations:
        cartesian = matrix.T @ rotation @ inverse.T
        accumulated = np.vstack([accumulated, cartesian - np.eye(3)])
    if not len(accumulated):
        return inverse.copy()  # no constraints: all of R^3
    _, singular, vt = np.linalg.svd(accumulated)
    rank = int((singular > 1e-6).sum()) if len(singular) else 0
    cartesian_basis = vt[rank:] if rank < 3 else np.zeros((0, 3))
    # cartesian direction -> fractional displacement
    return cartesian_basis @ inverse


def orbit_displacements(
    topology: Topology, operations: list[SymmetryOperation] | None = None
) -> OrbitDisplacements:
    """Symmetry-allowed slot-displacement parametrization of a blueprint.

    For each orbit, one representative gets the displacement freedom
    its site's stabilizer allows (the common fixed space of the
    stabilizing rotations); every other member of the orbit moves as
    the image of the representative's displacement under one operation
    relating them. A blueprint whose every site is fully pinned - pcu,
    and high-symmetry nets generally - yields ``n_free == 0``, which is
    exactly why those nets build correctly today.

    Parameters
    ----------
    topology : Topology
        The blueprint; not modified.
    operations : list[SymmetryOperation] or None, optional
        Reuse operations from blueprint_operations; computed when
        omitted.

    Returns
    -------
    OrbitDisplacements
    """
    if operations is None:
        operations = blueprint_operations(topology)
    centers, orbits = _slot_centers(topology)
    n_slots = len(centers)
    matrix = np.asarray(topology.cell.matrix)
    representatives: dict[int, int] = {}
    for slot in range(n_slots):
        representatives.setdefault(orbits[slot], slot)
    # one operation taking the representative onto each member; the
    # identity (always present) covers the representative itself
    propagation: list[np.ndarray | None] = [None] * n_slots
    for orbit, representative in representatives.items():
        for operation in operations:
            target = operation.slot_image[representative]
            if propagation[target] is None and orbits[target] == orbit:
                propagation[target] = operation.rotation_array
    unreached = [i for i, w in enumerate(propagation) if w is None]
    if unreached:
        # orbit labels finer than the found symmetry (or operations
        # missing): those slots become their own representatives
        logger.debug(
            f"{len(unreached)} slots not reached from their orbit "
            f"representative on {topology.name!r}; freeing them "
            "independently."
        )
        for slot in unreached:
            orbits[slot] = max(orbits) + 1
            representatives[orbits[slot]] = slot
            propagation[slot] = np.eye(3)
        # rebuild orbit_of_slot consistency below
    bases: dict[int, np.ndarray] = {}
    for orbit, representative in representatives.items():
        stabilizer = [
            operation.rotation_array
            for operation in operations
            if operation.slot_image[representative] == representative
        ]
        bases[orbit] = _fixed_subspace(stabilizer, matrix)
    return OrbitDisplacements(
        orbit_of_slot=tuple(orbits),
        representatives=representatives,
        bases=bases,
        propagation=tuple(w for w in propagation if w is not None),
    )
