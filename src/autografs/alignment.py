"""
Numpy alignment core for framework building.

This module aligns SBUs onto topology slots and optimizes the cell,
replacing the earlier pymatgen-object pipeline. Two ideas drive it:

1. **Directional matching.** Slot dummies sit at blueprint positions
   (quarter-edge points of a unit-scaled net) whose arm *lengths* carry
   no chemistry; only their *directions* do. SBUs are therefore matched
   to slots by optimally rotating their unit arm vectors onto the
   slot's unit arm vectors (Hungarian assignment + Kabsch, proper
   rotations only, so chiral SBUs are never silently mirrored).

2. **Pair-coincidence cell objective.** Every blueprint dummy site is
   shared by the two slots it connects (their tags coincide), and the
   built structure bonds those two SBU dummies together. The correct
   cell is therefore the one where paired dummy images coincide, and
   the optimization objective is the RMS distance between paired
   dummies - not the distance to blueprint positions, whose arbitrary
   arm lengths previously pulled cubic nets into anisotropic cells.

Everything is precomputed once per build into plain numpy arrays: the
cell optimization loop runs no pymatgen construction and no deep
copies.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Molecule
from scipy.optimize import linear_sum_assignment

from autografs.exceptions import AlignmentError
from autografs.fragment import Fragment
from autografs.topology import Topology

logger = logging.getLogger(__name__)

# Deterministic rotation multi-start for the direction matcher: the 24
# proper rotations of the cube spread well over SO(3) and are cheap.
_CUBE_ROTATIONS: np.ndarray = np.array(
    [
        m
        for m in (
            np.array(p, dtype=float) * np.array(s, dtype=float)
            for p in itertools.permutations(np.eye(3))
            for s in itertools.product((1.0, -1.0), repeat=3)
        )
        if np.isclose(np.linalg.det(m), 1.0)
    ]
)

# Iterations of the assignment <-> rotation refinement per start
_MATCH_ITERATIONS = 8


def kabsch(sources: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Optimal proper rotation taking sources onto targets.

    Parameters
    ----------
    sources, targets : np.ndarray
        (n, 3) paired coordinate sets, already centered.

    Returns
    -------
    np.ndarray
        (3, 3) rotation matrix R with det(R) = +1 minimizing
        sum_i |targets[i] - R @ sources[i]|^2. The determinant
        constraint means chiral arrangements are never mirrored.
    """
    covariance = sources.T @ targets
    u, _, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, sign])
    return vt.T @ correction @ u.T


def match_directions(
    targets: np.ndarray, arms: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Match SBU arm directions onto slot dummy directions.

    Runs a deterministic multi-start of alternating Hungarian
    assignment and Kabsch rotation, keeping the best result.

    Parameters
    ----------
    targets : np.ndarray
        (n, 3) unit vectors of the slot dummy directions.
    arms : np.ndarray
        (n, 3) unit vectors of the SBU arm directions.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        - R: (3, 3) proper rotation such that R @ arms[perm[i]] aims
          at targets[i].
        - perm: (n,) arm index assigned to each target.
        - rmsd: root mean square residual between the matched unit
          vectors (dimensionless, 0 for a perfect shape match, up to
          2 for opposite directions).
    """
    n = len(targets)
    if len(arms) != n:
        raise AlignmentError(
            f"Cannot match {len(arms)} arms onto {n} slot dummies."
        )
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for start in _CUBE_ROTATIONS:
        rotation = start
        perm = None
        for _ in range(_MATCH_ITERATIONS):
            rotated = arms @ rotation.T
            # cost = squared distance between unit vectors
            cost = -2.0 * (targets @ rotated.T)
            _, new_perm = linear_sum_assignment(cost)
            if perm is not None and np.array_equal(new_perm, perm):
                break
            perm = new_perm
            rotation = kabsch(arms[perm], targets)
        residual = targets - arms[perm] @ rotation.T
        score = float(np.sqrt((residual**2).sum(axis=1).mean()))
        if best is None or score < best[0]:
            best = (score, rotation, perm)
            if score < 1e-9:
                break
    score, rotation, perm = best
    return rotation, perm, score


def _unit(vectors: np.ndarray) -> np.ndarray:
    """Row-normalize a (n, 3) array."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms < 1e-9):
        raise AlignmentError("Degenerate (zero-length) arm vector.")
    return vectors / norms


@dataclass
class SlotPlacement:
    """Precomputed geometry for one slot and its assigned SBU."""

    slot_index: int
    frac_center: np.ndarray  # (3,) dummy centroid, fractional
    frac_arms: np.ndarray  # (n, 3) dummy positions minus center, fractional
    tags: list[int]  # (n,) blueprint tag per slot dummy
    sbu: Fragment
    arm_units: np.ndarray  # (n, 3) SBU unit arm vectors
    arm_lengths: np.ndarray  # (n,) SBU arm lengths, Angstrom
    sbu_center: np.ndarray  # (3,) SBU dummy centroid, cartesian
    dummy_indices: list[int]  # SBU atom index per arm
    # arm assignment (perm) computed at the reference cell
    arm_for_target: np.ndarray | None = field(default=None)

    def rotation_for(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rotation and target directions in the given cell matrix."""
        directions = _unit(self.frac_arms @ matrix)
        ordered_arms = self.arm_units[self.arm_for_target]
        return kabsch(ordered_arms, directions), directions

    def dummy_positions(self, matrix: np.ndarray) -> np.ndarray:
        """Cartesian SBU dummy positions (target order) in this cell."""
        rotation, _ = self.rotation_for(matrix)
        center = self.frac_center @ matrix
        ordered = (self.arm_units * self.arm_lengths[:, None])[self.arm_for_target]
        return center + ordered @ rotation.T


@dataclass
class BuildPlan:
    """All geometry needed to optimize and realize one framework build.

    Created once per build by prepare_build(); the cell optimization
    objective then runs on plain arrays with no object construction.
    """

    placements: list[SlotPlacement]
    # pairs of (placement index a, target index a, placement index b,
    # target index b, integer image offset) for tag-shared dummies
    pairs: list[tuple[int, int, int, int, np.ndarray]]
    blueprint_abc: np.ndarray  # (3,)
    angles: tuple[float, float, float]

    @property
    def has_pairs(self) -> bool:
        return bool(self.pairs)

    def matrix_for(self, scales: np.ndarray) -> np.ndarray:
        a, b, c = np.abs(scales)
        return Lattice.from_parameters(a, b, c, *self.angles).matrix

    def initial_scales(self) -> np.ndarray:
        """Analytic starting cell from required bond-through distances.

        Each dummy pair wants its two slot centers separated by the sum
        of the two arm lengths; the blueprint separation scales with
        the cell, so the ratio of sums is a near-optimal isotropic
        starting scale (exact for uninodal cubic nets).
        """
        blueprint = self.matrix_for(self.blueprint_abc)
        required = 0.0
        current = 0.0
        for index_a, target_a, index_b, target_b, offset in self.pairs:
            pa = self.placements[index_a]
            pb = self.placements[index_b]
            required += (
                pa.arm_lengths[pa.arm_for_target[target_a]]
                + pb.arm_lengths[pb.arm_for_target[target_b]]
            )
            separation = (pa.frac_center - pb.frac_center - offset) @ blueprint
            current += np.linalg.norm(separation)
        if not self.pairs or current < 1e-9:
            # no shared dummies: fall back to matching mean arm lengths
            arms = np.concatenate(
                [p.arm_lengths for p in self.placements]
            ).mean()
            slot_arms = np.concatenate(
                [
                    np.linalg.norm(p.frac_arms @ blueprint, axis=1)
                    for p in self.placements
                ]
            ).mean()
            return self.blueprint_abc * (arms / slot_arms)
        return self.blueprint_abc * (required / current)

    def residual(self, scales: np.ndarray) -> float:
        """RMS distance between paired dummy images at these scales."""
        matrix = self.matrix_for(scales)
        positions = [p.dummy_positions(matrix) for p in self.placements]
        total = 0.0
        for index_a, target_a, index_b, target_b, offset in self.pairs:
            delta = (
                positions[index_a][target_a]
                - positions[index_b][target_b]
                - offset @ matrix
            )
            total += float(delta @ delta)
        return float(np.sqrt(total / len(self.pairs)))

    def finalize(
        self, scales: np.ndarray
    ) -> tuple[list[Fragment], Lattice, dict[int, float]]:
        """Place every SBU rigidly in the final cell.

        Re-solves the full direction match (assignment included) at the
        final cell, then maps each SBU's atoms with the resulting rigid
        rotation + translation. Atoms keep their internal geometry;
        dummies receive the tags of their assigned slot dummies so the
        graph builder can bond paired fragments.

        Returns
        -------
        tuple[list[Fragment], Lattice, dict[int, float]]
            Aligned fragments, the final lattice, and the per-slot
            directional RMSD (dimensionless shape mismatch).
        """
        a, b, c = np.abs(scales)
        lattice = Lattice.from_parameters(a, b, c, *self.angles)
        matrix = lattice.matrix
        fragments: list[Fragment] = []
        slot_rmsds: dict[int, float] = {}
        for placement in self.placements:
            directions = _unit(placement.frac_arms @ matrix)
            rotation, perm, rmsd = match_directions(
                directions, placement.arm_units
            )
            placement.arm_for_target = perm
            slot_rmsds[placement.slot_index] = rmsd
            center = placement.frac_center @ matrix
            sbu = placement.sbu
            coords = (
                sbu.atoms.cart_coords - placement.sbu_center
            ) @ rotation.T + center
            tags = [0] * len(sbu.atoms)
            for target_index, arm_index in enumerate(perm):
                atom_index = placement.dummy_indices[arm_index]
                tags[atom_index] = placement.tags[target_index]
            aligned = Molecule(
                sbu.atoms.species, coords, site_properties={"tags": tags}
            )
            fragments.append(
                Fragment(
                    atoms=aligned,
                    name=sbu.name,
                    pointgroup=sbu.pointgroup,
                )
            )
        return fragments, lattice, slot_rmsds


def prepare_build(
    topology: Topology, mappings: dict[int, Fragment]
) -> BuildPlan:
    """Precompute all geometry for building one framework.

    Parameters
    ----------
    topology : Topology
        The blueprint; not modified.
    mappings : dict[int, Fragment]
        Slot index to (private) SBU fragment, as produced by
        Autografs._validate_mappings.

    Returns
    -------
    BuildPlan
        Arrays and the shared-dummy pair table for the optimizer.

    Raises
    ------
    AlignmentError
        If an SBU's dummy count does not match its slot's.
    """
    blueprint = topology.cell
    placements: list[SlotPlacement] = []
    # tag -> list of (placement index, target index, fractional position)
    tag_sites: dict[int, list[tuple[int, int, np.ndarray]]] = {}
    for slot_index, sbu in sorted(mappings.items()):
        slot = topology.slots[slot_index]
        dummy_idx = [
            i for i, site in enumerate(slot.atoms) if site.specie.symbol == "X"
        ]
        slot_frac = blueprint.get_fractional_coords(
            slot.atoms.cart_coords[dummy_idx]
        )
        frac_center = slot_frac.mean(axis=0)
        slot_tags = [int(slot.atoms[i].properties["tags"]) for i in dummy_idx]

        sbu_dummy_idx = [
            i for i, site in enumerate(sbu.atoms) if site.specie.symbol == "X"
        ]
        if len(sbu_dummy_idx) != len(dummy_idx):
            raise AlignmentError(
                f"SBU {sbu.name!r} has {len(sbu_dummy_idx)} connection "
                f"points but slot {slot_index} of {topology.name!r} "
                f"needs {len(dummy_idx)}."
            )
        sbu_dummies = sbu.atoms.cart_coords[sbu_dummy_idx]
        sbu_center = sbu_dummies.mean(axis=0)
        arm_vectors = sbu_dummies - sbu_center
        arm_lengths = np.linalg.norm(arm_vectors, axis=1)
        arm_units = _unit(arm_vectors)

        placement = SlotPlacement(
            slot_index=slot_index,
            frac_center=frac_center,
            frac_arms=slot_frac - frac_center,
            tags=slot_tags,
            sbu=sbu,
            arm_units=arm_units,
            arm_lengths=arm_lengths,
            sbu_center=sbu_center,
            dummy_indices=sbu_dummy_idx,
        )
        # reference assignment at the blueprint cell
        directions = _unit(placement.frac_arms @ blueprint.matrix)
        _, perm, _ = match_directions(directions, arm_units)
        placement.arm_for_target = perm
        index = len(placements)
        placements.append(placement)
        for target_index, (tag, frac) in enumerate(zip(slot_tags, slot_frac)):
            tag_sites.setdefault(tag, []).append((index, target_index, frac))

    pairs: list[tuple[int, int, int, int, np.ndarray]] = []
    unpaired = 0
    for tag, entries in sorted(tag_sites.items()):
        if len(entries) == 1:
            unpaired += 1
            continue
        if len(entries) > 2:
            logger.warning(
                f"Tag {tag} shared by {len(entries)} slots; using the "
                "first two."
            )
        (index_a, target_a, frac_a), (index_b, target_b, frac_b) = entries[:2]
        offset = np.round(frac_a - frac_b)
        pairs.append((index_a, target_a, index_b, target_b, offset))
    if unpaired:
        logger.debug(
            f"{unpaired} dummies are not shared between slots; they do "
            "not constrain the cell."
        )
    abc = np.array(blueprint.abc, dtype=float)
    return BuildPlan(
        placements=placements,
        pairs=pairs,
        blueprint_abc=abc,
        angles=tuple(blueprint.angles),
    )
