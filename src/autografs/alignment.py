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

2. **Bond-length pair objective.** Every blueprint dummy site is
   shared by the two slots it connects (their tags coincide), and the
   built structure bonds the two SBU atoms that carried those dummies
   (the *anchors*). The correct cell is therefore the one where each
   paired anchor image sits one covalent bond length from its partner,
   and the optimization objective is the RMS deviation of the paired
   anchor distances from their Cordero covalent-radius targets - not
   the distance to blueprint positions, whose arbitrary arm lengths
   previously pulled cubic nets into anisotropic cells, and not dummy
   coincidence, which hard-codes whatever bond length the SBU library's
   dummy placement implies (0.7 A dummies = a 1.4 A bond for every
   element pair; MOF-5 came out at 12.77 A instead of ~12.9).

Everything is precomputed once per build into plain numpy arrays: the
cell optimization loop runs no pymatgen construction and no deep
copies.
"""

from __future__ import annotations

import functools
import itertools
import logging
from dataclasses import dataclass

import numpy as np
from pymatgen.analysis.local_env import CovalentRadius
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Molecule
from scipy.optimize import linear_sum_assignment

from autografs import plane_groups
from autografs.exceptions import AlignmentError
from autografs.fragment import Fragment
from autografs.topology import Topology

__all__ = [
    "kabsch",
    "match_directions",
    "CellParametrization",
    "SlotPlacement",
    "BuildPlan",
    "prepare_build",
]

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
    return np.asarray(vt.T @ correction @ u.T)


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
        raise AlignmentError(f"Cannot match {len(arms)} arms onto {n} slot dummies.")
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for start in _CUBE_ROTATIONS:
        rotation = start
        # impossible assignment: never equal to a real permutation, so
        # the first iteration always refines
        perm = np.full(n, -1)
        for _ in range(_MATCH_ITERATIONS):
            rotated = arms @ rotation.T
            # cost = squared distance between unit vectors
            cost = -2.0 * (targets @ rotated.T)
            _, new_perm = linear_sum_assignment(cost)
            if np.array_equal(new_perm, perm):
                break
            perm = new_perm
            rotation = kabsch(arms[perm], targets)
        residual = targets - arms[perm] @ rotation.T
        score = float(np.sqrt((residual**2).sum(axis=1).mean()))
        if best is None or score < best[0]:
            best = (score, rotation, perm)
            if score < 1e-9:
                break
    if best is None:  # pragma: no cover - _CUBE_ROTATIONS is never empty
        raise AlignmentError("No rotation start produced a match.")
    score, rotation, perm = best
    return rotation, perm, score


def _unit(vectors: np.ndarray) -> np.ndarray:
    """Row-normalize a (n, 3) array."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms < 1e-9):
        raise AlignmentError("Degenerate (zero-length) arm vector.")
    return np.asarray(vectors / norms)


# cell angles are clipped into this range during optimization to keep
# the lattice from collapsing into a degenerate matrix
_ANGLE_BOUNDS = (30.0, 150.0)


@dataclass(frozen=True)
class CellParametrization:
    """The free cell parameters allowed by a net's crystal system.

    Optimizing all six lattice parameters independently can break the
    symmetry the net declares (a cubic net drifting to a != b != c).
    This class maps a reduced free-parameter vector onto the full
    (a, b, c, alpha, beta, gamma), exposing exactly the degrees of
    freedom the crystal system allows: cubic 1, hexagonal/tetragonal/
    rhombohedral 2, orthorhombic 3, monoclinic 4 (unique angle freed),
    triclinic 6. Topologies without a spacegroup number fall back to
    free lengths with blueprint angles (the legacy behavior).

    Layer nets (is_2d) get a layer mode instead: c is the slab padding,
    not a bonded direction - no dummy pair crosses it, the objective is
    flat in c and Nelder-Mead would drift - so **c stays exactly
    frozen** at the blueprint value. The free parameters come from the
    plane-group lattice family: hexagonal/square 1 (a), rectangular 2
    (a, b), oblique 3 (a, b, gamma).
    """

    spacegroup_number: int | None
    blueprint_abc: tuple[float, float, float]
    blueprint_angles: tuple[float, float, float]
    is_2d: bool = False

    @functools.cached_property
    def system(self) -> str:
        """Crystal system name derived from the spacegroup number."""
        number = self.spacegroup_number or 0
        if self.is_2d:
            # the stored number is the plane-group number (1-17); an
            # out-of-range or missing number gets the most general
            # layer family, which still keeps c frozen
            family = (
                plane_groups.layer_system(number) if 1 <= number <= 17 else "oblique"
            )
            return f"layer_{family}"
        if number >= 195:
            return "cubic"
        if number >= 143:
            hexagonal_axes = np.allclose(
                self.blueprint_angles, (90.0, 90.0, 120.0), atol=1e-3
            )
            return "hexagonal" if hexagonal_axes else "rhombohedral"
        if number >= 75:
            return "tetragonal"
        if number >= 16:
            return "orthorhombic"
        if number >= 3:
            return "monoclinic"
        if number >= 1:
            return "triclinic"
        return "unknown"

    @functools.cached_property
    def _unique_angle_index(self) -> int:
        """Index of the monoclinic unique angle (largest |angle - 90|)."""
        deviations = np.abs(np.asarray(self.blueprint_angles) - 90.0)
        # default to beta when the blueprint is numerically orthogonal
        return int(np.argmax(deviations)) if deviations.max() > 1e-3 else 1

    @property
    def n_free(self) -> int:
        return {
            "cubic": 1,
            "hexagonal": 2,
            "rhombohedral": 2,
            "tetragonal": 2,
            "orthorhombic": 3,
            "monoclinic": 4,
            "triclinic": 6,
            "unknown": 3,
            "layer_hexagonal": 1,
            "layer_square": 1,
            "layer_rectangular": 2,
            "layer_oblique": 3,
        }[self.system]

    def expand(self, free: np.ndarray) -> tuple[float, ...]:
        """Full (a, b, c, alpha, beta, gamma) from the free parameters."""
        free = np.abs(np.asarray(free, dtype=float))
        system = self.system
        if system.startswith("layer_"):
            # c is the frozen slab padding, alpha = beta = 90 by
            # construction of the padded 2D lattice
            pad_c = self.blueprint_abc[2]
            if system == "layer_hexagonal":
                # 120 exactly: the plane-group operator tables assume
                # the standard hexagonal setting, and the blueprint
                # angle carries matrix-roundtrip noise
                (a,) = free
                return (a, a, pad_c, 90.0, 90.0, 120.0)
            if system == "layer_square":
                (a,) = free
                return (a, a, pad_c, 90.0, 90.0, 90.0)
            if system == "layer_rectangular":
                a, b = free
                return (a, b, pad_c, 90.0, 90.0, 90.0)
            a, b, gamma = free
            gamma = float(np.clip(gamma, *_ANGLE_BOUNDS))
            return (a, b, pad_c, 90.0, 90.0, gamma)
        if system == "cubic":
            (a,) = free
            return (a, a, a, 90.0, 90.0, 90.0)
        if system == "hexagonal":
            a, c = free
            return (a, a, c, 90.0, 90.0, 120.0)
        if system == "rhombohedral":
            a, alpha = free
            alpha = float(np.clip(alpha, *_ANGLE_BOUNDS))
            return (a, a, a, alpha, alpha, alpha)
        if system == "tetragonal":
            a, c = free
            return (a, a, c, 90.0, 90.0, 90.0)
        if system == "orthorhombic":
            a, b, c = free
            return (a, b, c, 90.0, 90.0, 90.0)
        if system == "monoclinic":
            a, b, c, unique = free
            angles = [90.0, 90.0, 90.0]
            angles[self._unique_angle_index] = float(np.clip(unique, *_ANGLE_BOUNDS))
            return (a, b, c, *angles)
        if system == "triclinic":
            a, b, c, alpha, beta, gamma = free
            alpha, beta, gamma = np.clip((alpha, beta, gamma), *_ANGLE_BOUNDS)
            return (a, b, c, float(alpha), float(beta), float(gamma))
        # unknown: free lengths, blueprint angles
        a, b, c = free
        return (a, b, c, *self.blueprint_angles)

    def seed(self, abc_guess: np.ndarray) -> np.ndarray:
        """Free parameters approximating an (a, b, c) length guess."""
        abc_guess = np.asarray(abc_guess, dtype=float)
        system = self.system
        if system in ("layer_hexagonal", "layer_square"):
            return np.array([abc_guess[:2].mean()])
        if system == "layer_rectangular":
            return abc_guess[:2].copy()
        if system == "layer_oblique":
            return np.array([*abc_guess[:2], self.blueprint_angles[2]])
        if system == "cubic":
            return np.array([abc_guess.mean()])
        if system in ("hexagonal", "tetragonal"):
            return np.array([abc_guess[:2].mean(), abc_guess[2]])
        if system == "rhombohedral":
            return np.array([abc_guess.mean(), self.blueprint_angles[0]])
        if system == "monoclinic":
            unique = self.blueprint_angles[self._unique_angle_index]
            return np.array([*abc_guess, unique])
        if system == "triclinic":
            return np.array([*abc_guess, *self.blueprint_angles])
        # orthorhombic and unknown
        return abc_guess.copy()


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
    # per arm: the anchor is the SBU atom bonded through that arm (the
    # dummy's nearest real atom); paired anchors bond in the output
    anchor_vecs: np.ndarray  # (n, 3) anchor positions minus center, cartesian
    anchor_radii: np.ndarray  # (n,) anchor covalent radii, Angstrom
    # arm assignment (perm) computed at the reference cell
    arm_for_target: np.ndarray

    def rotation_for(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rotation and target directions in the given cell matrix."""
        directions = _unit(self.frac_arms @ matrix)
        ordered_arms = self.arm_units[self.arm_for_target]
        return kabsch(ordered_arms, directions), directions

    def anchor_positions(self, matrix: np.ndarray) -> np.ndarray:
        """Cartesian SBU anchor positions (target order) in this cell."""
        rotation, _ = self.rotation_for(matrix)
        center = self.frac_center @ matrix
        ordered = self.anchor_vecs[self.arm_for_target]
        return np.asarray(center + ordered @ rotation.T)


@dataclass
class BuildPlan:
    """All geometry needed to optimize and realize one framework build.

    Created once per build by prepare_build(); the cell optimization
    objective then runs on plain arrays with no object construction.
    The cell is parametrized by the crystal system's free parameters
    only, so the optimizer cannot break the net's declared symmetry.
    """

    placements: list[SlotPlacement]
    # pairs of (placement index a, target index a, placement index b,
    # target index b, integer image offset) for tag-shared dummies
    pairs: list[tuple[int, int, int, int, np.ndarray]]
    blueprint_abc: np.ndarray  # (3,)
    angles: tuple[float, float, float]
    cell_param: CellParametrization

    @property
    def has_pairs(self) -> bool:
        return bool(self.pairs)

    @functools.cached_property
    def _blueprint_matrix(self) -> np.ndarray:
        a, b, c = self.blueprint_abc
        return Lattice.from_parameters(a, b, c, *self.angles).matrix

    def matrix_for(self, free: np.ndarray) -> np.ndarray:
        return Lattice.from_parameters(*self.cell_param.expand(free)).matrix

    def initial_parameters(self) -> np.ndarray:
        """Analytic starting parameters from bond-through distances.

        Each anchor pair wants its two slot centers separated by the
        center-to-anchor spans plus the bond length; the blueprint
        separation scales with the cell, so the ratio of sums is a
        near-optimal isotropic starting scale (exact for uninodal
        cubic nets). The resulting (a, b, c) guess is reduced to the
        crystal system's free parameters.
        """
        blueprint = self._blueprint_matrix
        required = 0.0
        current = 0.0
        for index_a, target_a, index_b, target_b, offset in self.pairs:
            pa = self.placements[index_a]
            pb = self.placements[index_b]
            span_a = np.linalg.norm(pa.anchor_vecs[pa.arm_for_target[target_a]])
            span_b = np.linalg.norm(pb.anchor_vecs[pb.arm_for_target[target_b]])
            required += (
                float(span_a)
                + float(span_b)
                + self._pair_bond_length(index_a, target_a, index_b, target_b)
            )
            separation = (pa.frac_center - pb.frac_center - offset) @ blueprint
            current += float(np.linalg.norm(separation))
        if not self.pairs or current < 1e-9:
            # no shared dummies: fall back to matching mean arm lengths
            arms = np.concatenate([p.arm_lengths for p in self.placements]).mean()
            slot_arms = np.concatenate(
                [
                    np.linalg.norm(p.frac_arms @ blueprint, axis=1)
                    for p in self.placements
                ]
            ).mean()
            abc_guess = self.blueprint_abc * (arms / slot_arms)
        else:
            abc_guess = self.blueprint_abc * (required / current)
        return self.cell_param.seed(abc_guess)

    def _pair_bond_length(
        self, index_a: int, target_a: int, index_b: int, target_b: int
    ) -> float:
        """Target bond length between a pair's two anchors."""
        pa = self.placements[index_a]
        pb = self.placements[index_b]
        return float(
            pa.anchor_radii[pa.arm_for_target[target_a]]
            + pb.anchor_radii[pb.arm_for_target[target_b]]
        )

    def residual(self, free: np.ndarray) -> float:
        """RMS deviation of paired anchor distances from their bond
        lengths at these parameters."""
        matrix = self.matrix_for(free)
        positions = [p.anchor_positions(matrix) for p in self.placements]
        total = 0.0
        for index_a, target_a, index_b, target_b, offset in self.pairs:
            delta = (
                positions[index_a][target_a]
                - positions[index_b][target_b]
                - offset @ matrix
            )
            gap = float(np.sqrt(delta @ delta)) - self._pair_bond_length(
                index_a, target_a, index_b, target_b
            )
            total += gap * gap
        return float(np.sqrt(total / len(self.pairs)))

    def finalize(
        self, free: np.ndarray
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
        lattice = Lattice.from_parameters(*self.cell_param.expand(free))
        matrix = lattice.matrix
        fragments: list[Fragment] = []
        slot_rmsds: dict[int, float] = {}
        for placement in self.placements:
            directions = _unit(placement.frac_arms @ matrix)
            rotation, perm, rmsd = match_directions(directions, placement.arm_units)
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


def _find_anchors(
    sbu: Fragment, dummy_indices: list[int], center: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Anchor vectors (relative to center) and covalent radii per arm.

    The anchor of an arm is the real atom bonded through it - the
    dummy's nearest non-dummy atom; the built structure bonds paired
    anchors together, so the cell objective targets one covalent bond
    length between them. A fragment with no real atoms (synthetic test
    fixtures) falls back to the dummies themselves with zero radius,
    which turns the pair objective back into dummy coincidence.
    """
    coords = np.asarray(sbu.atoms.cart_coords)
    real = [i for i, site in enumerate(sbu.atoms) if site.specie.symbol != "X"]
    if not real:
        return coords[dummy_indices] - center, np.zeros(len(dummy_indices))
    vecs = np.empty((len(dummy_indices), 3))
    radii = np.empty(len(dummy_indices))
    for k, dummy in enumerate(dummy_indices):
        distances = np.linalg.norm(coords[real] - coords[dummy], axis=1)
        nearest = real[int(np.argmin(distances))]
        vecs[k] = coords[nearest] - center
        symbol = sbu.atoms[nearest].specie.symbol
        # an element missing from the Cordero table keeps the library's
        # half-bond convention: the dummy sits half a bond from its anchor
        radii[k] = CovalentRadius.radius.get(symbol, float(distances.min()))
    return vecs, radii


def prepare_build(topology: Topology, mappings: dict[int, Fragment]) -> BuildPlan:
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
        slot_frac = blueprint.get_fractional_coords(slot.atoms.cart_coords[dummy_idx])
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
        anchor_vecs, anchor_radii = _find_anchors(sbu, sbu_dummy_idx, sbu_center)

        frac_arms = slot_frac - frac_center
        # reference assignment at the blueprint cell
        directions = _unit(frac_arms @ blueprint.matrix)
        _, perm, _ = match_directions(directions, arm_units)
        placement = SlotPlacement(
            slot_index=slot_index,
            frac_center=frac_center,
            frac_arms=frac_arms,
            tags=slot_tags,
            sbu=sbu,
            arm_units=arm_units,
            arm_lengths=arm_lengths,
            sbu_center=sbu_center,
            dummy_indices=sbu_dummy_idx,
            anchor_vecs=anchor_vecs,
            anchor_radii=anchor_radii,
            arm_for_target=perm,
        )
        index = len(placements)
        placements.append(placement)
        for target_index, (tag, frac) in enumerate(
            zip(slot_tags, slot_frac, strict=True)
        ):
            tag_sites.setdefault(tag, []).append((index, target_index, frac))

    pairs: list[tuple[int, int, int, int, np.ndarray]] = []
    unpaired = 0
    for tag, entries in sorted(tag_sites.items()):
        if len(entries) == 1:
            unpaired += 1
            continue
        if len(entries) > 2:
            logger.warning(
                f"Tag {tag} shared by {len(entries)} slots; using the first two."
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
    angles = (
        float(blueprint.angles[0]),
        float(blueprint.angles[1]),
        float(blueprint.angles[2]),
    )
    cell_param = CellParametrization(
        spacegroup_number=getattr(topology, "spacegroup_number", None),
        blueprint_abc=(float(abc[0]), float(abc[1]), float(abc[2])),
        blueprint_angles=angles,
        is_2d=getattr(topology, "is_2d", False),
    )
    return BuildPlan(
        placements=placements,
        pairs=pairs,
        blueprint_abc=abc,
        angles=angles,
        cell_param=cell_param,
    )
