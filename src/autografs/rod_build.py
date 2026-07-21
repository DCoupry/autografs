"""
Forward building of rod frameworks — straight and helical (rod Stage C).

``build_rod_framework`` places a harvested :class:`~autografs.rods.RodFragment`
onto one of a blueprint's axial slot runs (``autografs.net.axial_runs``)
and ditopic linkers onto the remaining slots — the forward counterpart
of rod deconstruction:

- the rod must carry its internal bond graph (``bonds``, recorded at
  harvest since Stage C1); helical rods are supported since the
  screw-aware bond recording (#158);
- the run must follow a single cell axis and contain exactly one
  PoE-bearing slot per period;
- every non-run slot must be 2-connected, all taking one ditopic
  linker species.

Multi-axis (woven) runs and mixed SBU mappings stay future work. A
general (non-180-degree) helical rod also needs a blueprint whose run
has matching spiralling lateral slots; ``pcu``-family runs suffice for
2_1 screws with ditopic linkers.

What is structurally different from the finite-SBU pipeline:

- **The rod pins a cell parameter** (pitfall 9). The run axis length
  is fixed to ``n_repeats x chemical repeat`` — never a free
  parameter. The remaining freedom reduces (v1) to one in-plane
  scale, optimized together with the rod's own two placement freedoms
  — rotation about the axis and axial phase (pitfall 10) — against
  covalent-length targets for the rod-to-linker anchor pairs.
- **Repeats are placed by the screw operation.** ``n_repeats`` =
  ``max(2, screw_order)`` copies are laid down the axis; copy *n* is
  rotated ``n x screw_angle`` about the axis (pure translation for a
  straight rod) and the linkers spiral with it, so the built
  framework is a genuine helix. Two is the minimum so a continuation
  bond is a distinct node pair, not a parallel edge.
- **Rod-to-linker bonds are explicit edges, not tag pairs.** The tag
  mechanism assumes one connection per atom; a rod anchor (the
  pillar's Zn, MOF-74's metal) carries several. Rod frameworks hold
  inter-unit bonds directly and all atoms carry tag 0; editing
  operations that rely on anchor tags refuse rods (Stage C4).
- **Self-closure is exact by construction** (pitfall 11): the cell
  pin makes continuation bonds screw images of the recorded internal
  bonds, and ``min_contact`` exempts them like any other graph edge
  (pitfall 15).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import networkx
import numpy as np
from pymatgen.analysis.local_env import CovalentRadius
from pymatgen.core.structure import Molecule
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial.transform import Rotation

from autografs.exceptions import AlignmentError, OverlapError
from autografs.fragment import Fragment
from autografs.framework import Framework
from autografs.net import SlotRun, axial_runs

if TYPE_CHECKING:
    from autografs.rods import RodFragment
    from autografs.topology import Topology

logger = logging.getLogger(__name__)

__all__ = ["build_rod_framework"]

DEFAULT_MAX_RMSD = 0.35
DEFAULT_BOND_TOLERANCE = 0.35

# arms within this angle cosine of the run axis are the axial dummies
# of the node slot (consumed by the rod's own continuation)
AXIAL_DOT = 0.7


def _covalent_radius(symbol: str, fallback: float = 0.75) -> float:
    return float(CovalentRadius.radius.get(symbol, fallback))


def _slot_geometry(slot, blueprint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(fractional center, cartesian arm vectors) of a blueprint slot."""
    inv = np.linalg.inv(blueprint)
    coords = np.asarray(slot.atoms.cart_coords)
    dummy_idx = list(slot.atoms.indices_from_symbol("X"))
    center = coords.mean(axis=0)
    arms = coords[dummy_idx] - center
    return center @ inv, arms


def _linker_anchor_rows(linker: Fragment) -> list[int]:
    """The linker atom bonded through each dummy (nearest real atom)."""
    coords = np.asarray(linker.atoms.cart_coords)
    dummy_idx = list(linker.atoms.indices_from_symbol("X"))
    real_idx = [i for i, site in enumerate(linker.atoms) if site.specie.symbol != "X"]
    if not real_idx:
        raise AlignmentError("Linker fragment has no real atoms.")
    anchors = []
    for dummy in dummy_idx:
        distances = np.linalg.norm(coords[real_idx] - coords[dummy], axis=1)
        anchors.append(real_idx[int(np.argmin(distances))])
    return anchors


class _RodBuild:
    """Geometry of one rod build, evaluated per (scale, theta, z0)."""

    def __init__(
        self,
        topology: Topology,
        rod: RodFragment,
        linker: Fragment,
        run: SlotRun,
    ) -> None:
        self.rod = rod
        self.linker = linker
        blueprint = np.asarray(topology.cell.matrix, dtype=float)
        self.blueprint = blueprint
        self.axis_index = int(np.argmax(np.abs(np.asarray(run.direction))))
        axis_row = blueprint[self.axis_index]
        self.axis_hat = axis_row / np.linalg.norm(axis_row)
        self.period = float(rod.repeat.repeat_length)
        # repeats placed along the axis: enough to close the screw (a
        # multiple of the screw order) and at least two, so a
        # continuation bond is always a distinct node pair rather than
        # a parallel edge. A straight rod (order 1) uses two.
        self.screw_rad = np.radians(rod.repeat.screw_angle)
        self.n_repeats = max(2, rod.repeat.screw_order)
        self.axis_length = self.n_repeats * self.period

        # transverse orthonormal pair the rod's local x/y embed along
        seed = np.array([1.0, 0.0, 0.0])
        if abs(float(seed @ self.axis_hat)) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        e1 = seed - (seed @ self.axis_hat) * self.axis_hat
        self.e1 = e1 / np.linalg.norm(e1)
        self.e2 = np.cross(self.axis_hat, self.e1)

        node_slots = [
            s
            for s in run.slots
            if len(topology.slots[s].atoms.indices_from_symbol("X")) > 2
        ]
        self.node_slot_index = node_slots[0]
        self.node_center_frac, node_arms = _slot_geometry(
            topology.slots[self.node_slot_index], blueprint
        )

        # lateral (non-axial) node arms: the directions the rod's own
        # arms must satisfy; used for the initial theta estimate
        units = node_arms / np.linalg.norm(node_arms, axis=1, keepdims=True)
        self.lateral_units = units[np.abs(units @ self.axis_hat) <= AXIAL_DOT]

        self.lateral_slot_indices = [
            s for s in range(len(topology.slots)) if s not in run.slots
        ]
        self.lateral_centers = []
        self.lateral_arm_units = []
        for s in self.lateral_slot_indices:
            center, arms = _slot_geometry(topology.slots[s], blueprint)
            self.lateral_centers.append(center)
            self.lateral_arm_units.append(
                arms / np.linalg.norm(arms, axis=1, keepdims=True)
            )

        # a ditopic linker is free to rotate about its own arm axis
        # without moving its anchors; symmetry-equivalent slots share
        # that rotation (one free phi per orbit) so the built framework
        # keeps its symmetry. Slots with no recorded orbit each get
        # their own phi.
        lateral_orbits = [
            topology.slots[s].equivalence_class for s in self.lateral_slot_indices
        ]
        self.lateral_orbits = [
            orbit if orbit is not None else -1 - i
            for i, orbit in enumerate(lateral_orbits)
        ]
        distinct = sorted(set(self.lateral_orbits))
        self.orbit_position = {orbit: k for k, orbit in enumerate(distinct)}
        self.n_orbits = len(distinct)
        # the placement index of each linker maps to its orbit's phi
        self.placement_orbit = [
            self.orbit_position[orbit]
            for orbit in self.lateral_orbits
            for _ in range(self.n_repeats)
        ]

        coords = np.asarray(linker.atoms.cart_coords)
        dummy_idx = list(linker.atoms.indices_from_symbol("X"))
        self.linker_dummy_span = (
            float(np.linalg.norm(coords[dummy_idx[0]] - coords[dummy_idx[1]]))
            if len(dummy_idx) == 2
            else 0.0
        )
        self.linker_dummy_center = coords[dummy_idx].mean(axis=0)
        self.linker_units = linker.arm_units
        # the linker's arm axis in its own frame: rotating the centered
        # coordinates about it before alignment swings the ring plane
        # while leaving the (on-axis) anchors - and the alignment
        # itself - unchanged
        self.linker_axis_local = self.linker_units[0]
        self.linker_anchor_rows = _linker_anchor_rows(linker)
        self.linker_real_rows = [
            i for i, site in enumerate(linker.atoms) if site.specie.symbol != "X"
        ]
        self.linker_coords = coords
        linker_symbols = [site.specie.symbol for site in linker.atoms]
        self._anchor_radius = {
            row: _covalent_radius(linker_symbols[row])
            for row in self.linker_anchor_rows
        }
        self._rod_radius = [_covalent_radius(symbol) for symbol in rod.repeat.symbols]
        self._arm_rows = [row for row, _ in rod.arms]
        self._arm_local = np.array([vec for _, vec in rod.arms])

    # ------------------------------------------------------------------
    # parameterized geometry
    # ------------------------------------------------------------------

    def cell(self, scale: float) -> np.ndarray:
        cell = self.blueprint * scale
        cell[self.axis_index] = self.axis_hat * self.axis_length
        return cell

    def _cell_one_period(self, scale: float) -> np.ndarray:
        cell = self.blueprint * scale
        cell[self.axis_index] = self.axis_hat * self.period
        return cell

    def _embed(self, local: np.ndarray, theta: float) -> np.ndarray:
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x = cos_t * local[..., 0] - sin_t * local[..., 1]
        y = sin_t * local[..., 0] + cos_t * local[..., 1]
        return (
            np.outer(x, self.e1)
            + np.outer(y, self.e2)
            + np.outer(local[..., 2], self.axis_hat)
        )

    def evaluate(
        self,
        scale: float,
        theta: float,
        z0: float,
        phi: np.ndarray | None = None,
    ) -> dict:
        """Place everything for one parameter set.

        Returns a dict with rod positions, arm tips, linker
        placements, the Hungarian arm-anchor matching, and the
        per-bond residuals |distance - covalent target|. ``phi`` (one
        angle per lateral orbit) rotates each linker about its own arm
        axis to relieve ring-plane clashes without moving its anchors.
        """
        n_atoms = len(self.rod.positions)
        cell_p = self._cell_one_period(scale)
        line = self.node_center_frac @ cell_p
        line_perp = line - (line @ self.axis_hat) * self.axis_hat

        def screw(points: np.ndarray, n: int) -> np.ndarray:
            """The nth screw image about the rod axis line: rotate the
            perpendicular part of (points - line_perp) by n*screw, keep
            the axial part, then translate n*period along the axis. For
            a straight rod (screw 0) this is pure translation."""
            rel = points - line_perp
            axial = (rel @ self.axis_hat)[:, None] * self.axis_hat
            perp = rel - axial
            if self.screw_rad:
                perp = Rotation.from_rotvec(n * self.screw_rad * self.axis_hat).apply(
                    perp
                )
            return np.asarray(
                line_perp + axial + perp + n * self.period * self.axis_hat
            )

        # repeat 0: rod atoms in place; arm vectors as free vectors
        # (rod atom -> cut midpoint), rotated with the screw per repeat
        rod0 = self._embed(self.rod.positions, theta) + (line_perp + z0 * self.axis_hat)
        arm_vec0 = (
            self._embed(self._arm_local, theta)
            if len(self._arm_local)
            else np.empty((0, 3))
        )
        rod_positions = np.empty((self.n_repeats * n_atoms, 3))
        arms: list[tuple[int, np.ndarray]] = []
        for n in range(self.n_repeats):
            start = n * n_atoms
            rod_positions[start : start + n_atoms] = screw(rod0, n)
            arm_vec_n = (
                Rotation.from_rotvec(n * self.screw_rad * self.axis_hat).apply(arm_vec0)
                if self.screw_rad and len(arm_vec0)
                else arm_vec0
            )
            for k, row in enumerate(self._arm_rows):
                arms.append((start + row, rod_positions[start + row] + arm_vec_n[k]))

        placements = []
        for slot_pos, (center_frac, slot_units) in enumerate(
            zip(self.lateral_centers, self.lateral_arm_units, strict=True)
        ):
            centered = self.linker_coords - self.linker_dummy_center
            if phi is not None:
                # pre-rotate about the arm axis: it fixes the arm
                # directions, so the alignment below is unchanged, but
                # the ring plane swings
                angle = float(phi[self.orbit_position[self.lateral_orbits[slot_pos]]])
                centered = Rotation.from_rotvec(angle * self.linker_axis_local).apply(
                    centered
                )
            rotation, rssd = Rotation.align_vectors(slot_units, self.linker_units)
            rmsd = float(rssd) / np.sqrt(len(slot_units))
            linker0 = rotation.apply(centered) + center_frac @ cell_p
            for n in range(self.n_repeats):
                placed = screw(linker0, n)
                placements.append(
                    {
                        "coords": placed,
                        "anchors": [
                            (row, placed[row]) for row in self.linker_anchor_rows
                        ],
                        "rmsd": rmsd,
                    }
                )

        ends = []
        for p_index, placement in enumerate(placements):
            for row, position in placement["anchors"]:
                ends.append((p_index, row, position))
        if not arms or not ends:
            raise AlignmentError("Rod build has no connectable arms.")

        cell = self.cell(scale)
        inv = np.linalg.inv(cell)

        def min_image_distance(a: np.ndarray, b: np.ndarray) -> float:
            delta = (b - a) @ inv
            delta -= np.round(delta)
            return float(np.linalg.norm(delta @ cell))

        cost = np.empty((len(arms), len(ends)))
        for i, (_, tip) in enumerate(arms):
            for j, (_, _, position) in enumerate(ends):
                cost[i, j] = min_image_distance(tip, position)
        rows, cols = linear_sum_assignment(cost)

        residuals = np.empty(len(rows))
        matching = []
        for k, (i, j) in enumerate(zip(rows, cols, strict=True)):
            atom_index, _ = arms[i]
            p_index, anchor_row, anchor_pos = ends[j]
            target = (
                self._rod_radius[atom_index % n_atoms] + self._anchor_radius[anchor_row]
            )
            distance = min_image_distance(rod_positions[atom_index], anchor_pos)
            residuals[k] = abs(distance - target)
            matching.append((atom_index, p_index, anchor_row))
        return {
            "cell": cell,
            "rod_positions": rod_positions,
            "arms": arms,
            "placements": placements,
            "matching": matching,
            "residuals": residuals,
        }

    def objective(self, params: np.ndarray) -> float:
        scale, theta, z0 = params
        if scale <= 0.05:
            return 1e6
        residuals = self.evaluate(float(scale), float(theta), float(z0))["residuals"]
        return float(np.sqrt((residuals**2).mean()))

    def initial_guess(self) -> np.ndarray:
        """(scale, theta, z0) heuristics good enough for Nelder-Mead."""
        # scale: lateral node-to-node distance should fit two rod arms
        # plus the linker span between its dummies
        center_cart = self.node_center_frac @ self.blueprint
        lateral_center = self.lateral_centers[0] @ self.blueprint
        blueprint_half = np.linalg.norm(
            lateral_center
            - center_cart
            - ((lateral_center - center_cart) @ self.axis_hat) * self.axis_hat
        )
        arm_length = (
            float(np.linalg.norm(self._arm_local, axis=1).mean())
            if len(self._arm_local)
            else 1.0
        )
        wanted = 2.0 * arm_length + self.linker_dummy_span
        scale0 = (
            float(wanted / (2.0 * blueprint_half)) if blueprint_half > 1e-9 else 1.0
        )
        # z0: put the mean arm-carrying atom at the node center height
        cell_p = self._cell_one_period(scale0)
        z_node = float((self.node_center_frac @ cell_p) @ self.axis_hat)
        z_arms = (
            float(np.mean([self.rod.positions[row][2] for row in self._arm_rows]))
            if self._arm_rows
            else 0.0
        )
        return np.array([scale0, 0.0, z_node - z_arms])

    def min_inter_unit_contact(self, placed: dict) -> float:
        """Closest approach between atoms of different building units.

        Rod atoms and each linker's real atoms are separate units; a
        bond within a unit is rigid under the linker rotation, so only
        inter-unit distances change and only they can clash. Matched
        rod-linker anchor bonds are ~one covalent length and never the
        binding constraint, so they need no special exemption.
        """
        cell = placed["cell"]
        inv = np.linalg.inv(cell)
        n_atoms = len(self.rod.positions)
        coords = [placed["rod_positions"]]
        labels = [np.full(self.n_repeats * n_atoms, -1)]
        for p_index, placement in enumerate(placed["placements"]):
            reals = placement["coords"][self.linker_real_rows]
            coords.append(reals)
            labels.append(np.full(len(reals), p_index))
        points = np.vstack(coords)
        unit = np.concatenate(labels)
        # minimum-image pairwise distances between different units
        delta = points[:, None, :] - points[None, :, :]
        frac = delta @ inv
        frac -= np.round(frac)
        dist = np.linalg.norm(frac @ cell, axis=2)
        cross_unit = unit[:, None] != unit[None, :]
        return float(dist[cross_unit].min())


def _typed_repeat_molgraph(build: _RodBuild, result: dict):
    """UFF-typed molgraph of one placed rod repeat.

    The repeat is typed like any fragment: its molecule plus X dummies
    at the arm tips and at the axial continuation midpoints, through
    ``fragment_to_molgraph`` (X -> H for typing, dummies stripped).
    """
    from autografs.utils import fragment_to_molgraph

    n_atoms = len(build.rod.positions)
    positions = result["rod_positions"][:n_atoms]
    species = list(build.rod.repeat.symbols)
    coords = [positions[i] for i in range(n_atoms)]
    for _atom_index, tip in result["arms"][: len(build.rod.arms)]:
        species.append("X")
        coords.append(tip)
    full = result["rod_positions"]
    for a, b, m in build.rod.bonds:
        if m != 0:
            partner = (0 + m) % build.n_repeats * n_atoms + b
            species.append("X")
            coords.append(0.5 * (full[a] + full[partner]))
    molecule = Molecule(species, coords, site_properties={"tags": [0] * len(species)})
    return fragment_to_molgraph(Fragment(atoms=molecule, name=build.rod.name))


def build_rod_framework(
    topology: Topology,
    rod: RodFragment,
    linker: Fragment,
    run: SlotRun | None = None,
    max_rmsd: float = DEFAULT_MAX_RMSD,
    min_distance: float | None = 1.0,
    bond_tolerance: float = DEFAULT_BOND_TOLERANCE,
    verbose: bool = False,
) -> Framework:
    """Build a straight rod framework from a rod fragment and a linker.

    Parameters
    ----------
    topology : Topology
        The blueprint. Must have at least one straight axial slot run
        (``autografs.net.axial_runs``).
    rod : RodFragment
        A harvested rod (``HarvestResult.rods`` / ``load_rods``);
        screwless (``screw_order == 1``) and carrying its internal
        bond graph.
    linker : Fragment
        One ditopic linker species, placed on every non-run slot.
    run : SlotRun or None, optional
        Which run the rod occupies; the first detected run when None
        (symmetry-equivalent runs give equivalent frameworks).
    max_rmsd : float, optional
        Directional gate on each linker's arm alignment.
    min_distance : float or None, optional
        Post-build closest-contact gate (continuation and rod-linker
        bonds are graph edges, hence exempt).
    bond_tolerance : float, optional
        Closure gate: largest allowed deviation of an optimized
        rod-linker bond from its covalent-length target, in Angstrom.
    verbose : bool, optional
        Log the optimized cell and residuals.

    Returns
    -------
    Framework
        The built framework: two chemical repeats of the rod along the
        pinned axis plus one linker per lateral slot per repeat.

    Raises
    ------
    AlignmentError
        For out-of-scope inputs (screw rods, missing bonds, no run,
        non-ditopic linkers) and failed gates.
    OverlapError
        When the built structure violates ``min_distance``.
    """
    from autografs.utils import find_element_cutoffs, load_uff_lib

    if not rod.bonds:
        raise AlignmentError(
            f"Rod {rod.name!r} carries no internal bond graph; re-harvest "
            "with a current AuToGraFS (RodFragment.bonds is required)."
        )
    if not rod.arms:
        raise AlignmentError(f"Rod {rod.name!r} has no connection arms.")
    if len(list(linker.atoms.indices_from_symbol("X"))) != 2:
        raise AlignmentError("Rod builds take one ditopic linker species.")

    runs = axial_runs(topology)
    if run is None:
        if not runs:
            raise AlignmentError(
                f"Topology {topology.name!r} has no straight axial slot "
                "run; rod building needs one."
            )
        run = runs[0]
    direction = np.asarray(run.direction)
    if sorted(np.abs(direction).tolist()) != [0, 0, 1]:
        raise AlignmentError(
            "Rod building supports runs along a single cell axis for "
            f"now (got direction {run.direction})."
        )
    node_slots = [
        s
        for s in run.slots
        if len(topology.slots[s].atoms.indices_from_symbol("X")) > 2
    ]
    if len(node_slots) != 1:
        raise AlignmentError(
            "Rod building supports runs with one PoE-bearing slot per "
            f"period for now (run has {len(node_slots)})."
        )
    lateral_slots = [s for s in range(len(topology.slots)) if s not in run.slots]
    for s in lateral_slots:
        if len(topology.slots[s].atoms.indices_from_symbol("X")) != 2:
            raise AlignmentError(
                f"Every non-run slot must be 2-connected for now (slot {s} is not)."
            )
    node_arm_count = len(topology.slots[node_slots[0]].atoms.indices_from_symbol("X"))
    if len(rod.arms) != node_arm_count - 2:
        raise AlignmentError(
            f"Rod {rod.name!r} carries {len(rod.arms)} arms per repeat "
            f"but the run's node slot expects {node_arm_count - 2} "
            "lateral connections."
        )

    build = _RodBuild(topology, rod, linker, run)

    # optimize: coarse rotation grid, refine the best with Nelder-Mead
    guess = build.initial_guess()
    best = None
    for angle in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
        value = build.objective(np.array([guess[0], angle, guess[2]]))
        if best is None or value < best[1]:
            best = (float(angle), value)
    assert best is not None
    start = np.array([guess[0], best[0], guess[2]])
    result = minimize(
        build.objective,
        start,
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000},
    )
    scale, theta, z0 = (float(x) for x in result.x)

    # relieve ditopic-linker ring-plane clashes: rotating each linker
    # about its own arm axis keeps the anchors (hence the bonds) fixed,
    # so maximize the minimum inter-unit contact over one phi per
    # lateral orbit. A ring has pi symmetry, so the search spans
    # [0, pi); grid then refine.
    def relief(phi: np.ndarray) -> float:
        return -build.min_inter_unit_contact(build.evaluate(scale, theta, z0, phi=phi))

    best_phi = np.zeros(build.n_orbits)
    if build.n_orbits == 1:
        grid = np.linspace(0.0, np.pi, 24, endpoint=False)
        best_phi = np.array([min(grid, key=lambda a: relief(np.array([a])))])
    elif build.n_orbits > 1:
        for _ in range(4):  # coordinate ascent over the orbits
            for k in range(build.n_orbits):
                grid = np.linspace(0.0, np.pi, 16, endpoint=False)

                def score(a: float, k: int = k) -> float:
                    trial = best_phi.copy()
                    trial[k] = a
                    return relief(trial)

                best_phi[k] = min(grid, key=score)
    refined = minimize(
        relief,
        best_phi,
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 400},
    )
    if -refined.fun >= -relief(best_phi):
        best_phi = refined.x
    placed = build.evaluate(scale, theta, z0, phi=best_phi)

    worst_bond = float(placed["residuals"].max())
    if worst_bond > bond_tolerance:
        raise AlignmentError(
            f"Rod build on {topology.name!r} does not close: worst "
            f"rod-linker bond deviates {worst_bond:.2f} A from its "
            f"covalent target (bond_tolerance={bond_tolerance})."
        )
    worst_rmsd = max(p["rmsd"] for p in placed["placements"])
    if worst_rmsd > max_rmsd:
        raise AlignmentError(
            f"Linker arm alignment RMSD {worst_rmsd:.3f} exceeds "
            f"max_rmsd={max_rmsd} on topology {topology.name!r}."
        )
    if verbose:
        a, b, c = np.linalg.norm(placed["cell"], axis=1)
        logger.info(
            f"\t[x] Rod build cell a={a:.2f} b={b:.2f} c={c:.2f}; worst "
            f"bond residual {worst_bond:.3f} A."
        )

    graph = _assemble_graph(build, placed, find_element_cutoffs, load_uff_lib)
    framework = Framework(graph, name=f"{topology.name}_{rod.name}")
    if min_distance is not None:
        contact = framework.min_contact(cutoff=min_distance)
        if contact < min_distance:
            raise OverlapError(
                f"Closest non-bonded contact is {contact:.2f} A, below "
                f"min_distance={min_distance:.2f} A, on rod build "
                f"{framework.name!r}."
            )
    return framework


def _assemble_graph(build: _RodBuild, result: dict, find_cutoffs, load_uff):
    """The built framework's bond graph, from placed geometry.

    Mirrors ``utils.fragments_to_networkx`` conventions (node attrs,
    intra-unit bonds with orders) with two rod specifics: rod internal
    bonds come from the recorded ``RodFragment.bonds`` (continuations
    wrap between repeats), and rod-linker bonds are explicit edges
    from the optimizer's matching. All tags are 0.
    """
    from pymatgen.core.bonds import get_bond_order

    # rod_build marks the framework so tag/anchor-based editing ops
    # (defects, flip, rotate, functionalize) refuse it - see editing._reject_rod
    graph = networkx.Graph(cell=result["cell"], rod_build=True)
    n_atoms = len(build.rod.positions)

    # rod atoms: typed once on the first repeat, replicated
    rod_molgraph = _typed_repeat_molgraph(build, result)
    rod_types = [rod_molgraph.molecule[i].properties["ufftype"] for i in range(n_atoms)]
    for n in range(build.n_repeats):
        start = n * n_atoms
        for i in range(n_atoms):
            graph.add_node(
                start + i,
                symbol=build.rod.repeat.symbols[i],
                coord=result["rod_positions"][start + i],
                tag=0,
                ufftype=rod_types[i],
                slot=n,
                sbu=build.rod.name,
            )
    for n in range(build.n_repeats):
        start = n * n_atoms
        for a, b, m in build.rod.bonds:
            partner_start = ((n + m) % build.n_repeats) * n_atoms
            graph.add_edge(start + a, partner_start + b, bond_order=1.0)

    # linkers: one molgraph per placement for types, bonds and orders
    offset = build.n_repeats * n_atoms
    anchor_nodes: list[dict[int, int]] = []
    for p_index in range(len(result["placements"])):
        molgraph = _placed_linker_molgraph(build, p_index, result)
        molecule = molgraph.molecule
        species = [s.specie.symbol for s in molecule]
        bond_lengths = find_cutoffs(*load_uff(molecule))
        row_to_node: dict[int, int] = {}
        for i in range(len(molecule)):
            graph.add_node(
                offset + i,
                symbol=species[i],
                coord=np.asarray(molecule[i].coords),
                tag=0,
                ufftype=molecule[i].properties["ufftype"],
                slot=build.n_repeats + p_index,
                sbu=build.linker.name,
            )
            row_to_node[i] = offset + i
        for i in range(len(molecule)):
            for site in molgraph.get_connected_sites(i):
                j, distance = site.index, site.dist
                uff_bl = bond_lengths[(species[i], species[j])]
                order = get_bond_order(
                    species[i], species[j], distance, tol=0.2, default_bl=uff_bl
                )
                graph.add_edge(offset + i, offset + j, bond_order=order)
        anchor_nodes.append(row_to_node)
        offset += len(molecule)

    # rod-linker bonds from the optimizer's matching; the anchor row
    # indexes the original linker (with dummies) - map it to the
    # dummy-stripped molgraph row
    for atom_index, p_index, anchor_row in result["matching"]:
        stripped_row = _stripped_row(build, anchor_row)
        graph.add_edge(
            atom_index,
            anchor_nodes[p_index][stripped_row],
            bond_order=1.0,
        )
    return graph


def _placed_linker_molgraph(build: _RodBuild, p_index: int, result: dict):
    """UFF-typed molgraph of one placed linker (dummies included then
    stripped by fragment_to_molgraph)."""
    from autografs.utils import fragment_to_molgraph

    placement = result["placements"][p_index]
    # the placement transformed every atom of the original linker
    # (real and dummy) together; "coords" holds them all
    molecule = Molecule(
        [site.specie.symbol for site in build.linker.atoms],
        placement["coords"],
        site_properties={"tags": [0] * len(build.linker.atoms)},
    )
    return fragment_to_molgraph(Fragment(atoms=molecule, name=build.linker.name))


def _stripped_row(build: _RodBuild, anchor_row: int) -> int:
    """Row of an anchor atom after dummies are stripped from the
    linker molecule (fragment_to_molgraph removes X sites)."""
    dummy_rows = set(build.linker.atoms.indices_from_symbol("X"))
    return anchor_row - sum(1 for d in dummy_rows if d < anchor_row)
