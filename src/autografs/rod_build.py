"""
Forward building of rod frameworks — straight and helical (rod Stage C).

``build_rod_framework`` places a harvested :class:`~autografs.rods.RodFragment`
onto one of a blueprint's axial slot runs (``autografs.net.axial_runs``)
and finite SBUs onto the remaining slots — the forward counterpart
of rod deconstruction:

- the rod must carry its internal bond graph (``bonds``, recorded at
  harvest since Stage C1); helical rods are supported since the
  screw-aware bond recording (#158);
- the run follows a single cell axis: a *straight* run
  (``autografs.net.axial_runs``) has one PoE-bearing slot per period,
  supercelled ``n_repeats`` up; a *helical* run
  (``autografs.net.helical_runs``, a MOF-74 / etb-class screw) has
  ``screw_order`` node slots that the rod's chemical repeats fill 1:1
  by the screw operation, no supercell (#158);
- the non-run (*lateral*) slots take a **per-slot mapping** of finite
  SBUs of any connectivity — one ditopic species everywhere is just
  the common case, and a single Fragment still means exactly that
  (#168).

A *cross-linked multi-rod* net (etb / MOF-74 proper) passes **several**
helical runs: one rod is placed on each of the net's interleaved
helices (the enantiomer on the opposite-handedness runs — etb is
centrosymmetric, three helices of each hand), and the linkers
bridge *between* different rods rather than within one (#158).

A *woven* net (bmn) spirals along several cell axes at once. Runs are
grouped into **axis families**, each pinning its own cell parameter and
carrying its own rotation and axial phase (#168); a fully woven packing
pins all three axes, which determines the cell outright. One rod pins
every axis it runs along to the same length, so only runs of equal
period can be woven together. Runs along a lattice *diagonal* stay out
of scope — they would pin a combination of cell parameters.

What is structurally different from the finite-SBU pipeline:

- **The rod pins a cell parameter** (pitfall 9). The run axis length
  is fixed to ``n_repeats x chemical repeat`` — never a free
  parameter. The remaining freedom reduces (v1) to one in-plane
  scale, optimized together with the rod's own two placement freedoms
  — rotation about the axis and axial phase (pitfall 10) — against
  covalent-length targets for the bonded anchor pairs.
- **Repeats are placed by the screw operation.** ``n_repeats`` =
  ``max(2, screw_order)`` copies are laid down the axis; copy *n* is
  rotated ``n x screw_angle`` about the axis (pure translation for a
  straight rod) and the linkers spiral with it, so the built
  framework is a genuine helix. Two is the minimum so a continuation
  bond is a distinct node pair, not a parallel edge.
- **Inter-unit bonds are explicit edges, not tag pairs.** The tag
  mechanism assumes one connection per atom; a rod anchor (the
  pillar's Zn, MOF-74's metal) carries several. Rod frameworks hold
  inter-unit bonds directly and all atoms carry tag 0; editing
  operations that rely on anchor tags refuse rods (Stage C4).
- **Every connection point is paired against every other** (#168).
  Polytopic lateral slots bond to *each other*, not only to the rod
  (no library net has polytopic laterals that touch run nodes
  alone), so the optimizer pairs the whole port set — rod arm tips
  and every SBU dummy — by shortest min-image separation rather than
  matching rod arms onto linker anchors bipartitely. Greedy nearest
  pairing, not a general min-weight matching: the blueprint puts the
  right partners nearest by a wide margin, and it stays cheap enough
  to run inside the optimizer loop.
- **Self-closure is exact by construction** (pitfall 11): the cell
  pin makes continuation bonds screw images of the recorded internal
  bonds, and ``min_contact`` exempts them like any other graph edge
  (pitfall 15).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx
import numpy as np
from pymatgen.analysis.local_env import CovalentRadius
from pymatgen.core.structure import Molecule
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from autografs.alignment import kabsch, match_directions
from autografs.exceptions import AlignmentError, OverlapError
from autografs.fragment import Fragment
from autografs.framework import Framework
from autografs.net import HelicalRun, SlotRun, axial_runs, helical_runs

if TYPE_CHECKING:
    from autografs.rods import RodFragment
    from autografs.topology import Topology

logger = logging.getLogger(__name__)

__all__ = ["build_rod_framework"]

DEFAULT_MAX_RMSD = 0.35
DEFAULT_BOND_TOLERANCE = 0.35

# arms this close to antiparallel share one rotation axis, so a linker
# can be spun about it without moving its anchors (the relief pass)
RELIEF_COLLINEAR_DOT = 0.98

# port kinds, as recorded in a bond's endpoint tuples
_ROD_PORT = 0
_LINKER_PORT = 1


def _covalent_radius(symbol: str, fallback: float = 0.75) -> float:
    return float(CovalentRadius.radius.get(symbol, fallback))


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms < 1e-9):
        raise AlignmentError("Degenerate (zero-length) arm vector.")
    return np.asarray(vectors / norms)


def _slot_geometry(slot, blueprint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(fractional center, fractional arm vectors) of a blueprint slot.

    Arms are kept fractional so their *directions* can be recomputed in
    whatever cell the optimizer is trying: the rod pins one axis while
    the others scale, so a slot's arm directions are not the blueprint's
    (this matters for polytopic slots, where the arm assignment and the
    alignment RMSD both depend on them).
    """
    inv = np.linalg.inv(blueprint)
    coords = np.asarray(slot.atoms.cart_coords)
    dummy_idx = list(slot.atoms.indices_from_symbol("X"))
    center = coords.mean(axis=0)
    arms = coords[dummy_idx] - center
    return center @ inv, arms @ inv


def _unwrapped_node_frac(topology: Topology, run: HelicalRun) -> np.ndarray:
    """Fractional coord of a helical run's first node, unwrapped along
    the run's own walk so it lies on the same cylinder as
    ``run.axis_point``.

    ``helical_runs`` derives ``axis_point`` from the *unwrapped* walk
    (nodes accumulated step by step from ``run.slots[0]``); a node's
    home-cell image can be a perpendicular-wrapped copy sitting well off
    that cylinder, which would place the rod at the wrong radius and
    azimuth. Walking the run the same way puts the reference node back on
    the axis line's cylinder.
    """
    from autografs.net import _topology_slot_images, topology_quotient_edges

    cell = np.asarray(topology.cell.matrix, dtype=float)
    images = _topology_slot_images(topology)
    centers = np.array(
        [np.asarray(s.atoms.cart_coords).mean(axis=0) for s in topology.slots]
    )
    wrapped = centers - np.array([images[i] for i in range(len(centers))]) @ cell
    steps: dict[int, list[tuple[int, np.ndarray]]] = {
        i: [] for i in range(len(centers))
    }
    for (a, b, voltage), _count in topology_quotient_edges(topology).items():
        for u, w, sign in ((a, b, 1), (b, a, -1)):
            disp = (
                wrapped[w]
                + (sign * np.asarray(voltage, dtype=float)) @ cell
                - wrapped[u]
            )
            if np.linalg.norm(disp) > 1e-9:
                steps[u].append((w, disp))
    pos = {run.slots[0]: wrapped[run.slots[0]]}
    cur = run.slots[0]
    for nxt in run.slots[1:]:
        disp = next(d for w, d in steps[cur] if w == nxt)
        pos[nxt] = pos[cur] + disp
        cur = nxt
    node0 = next(
        s
        for s in run.slots
        if len(topology.slots[s].atoms.indices_from_symbol("X")) > 2
    )
    return np.asarray(pos[node0] @ np.linalg.inv(cell))


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


@dataclass
class _Species:
    """Geometry of one lateral SBU species, precomputed once.

    Several slots usually share a species (a whole orbit takes the same
    SBU), and the placement loop runs it thousands of times inside the
    optimizer, so everything that does not depend on the cell lives
    here.
    """

    fragment: Fragment
    symbols: list[str]
    coords: np.ndarray  # (n, 3) cartesian, as given
    dummy_center: np.ndarray  # (3,) dummy centroid, the placement origin
    dummy_rows: list[int]
    real_rows: list[int]
    arm_units: np.ndarray  # (m, 3) unit arm vectors from the centroid
    anchor_rows: list[int]  # per dummy: the real atom bonded through it
    anchor_radii: np.ndarray  # (m,) covalent radius of each anchor
    stripped_rows: dict[int, int]  # row -> row once dummies are removed
    # the axis a ditopic linker can be spun about without moving its
    # anchors (the relief pass); None whenever no such axis exists
    relief_axis: np.ndarray | None

    @property
    def n_arms(self) -> int:
        return len(self.dummy_rows)


def _species_of(linker: Fragment) -> _Species:
    """Precompute one lateral SBU species."""
    coords = np.asarray(linker.atoms.cart_coords, dtype=float)
    dummy_rows = list(linker.atoms.indices_from_symbol("X"))
    if not dummy_rows:
        raise AlignmentError(f"Linker {linker.name!r} has no connection points.")
    real_rows = [i for i, site in enumerate(linker.atoms) if site.specie.symbol != "X"]
    anchor_rows = _linker_anchor_rows(linker)
    symbols = [site.specie.symbol for site in linker.atoms]
    center = coords[dummy_rows].mean(axis=0)
    arm_units = _unit_rows(coords[dummy_rows] - center)
    relief_axis = None
    if len(dummy_rows) == 2 and -arm_units[0] @ arm_units[1] > RELIEF_COLLINEAR_DOT:
        # a straight ditopic linker: its anchors sit on the arm axis, so
        # spinning about it leaves every bond closed
        relief_axis = arm_units[0]
    dummy_set = set(dummy_rows)
    stripped = {
        row: row - sum(1 for d in dummy_set if d < row) for row in range(len(symbols))
    }
    return _Species(
        fragment=linker,
        symbols=symbols,
        coords=coords,
        dummy_center=center,
        dummy_rows=dummy_rows,
        real_rows=real_rows,
        arm_units=arm_units,
        anchor_rows=anchor_rows,
        anchor_radii=np.array([_covalent_radius(symbols[r]) for r in anchor_rows]),
        stripped_rows=stripped,
        relief_axis=relief_axis,
    )


def _pair_ports(
    tips: np.ndarray,
    port_slot: np.ndarray,
    budget: dict[tuple[int, int], int],
    cell: np.ndarray,
    inv: np.ndarray,
) -> np.ndarray:
    """Pair every connection point with its nearest legitimate partner.

    Ports are the build's half-bonds - rod arm tips and SBU dummies -
    and each pairs with exactly one other. *Which* slots may bond is the
    blueprint's business, not geometry's: ``budget`` carries one entry
    per bonded slot pair of the blueprint's rod form, with the number of
    bonds it expects. Geometry only decides which individual ports (and
    hence which periodic images) realize them, greedily by ascending
    min-image separation.

    Letting geometry choose the partners too - the obvious "pair whatever
    is nearest" - is not good enough: the objective landscape over the
    cell scale is then multi-modal, and its *global* minimum can be a
    spurious pairing at a badly inflated cell, where unrelated ports
    happen to meet. Spending the blueprint's budget instead makes the
    built edge multiset equal the blueprint's by construction, so those
    minima do not exist and a wrong cell simply reads as a large residual.

    The budgets also make the matching complete for free: every slot's
    budgets sum to its port count, so ports run out exactly when budgets
    do.

    Parameters
    ----------
    tips : np.ndarray
        (p, 3) cartesian port positions.
    port_slot : np.ndarray
        (p,) blueprint slot each port belongs to.
    budget : dict
        (slot_a, slot_b) with slot_a <= slot_b -> number of bonds.
    cell, inv : np.ndarray
        The (3, 3) cell matrix and its inverse.

    Returns
    -------
    np.ndarray
        (p / 2, 2) array of paired port indices.

    Raises
    ------
    AlignmentError
        If the port count is odd, or the budgets and the ports do not
        add up (the rod and the linkers do not cover the blueprint's
        connectivity).
    """
    n_ports = len(tips)
    if n_ports % 2:
        raise AlignmentError(
            f"Rod build has {n_ports} connection points: an odd number "
            "cannot pair up. Check the rod's arms against the run's node "
            "slots and each linker against its slot."
        )
    if sum(budget.values()) * 2 != n_ports:
        raise AlignmentError(
            f"The blueprint expects {sum(budget.values())} inter-unit bonds "
            f"but the rod and the linkers bring {n_ports} connection points "
            f"({n_ports // 2} bonds); they do not cover its connectivity."
        )
    rows, cols = np.triu_indices(n_ports, k=1)
    slot_a = np.minimum(port_slot[rows], port_slot[cols])
    slot_b = np.maximum(port_slot[rows], port_slot[cols])
    allowed = np.array(
        [(int(a), int(b)) in budget for a, b in zip(slot_a, slot_b, strict=True)]
    )
    rows, cols = rows[allowed], cols[allowed]
    delta = tips[rows] - tips[cols]
    frac = delta @ inv
    frac -= np.round(frac)
    distance = np.linalg.norm(frac @ cell, axis=1)
    order = np.argsort(distance, kind="stable")
    free = np.ones(n_ports, dtype=bool)
    remaining = dict(budget)
    pairs: list[tuple[int, int]] = []
    for index in order:
        i, j = int(rows[index]), int(cols[index])
        if not (free[i] and free[j]):
            continue
        key = (
            int(min(port_slot[i], port_slot[j])),
            int(max(port_slot[i], port_slot[j])),
        )
        if not remaining[key]:
            continue
        remaining[key] -= 1
        free[i] = free[j] = False
        pairs.append((i, j))
        if len(pairs) * 2 == n_ports:
            break
    if len(pairs) * 2 != n_ports:  # pragma: no cover - the counts rule it out
        raise AlignmentError(
            f"{int(free.sum())} of {n_ports} connection points found no "
            "partner allowed by the blueprint."
        )
    return np.asarray(pairs)


@dataclass
class _Lateral:
    """One lateral (non-run) blueprint slot and the SBU filling it."""

    slot_index: int
    center_frac: np.ndarray  # (3,)
    frac_arms: np.ndarray  # (m, 3)
    species: int  # index into _RodBuild.species
    orbit: int  # crystallographic orbit, for the shared relief angle
    perm: np.ndarray  # (m,) SBU arm assigned to each slot dummy


class _RodBuild:
    """Geometry of one rod build, evaluated per (scale, theta, z0)."""

    def __init__(
        self,
        topology: Topology,
        rod: RodFragment,
        linkers: Fragment | dict[int, Fragment | None],
        run: SlotRun | HelicalRun | list[SlotRun | HelicalRun],
    ) -> None:
        self.rod = rod
        # a helical run spirals its nodes: the screw axis is a fixed
        # line offset from the nodes (run.axis_point), the blueprint is
        # the *full* crystallographic period (screw_order node slots,
        # filled 1:1 by the rod's repeats - no supercell), and each
        # lateral slot takes exactly one linker. A straight run keeps
        # the original path: node at the axis, one period supercelled.
        # A cross-linked *multi-rod* net (etb) passes several helical
        # runs: one rod is placed on each, and the lateral linkers
        # bridge between different rods (see rod_specs / evaluate).
        self.runs = list(run) if isinstance(run, list) else [run]
        primary = self.runs[0]
        self.helical = isinstance(primary, HelicalRun)
        blueprint = np.asarray(topology.cell.matrix, dtype=float)
        self.blueprint = blueprint
        self.period = float(rod.repeat.repeat_length)
        # repeats placed along the axis: enough to close the screw (a
        # multiple of the screw order) and at least two, so a
        # continuation bond is always a distinct node pair rather than
        # a parallel edge. A straight rod (order 1) uses two.
        self.screw_rad = np.radians(rod.repeat.screw_angle)
        self.n_repeats = max(2, rod.repeat.screw_order)
        self.axis_length = self.n_repeats * self.period

        # runs grouped into *axis families* - a woven packing spirals
        # along several cell axes at once (bmn: six 4_1 helices, two per
        # axis, covering every node slot). Each family pins its own cell
        # parameter and carries its own transverse frame and placement
        # freedoms; a single-axis build is just the one-family case.
        by_axis: dict[int, list[SlotRun | HelicalRun]] = {}
        for a_run in self.runs:
            axis = int(np.argmax(np.abs(np.asarray(a_run.direction))))
            by_axis.setdefault(axis, []).append(a_run)
        self.families: list[dict] = []
        for axis_index in sorted(by_axis):
            axis_hat = blueprint[axis_index] / np.linalg.norm(blueprint[axis_index])
            # transverse orthonormal pair the rod's local x/y embed along
            seed = np.array([1.0, 0.0, 0.0])
            if abs(float(seed @ axis_hat)) > 0.9:
                seed = np.array([0.0, 1.0, 0.0])
            e1 = seed - (seed @ axis_hat) * axis_hat
            e1 /= np.linalg.norm(e1)
            self.families.append(
                {
                    "axis_index": axis_index,
                    "axis_hat": axis_hat,
                    "e1": e1,
                    "e2": np.cross(axis_hat, e1),
                    "runs": by_axis[axis_index],
                }
            )
        self.n_families = len(self.families)
        # the primary family's frame, for the single-axis code paths
        self.axis_index = self.families[0]["axis_index"]
        self.axis_hat = self.families[0]["axis_hat"]
        self.e1 = self.families[0]["e1"]
        self.e2 = self.families[0]["e2"]

        def _node_slots(a_run: SlotRun | HelicalRun) -> list[int]:
            return [
                s
                for s in a_run.slots
                if len(topology.slots[s].atoms.indices_from_symbol("X")) > 2
            ]

        self.node_slot_index = _node_slots(primary)[0]
        self.node_center_frac, _node_arms = _slot_geometry(
            topology.slots[self.node_slot_index], blueprint
        )

        # one placement spec per rod (per helical run): which axis family
        # it belongs to, where its axis line sits, the fractional centre
        # of its first node (fixes the azimuth + height its repeat 0
        # lands on), its signed screw, and whether the harvested rod must
        # be reflected to match the run's handedness (etb hosts both
        # hands; the enantiomer is a proper copy for an achiral
        # metal-oxo rod).
        self.rod_specs: list[dict] = []
        if self.helical:
            rod_sign = 1.0 if rod.repeat.screw_angle >= 0 else -1.0
            inverse = np.linalg.inv(blueprint)
            for family_index, family in enumerate(self.families):
                for a_run in family["runs"]:
                    assert isinstance(a_run, HelicalRun)
                    self.rod_specs.append(
                        {
                            "family": family_index,
                            # fractional, so the axis line follows an
                            # anisotropic cell instead of a scalar scale
                            # (a woven build pins more than one axis)
                            "axis_point_frac": np.asarray(a_run.axis_point) @ inverse,
                            # the reference node, unwrapped onto the run's
                            # own cylinder (a home-cell image can sit off it)
                            "node_frac": _unwrapped_node_frac(topology, a_run),
                            "screw_rad": np.radians(a_run.screw_angle),
                            "reflect": (1.0 if a_run.screw_angle >= 0 else -1.0)
                            != rod_sign,
                        }
                    )

        # lateral (linker) slots: everything not claimed by *any* rod run.
        # Each takes its own SBU species - one uniform ditopic linker is
        # just the common case (#168) - of any connectivity, or None to
        # leave the slot empty so its neighbours bond directly.
        run_slots = {s for a_run in self.runs for s in a_run.slots}
        self.lateral_slot_indices = [
            s for s in range(len(topology.slots)) if s not in run_slots
        ]
        per_slot = (
            linkers
            if isinstance(linkers, dict)
            else dict.fromkeys(self.lateral_slot_indices, linkers)
        )
        self.empty_slots = [s for s in self.lateral_slot_indices if per_slot[s] is None]
        self.species: list[_Species] = []
        species_index: dict[int, int] = {}  # id(fragment) -> species index
        self.lateral: list[_Lateral] = []
        for s in self.lateral_slot_indices:
            fragment = per_slot[s]
            if fragment is None:
                continue
            key = id(fragment)
            if key not in species_index:
                species_index[key] = len(self.species)
                self.species.append(_species_of(fragment))
            species = self.species[species_index[key]]
            center, frac_arms = _slot_geometry(topology.slots[s], blueprint)
            if len(frac_arms) != species.n_arms:
                raise AlignmentError(
                    f"Linker {fragment.name!r} has {species.n_arms} connection "
                    f"points but lateral slot {s} of {topology.name!r} needs "
                    f"{len(frac_arms)}."
                )
            orbit = topology.slots[s].equivalence_class
            # the reference arm assignment, at the blueprint cell; the
            # optimizer refreshes it once the cell has moved
            _, perm, _ = match_directions(
                _unit_rows(frac_arms @ blueprint), species.arm_units
            )
            self.lateral.append(
                _Lateral(
                    slot_index=s,
                    center_frac=center,
                    frac_arms=frac_arms,
                    species=species_index[key],
                    orbit=orbit if orbit is not None else -1 - s,
                    perm=perm,
                )
            )

        # a straight ditopic linker is free to rotate about its own arm
        # axis without moving its anchors; symmetry-equivalent slots
        # share that rotation (one free phi per orbit) so the built
        # framework keeps its symmetry. Slots with no such axis (a
        # polytopic SBU has none) take no phi at all.
        distinct = sorted(
            {
                lateral.orbit
                for lateral in self.lateral
                if self.species[lateral.species].relief_axis is not None
            }
        )
        self.orbit_position = {orbit: k for k, orbit in enumerate(distinct)}
        self.n_orbits = len(distinct)
        # linker copies per lateral slot: a helical blueprint already
        # enumerates every lateral slot of the period (one linker each);
        # a straight one holds a single period, supercelled n_repeats up
        self.linker_copies = 1 if self.helical else self.n_repeats

        self._rod_radius = [_covalent_radius(symbol) for symbol in rod.repeat.symbols]
        self._arm_rows = [row for row, _ in rod.arms]
        self._arm_local = np.array([vec for _, vec in rod.arms])
        # the enantiomer template + arms (reflect the local tangential
        # axis y -> -y, i.e. azimuth -> -azimuth): placed on an
        # opposite-handedness run, screwed by that run's negative angle,
        # this reproduces the mirror helix - the bridging atoms land on
        # the correct side (a straight/single-hand build never uses it)
        self._reflect = np.array([1.0, -1.0, 1.0])
        self._positions_reflected = self.rod.positions * self._reflect
        self._arm_local_reflected = (
            self._arm_local * self._reflect if len(self._arm_local) else self._arm_local
        )
        # rods placed this build: one per helical run (etb: several),
        # a single rod otherwise
        self.n_rods = len(self.rod_specs) if self.helical else 1

        # which blueprint node slot each rod repeat sits on: a helical
        # run's nodes are screw images of one another in walk order, so
        # repeat n fills the nth; a straight run reuses one node slot for
        # every repeat of its supercell. Indexed like rod_specs, i.e.
        # grouped by axis family.
        self.repeat_slot: list[list[int]] = []
        ordered = [a_run for family in self.families for a_run in family["runs"]]
        for a_run in ordered:
            nodes = [
                s
                for s in a_run.slots
                if len(topology.slots[s].atoms.indices_from_symbol("X")) > 2
            ]
            self.repeat_slot.append(
                nodes if self.helical else [nodes[0]] * self.n_repeats
            )
        self.port_budget = self._port_budget(topology)

    def _port_budget(self, topology: Topology) -> dict[tuple[int, int], int]:
        """How many bonds the blueprint expects between each slot pair.

        Read off the blueprint's rod form - the runs' axial edge centers
        and any emptied lateral slot already contracted away - minus the
        edges that are the rods' *own* continuation (a node bonding
        along its own run is an internal rod bond, not a connection
        point). Multiplied by the number of copies a straight run's
        supercell lays down.
        """
        from autografs.net import topology_rod_quotient_edges

        run_of_node = {
            slot: index
            for index, slots in enumerate(self.repeat_slot)
            for slot in slots
        }
        budget: dict[tuple[int, int], int] = {}
        contracted = topology_rod_quotient_edges(
            topology, list(self.runs), self.empty_slots
        )
        for (slot_a, slot_b, _voltage), count in contracted.items():
            if (
                slot_a in run_of_node
                and slot_b in run_of_node
                and run_of_node[slot_a] == run_of_node[slot_b]
            ):
                continue  # the rod's own continuation
            key = (min(slot_a, slot_b), max(slot_a, slot_b))
            budget[key] = budget.get(key, 0) + count * self.linker_copies
        return budget

    # ------------------------------------------------------------------
    # parameterized geometry
    # ------------------------------------------------------------------

    def cell(self, scale: float) -> np.ndarray:
        """The cell at this scale, with every rod-bearing axis pinned.

        A family's axis length is fixed by its rods (``n_repeats`` x the
        chemical repeat); ``scale`` only stretches whatever axes carry
        no rod. A fully woven packing pins all three and leaves nothing
        for scale to do.
        """
        cell = self.blueprint * scale
        for family in self.families:
            cell[family["axis_index"]] = family["axis_hat"] * self.axis_length
        return cell

    def _cell_one_period(self, scale: float) -> np.ndarray:
        cell = self.blueprint * scale
        cell[self.axis_index] = self.axis_hat * self.period
        return cell

    def _embed(self, local: np.ndarray, theta: float, family: dict) -> np.ndarray:
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x = cos_t * local[..., 0] - sin_t * local[..., 1]
        y = sin_t * local[..., 0] + cos_t * local[..., 1]
        return (
            np.outer(x, family["e1"])
            + np.outer(y, family["e2"])
            + np.outer(local[..., 2], family["axis_hat"])
        )

    def evaluate(
        self,
        scale: float,
        theta: np.ndarray | float,
        z0: np.ndarray | float,
        phi: np.ndarray | None = None,
    ) -> dict:
        """Place everything for one parameter set.

        ``theta`` and ``z0`` carry one value per axis family (a scalar
        is accepted for the single-axis case): the rotation about that
        family's axis and the axial phase of its rods.

        Returns a dict with rod positions, arm tips, linker
        placements, the connection-point pairing and the per-bond
        residuals |distance - covalent target|. ``phi`` (one angle per
        relievable lateral orbit) rotates each straight ditopic linker
        about its own arm axis to relieve ring-plane clashes without
        moving its anchors.
        """
        n_atoms = len(self.rod.positions)
        thetas = np.broadcast_to(np.atleast_1d(theta), (self.n_families,))
        z0s = np.broadcast_to(np.atleast_1d(z0), (self.n_families,))
        cell_p = self._cell_one_period(scale)
        cell_full = self.cell(scale)

        def screw_about(
            points: np.ndarray,
            axis_point: np.ndarray,
            axis_hat: np.ndarray,
            screw_rad: float,
            n: int,
        ) -> np.ndarray:
            """The nth screw image about a given axis line: rotate the
            perpendicular part of (points - axis_point) by n*screw, keep
            the axial part, then translate n*period along the axis."""
            rel = points - axis_point
            axial = (rel @ axis_hat)[:, None] * axis_hat
            perp = rel - axial
            if screw_rad:
                perp = Rotation.from_rotvec(n * screw_rad * axis_hat).apply(perp)
            return np.asarray(axis_point + axial + perp + n * self.period * axis_hat)

        arms: list[tuple[int, np.ndarray]] = []
        if self.helical:
            # one rod per helical run: place its repeat 0 at that run's
            # first node (azimuth + height), reflected to the run's
            # handedness, then screw n_repeats copies about the run's own
            # axis line. Lateral linkers are placed once (not screwed).
            linker_cell = cell_full
            rod_positions = np.empty(
                (len(self.rod_specs) * self.n_repeats * n_atoms, 3)
            )
            for ri, spec in enumerate(self.rod_specs):
                family = self.families[spec["family"]]
                axis_hat, e1, e2 = family["axis_hat"], family["e1"], family["e2"]
                theta_f = float(thetas[spec["family"]])
                z0_f = float(z0s[spec["family"]])
                axis_point = spec["axis_point_frac"] @ cell_full
                node_cart = spec["node_frac"] @ cell_full
                rel = node_cart - axis_point
                phi0 = float(np.arctan2(rel @ e2, rel @ e1))
                z_node = float(node_cart @ axis_hat)
                template = (
                    self._positions_reflected if spec["reflect"] else self.rod.positions
                )
                arm_local = (
                    self._arm_local_reflected if spec["reflect"] else self._arm_local
                )
                screw_rad = spec["screw_rad"]
                rod0 = self._embed(template, phi0 + theta_f, family) + (
                    axis_point + (z_node + z0_f) * axis_hat
                )
                arm_vec0 = (
                    self._embed(arm_local, phi0 + theta_f, family)
                    if len(arm_local)
                    else np.empty((0, 3))
                )
                base = ri * self.n_repeats * n_atoms
                for n in range(self.n_repeats):
                    start = base + n * n_atoms
                    rod_positions[start : start + n_atoms] = screw_about(
                        rod0, axis_point, axis_hat, screw_rad, n
                    )
                    arm_vec_n = (
                        Rotation.from_rotvec(n * screw_rad * axis_hat).apply(arm_vec0)
                        if screw_rad and len(arm_vec0)
                        else arm_vec0
                    )
                    for k, row in enumerate(self._arm_rows):
                        arms.append(
                            (start + row, rod_positions[start + row] + arm_vec_n[k])
                        )

            def screw(points: np.ndarray, n: int) -> np.ndarray:
                return points  # helical linkers are placed once, unscrewed
        else:
            line = self.node_center_frac @ cell_p
            line_perp = line - (line @ self.axis_hat) * self.axis_hat
            linker_cell = cell_p

            def screw(points: np.ndarray, n: int) -> np.ndarray:
                """The nth screw image about the rod axis line (pure
                translation for a straight rod, screw 0)."""
                rel = points - line_perp
                axial = (rel @ self.axis_hat)[:, None] * self.axis_hat
                perp = rel - axial
                if self.screw_rad:
                    perp = Rotation.from_rotvec(
                        n * self.screw_rad * self.axis_hat
                    ).apply(perp)
                return np.asarray(
                    line_perp + axial + perp + n * self.period * self.axis_hat
                )

            rod0 = self._embed(
                self.rod.positions, float(thetas[0]), self.families[0]
            ) + (line_perp + float(z0s[0]) * self.axis_hat)
            arm_vec0 = (
                self._embed(self._arm_local, float(thetas[0]), self.families[0])
                if len(self._arm_local)
                else np.empty((0, 3))
            )
            rod_positions = np.empty((self.n_repeats * n_atoms, 3))
            for n in range(self.n_repeats):
                start = n * n_atoms
                rod_positions[start : start + n_atoms] = screw(rod0, n)
                arm_vec_n = (
                    Rotation.from_rotvec(n * self.screw_rad * self.axis_hat).apply(
                        arm_vec0
                    )
                    if self.screw_rad and len(arm_vec0)
                    else arm_vec0
                )
                for k, row in enumerate(self._arm_rows):
                    arms.append(
                        (start + row, rod_positions[start + row] + arm_vec_n[k])
                    )

        placements: list[dict] = []
        for lateral in self.lateral:
            species = self.species[lateral.species]
            directions = _unit_rows(lateral.frac_arms @ linker_cell)
            ordered = species.arm_units[lateral.perm]
            rotation = kabsch(ordered, directions)
            rmsd = float(
                np.sqrt(((directions - ordered @ rotation.T) ** 2).sum(axis=1).mean())
            )
            centered = species.coords - species.dummy_center
            if phi is not None and species.relief_axis is not None:
                # pre-rotate about the arm axis: it fixes the arm
                # directions, so the alignment above is unchanged, but
                # the ring plane swings
                angle = float(phi[self.orbit_position[lateral.orbit]])
                centered = Rotation.from_rotvec(angle * species.relief_axis).apply(
                    centered
                )
            linker0 = centered @ rotation.T + lateral.center_frac @ linker_cell
            for n in range(self.linker_copies):
                placements.append(
                    {
                        "coords": screw(linker0, n),
                        "species": lateral.species,
                        "slot": lateral.slot_index,
                        "rmsd": rmsd,
                    }
                )

        cell = self.cell(scale)
        inv = np.linalg.inv(cell)

        # every connection point in the build is a *port*: a rod arm tip
        # or an SBU dummy. Ports pair with each other - rod to linker,
        # but also linker to linker, which every polytopic lateral net
        # needs (#168) - and the two anchors behind a paired couple bond.
        per_rod = self.n_repeats * n_atoms
        tips = [tip for _, tip in arms]
        anchors = [rod_positions[atom_index] for atom_index, _ in arms]
        radii = [self._rod_radius[atom_index % n_atoms] for atom_index, _ in arms]
        # the blueprint slot behind each port: the node slot this rod
        # repeat fills, or the lateral slot this linker sits on
        slots = [
            self.repeat_slot[atom_index // per_rod][(atom_index % per_rod) // n_atoms]
            for atom_index, _ in arms
        ]
        owners: list[tuple[int, ...]] = [
            (_ROD_PORT, atom_index) for atom_index, _ in arms
        ]
        for p_index, placement in enumerate(placements):
            species = self.species[placement["species"]]
            coords = placement["coords"]
            for k, dummy_row in enumerate(species.dummy_rows):
                anchor_row = species.anchor_rows[k]
                tips.append(coords[dummy_row])
                anchors.append(coords[anchor_row])
                radii.append(float(species.anchor_radii[k]))
                slots.append(placement["slot"])
                owners.append((_LINKER_PORT, p_index, anchor_row))
        if not arms:
            raise AlignmentError("Rod build has no connectable arms.")

        pairs = _pair_ports(
            np.asarray(tips), np.asarray(slots), self.port_budget, cell, inv
        )
        left, right = pairs[:, 0], pairs[:, 1]
        anchor_array = np.asarray(anchors)
        delta = (anchor_array[left] - anchor_array[right]) @ inv
        delta -= np.round(delta)
        distances = np.linalg.norm(delta @ cell, axis=1)
        radius_array = np.asarray(radii)
        residuals = np.abs(distances - (radius_array[left] + radius_array[right]))
        return {
            "cell": cell,
            "rod_positions": rod_positions,
            "arms": arms,
            "placements": placements,
            "bonds": [(owners[i], owners[j]) for i, j in pairs],
            "residuals": residuals,
        }

    def unpack(self, params: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """(scale, theta per family, z0 per family) from a flat vector."""
        rest = np.asarray(params[1:], dtype=float).reshape(self.n_families, 2)
        return float(params[0]), rest[:, 0], rest[:, 1]

    def objective(self, params: np.ndarray) -> float:
        scale, theta, z0 = self.unpack(params)
        if scale <= 0.05:
            return 1e6
        residuals = self.evaluate(scale, theta, z0)["residuals"]
        return float(np.sqrt((residuals**2).mean()))

    def initial_guess(self) -> np.ndarray:
        """(scale, then theta/z0 per family) good enough for Nelder-Mead."""
        if self.helical:
            # scale: the rod sits on the run's cylinder, so match the
            # rod's own radius to the (unscaled) blueprint node radius
            spec = self.rod_specs[0]
            node0 = spec["node_frac"] @ self.blueprint
            rel = node0 - spec["axis_point_frac"] @ self.blueprint
            node_radius = float(
                np.linalg.norm(rel - (rel @ self.axis_hat) * self.axis_hat)
            )
            rod_radius = (
                float(np.linalg.norm(self.rod.positions[self._arm_rows[0]][:2]))
                if self._arm_rows
                else float(np.linalg.norm(self.rod.positions[0][:2]))
            )
            scale0 = rod_radius / node_radius if node_radius > 1e-9 else 1.0
        elif not self.lateral:
            scale0 = 1.0  # nothing but the rod: the closure gate decides
        else:
            # scale: lateral node-to-node distance should fit two rod arms
            # plus the linker span between its dummies
            center_cart = self.node_center_frac @ self.blueprint
            lateral_center = self.lateral[0].center_frac @ self.blueprint
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
            first = self.species[self.lateral[0].species]
            span = (
                float(
                    np.linalg.norm(
                        first.coords[first.dummy_rows[0]]
                        - first.coords[first.dummy_rows[1]]
                    )
                )
                if first.n_arms == 2
                else 0.0
            )
            wanted = 2.0 * arm_length + span
            scale0 = (
                float(wanted / (2.0 * blueprint_half)) if blueprint_half > 1e-9 else 1.0
            )
        # z0: put the mean arm-carrying atom at the node height. For a
        # helical run z0 is an *offset* from each rod's own node axial
        # (added per spec in evaluate), so it starts near -z_arms; a
        # straight run places the rod absolutely, so z0 carries the whole
        # node height (in the supercelled single period).
        z_arms = (
            float(np.mean([self.rod.positions[row][2] for row in self._arm_rows]))
            if self._arm_rows
            else 0.0
        )
        if self.helical:
            per_family = [0.0, -z_arms] * self.n_families
            return np.array([scale0, *per_family])
        cell_p = self._cell_one_period(scale0)
        z_node = float((self.node_center_frac @ cell_p) @ self.axis_hat)
        return np.array([scale0, 0.0, z_node - z_arms])

    def refresh_assignments(self, scale: float) -> None:
        """Re-solve which SBU arm fills which slot dummy, at this cell.

        The reference assignment is made at the blueprint cell, but the
        rod pins one axis while the others scale, so the slot arm
        *directions* move with the optimized scale - and with them,
        for a polytopic SBU, the best assignment. Mirrors the finite
        pipeline's re-solve in ``BuildPlan.finalize``.
        """
        cell = self.cell(scale) if self.helical else self._cell_one_period(scale)
        for lateral in self.lateral:
            species = self.species[lateral.species]
            if species.n_arms < 3:
                continue  # a ditopic assignment is settled by the rotation
            _, perm, _ = match_directions(
                _unit_rows(lateral.frac_arms @ cell), species.arm_units
            )
            lateral.perm = perm

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
        # each rod is its own unit (a bond within a rod is rigid); in a
        # multi-rod net different rods can clash, so label them apart
        labels = [
            np.concatenate(
                [
                    np.full(self.n_repeats * n_atoms, -(ri + 1))
                    for ri in range(self.n_rods)
                ]
            )
        ]
        for p_index, placement in enumerate(placed["placements"]):
            reals = placement["coords"][self.species[placement["species"]].real_rows]
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


def _select_runs(
    topology: Topology, rod: RodFragment, rod_is_helical: bool
) -> list[SlotRun | HelicalRun]:
    """The run(s) a rod occupies.

    A straight rod takes one straight axial run. A helical rod takes
    *every* helical run whose screw matches the rod's - one distinct run
    per interleaved helix (etb has six, three of each hand). Runs are
    matched on node count (``screw_order``) and screw *magnitude*; the
    sign is handled per rod (an opposite-hand run hosts the reflected
    rod), so a centrosymmetric net with both hands is filled from one
    harvested rod. Runs are deduplicated by their node set, since
    ``helical_runs`` reports a helix once per starting node.

    Runs along **several** cell axes are taken together - a woven rod
    packing (bmn: six 4_1 helices, two along each cubic axis, covering
    every node slot). Each axis pins its own cell parameter to the rod's
    length, so only axes whose runs share the rod's period can be woven
    together; where they differ, the busiest period wins and the other
    runs' slots stay lateral, needing linkers like any other slot rather
    than being built along an axis pinned to the wrong length.
    """
    if rod_is_helical:

        def _nodeset(a_run: HelicalRun) -> frozenset[int]:
            return frozenset(
                s
                for s in a_run.slots
                if len(topology.slots[s].atoms.indices_from_symbol("X")) > 2
            )

        seen: set[frozenset[int]] = set()
        by_period: dict[float, list[SlotRun | HelicalRun]] = {}
        for a_run in helical_runs(topology):
            if a_run.screw_order != rod.repeat.screw_order:
                continue
            if abs(abs(a_run.screw_angle) - abs(rod.repeat.screw_angle)) > 5.0:
                continue
            ns = _nodeset(a_run)
            if ns in seen:
                continue
            seen.add(ns)
            key = next(
                (p for p in by_period if abs(p - a_run.period) < 1e-6), a_run.period
            )
            by_period.setdefault(key, []).append(a_run)
        if by_period:
            # busiest period first, its value as a deterministic tie-break
            period, matches = min(
                by_period.items(), key=lambda item: (-len(item[1]), item[0])
            )
            if len(by_period) > 1:
                logger.info(
                    f"\t[!] {topology.name!r} carries matching helical runs of "
                    f"{len(by_period)} different periods; building the "
                    f"{period:.3f} ones (one rod pins every axis it runs "
                    "along to the same length). The others' slots need "
                    "linkers like any other slot."
                )
            return matches
        # no matching helical run: a 2_1 (180 deg) screw still builds on
        # a straight run - a ditopic linker's arm sign-flip is a no-op,
        # so pcu hosts it - while a higher screw would fail the closure
        # gate there, which is the correct rejection
    runs = axial_runs(topology)
    if not runs:
        raise AlignmentError(
            f"Topology {topology.name!r} has no straight axial slot run "
            "(nor a matching helical one); rod building needs one."
        )
    return [runs[0]]


def _slot_adjacency(topology: Topology) -> dict[int, list[int]]:
    """Neighbour slot per half-edge, so len(adjacency[s]) is s's degree.

    Self-loops contribute both their ends, and parallel edges through
    different periodic images stay distinct: what is counted here is
    connections, not distinct neighbours.
    """
    from autografs.net import topology_quotient_edges

    adjacency: dict[int, list[int]] = {s: [] for s in range(len(topology.slots))}
    for (a, b, _voltage), count in topology_quotient_edges(topology).items():
        for _ in range(count):
            adjacency[a].append(b)
            adjacency[b].append(a)
    return adjacency


def _validate_linkers(
    topology: Topology,
    linkers: Fragment | dict[Fragment | int, Fragment | None],
    lateral_slots: list[int],
) -> Fragment | dict[int, Fragment | None]:
    """Normalize the lateral SBU mapping to one fragment per slot.

    Accepts a single Fragment (every lateral slot takes it - the common
    ditopic case) or a mapping keyed by slot type (a ``topology.mappings``
    key, covering a whole orbit) or by plain slot index, as
    ``Autografs.build`` does. Only the *lateral* slots need covering:
    the run slots are the rod's.

    A slot may be mapped to ``None``, which leaves it **empty**: nothing
    is placed and its two neighbours bond to each other directly. Real
    rod MOFs need this - MOF-74's metal-oxo rod bonds straight onto the
    4-connected DOBDC, while the blueprint (etb-e) decorates every edge
    with a 2-connected slot the structure has no unit for (#168). Only a
    2-connected slot can be emptied: contracting anything else is not a
    pass-through.
    """
    if not isinstance(linkers, dict):
        return linkers
    lateral = set(lateral_slots)
    per_slot: dict[int, Fragment | None] = {}
    for key, value in linkers.items():  # slot types first, indices override
        if isinstance(key, int):
            continue
        if key not in topology.mappings:
            valid = ", ".join(repr(s) for s in topology.mappings)
            raise AlignmentError(
                f"{key!r} is not a slot type of topology {topology.name!r}; "
                f"its slot types are: {valid}."
            )
        for slot in topology.mappings[key]:
            if slot in lateral:
                per_slot[slot] = value
    for key, value in linkers.items():
        if isinstance(key, int):
            if key not in lateral:
                raise AlignmentError(
                    f"Slot {key} of {topology.name!r} is not a lateral slot: "
                    "it belongs to the rod's run (or does not exist)."
                )
            per_slot[key] = value
    missing = lateral - per_slot.keys()
    if missing:
        raise AlignmentError(
            f"No linker given for lateral slot(s) {sorted(missing)} of "
            f"{topology.name!r}; every non-run slot needs an SBU (or None "
            "to leave it empty)."
        )
    for slot, value in per_slot.items():
        if value is None:
            degree = len(topology.slots[slot].atoms.indices_from_symbol("X"))
            if degree != 2:
                raise AlignmentError(
                    f"Slot {slot} of {topology.name!r} is {degree}-connected; "
                    "only a 2-connected slot can be left empty, since "
                    "emptying it bonds its two neighbours directly."
                )
    return per_slot


def build_rod_framework(
    topology: Topology,
    rod: RodFragment,
    linkers: Fragment | dict[Fragment | int, Fragment | None],
    run: SlotRun | HelicalRun | None = None,
    max_rmsd: float = DEFAULT_MAX_RMSD,
    min_distance: float | None = 1.0,
    bond_tolerance: float = DEFAULT_BOND_TOLERANCE,
    verify_net: bool = False,
    verbose: bool = False,
) -> Framework:
    """Build a rod framework - straight or helical - from a rod and linkers.

    Parameters
    ----------
    topology : Topology
        The blueprint. Must have a matching slot run: a straight axial
        run (``autografs.net.axial_runs``) for a screwless rod, or a
        helical run (``autografs.net.helical_runs``) whose screw agrees
        with the rod's for a helical one.
    rod : RodFragment
        A harvested rod (``HarvestResult.rods`` / ``load_rods``),
        straight or helical (``screw_order``/``screw_angle``), carrying
        its internal bond graph.
    linkers : Fragment or dict
        The finite SBUs filling the non-run (lateral) slots: one
        Fragment for all of them, or a mapping keyed by slot type
        (a ``topology.mappings`` key) or slot index, as
        ``Autografs.build`` takes. Any connectivity is allowed; a
        polytopic SBU bonds to its lateral neighbours as well as to
        the rod (#168). A 2-connected slot may be mapped to ``None``
        to leave it **empty**, bonding its two neighbours directly -
        how a real rod MOF binds its metal-oxo rod straight onto a
        polytopic linker where the blueprint decorates that edge.
    run : SlotRun or None, optional
        Which run the rod occupies; the first detected run when None
        (symmetry-equivalent runs give equivalent frameworks).
    max_rmsd : float, optional
        Directional gate on each linker's arm alignment.
    min_distance : float or None, optional
        Post-build closest-contact gate (continuation and inter-unit
        bonds are graph edges, hence exempt).
    bond_tolerance : float, optional
        Closure gate: largest allowed deviation of an optimized
        inter-unit bond from its covalent-length target, in Angstrom.
    verify_net : bool, optional
        When set, verify the built framework against ``topology``'s
        points-of-extension form (``Framework.verify_net``) and raise
        ``NetMismatchError`` if it does not realize the blueprint net.
    verbose : bool, optional
        Log the optimized cell and residuals.

    Returns
    -------
    Framework
        The built framework: ``n_repeats`` chemical repeats of the rod
        along the pinned axis (screw-generated for a helical rod) plus
        the linkers on the lateral slots.

    Raises
    ------
    AlignmentError
        For out-of-scope inputs (missing bonds, no matching run, an
        uncovered or mis-sized lateral slot) and failed gates.
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

    from autografs.rods import STRAIGHT_SCREW_TOL

    rod_is_helical = (
        rod.repeat.screw_order > 1 and abs(rod.repeat.screw_angle) > STRAIGHT_SCREW_TOL
    )
    runs = [run] if run is not None else _select_runs(topology, rod, rod_is_helical)
    run_slots = {s for a_run in runs for s in a_run.slots}
    adjacency = _slot_adjacency(topology)
    for a_run in runs:
        direction = np.asarray(a_run.direction)
        # each run must follow one cell axis (several *different* axes
        # across runs is fine - that is a woven packing); a diagonal run
        # would pin a combination of cell parameters, which the cell
        # parametrization cannot express
        if sorted(np.abs(direction).tolist()) != [0, 0, 1]:
            raise AlignmentError(
                "Rod building supports runs along a cell axis for now "
                f"(got direction {a_run.direction})."
            )
        if not isinstance(a_run, HelicalRun) and any(
            other.direction != a_run.direction for other in runs
        ):
            raise AlignmentError(
                "Woven (multi-axis) rod packings are supported for helical "
                "runs only; a straight run is supercelled along its own axis."
            )
        node_slots = [
            s
            for s in a_run.slots
            if len(topology.slots[s].atoms.indices_from_symbol("X")) > 2
        ]
        # a straight run has one PoE-bearing node per period; a helical
        # run has screw_order of them, filled 1:1 by the rod's repeats
        expected_nodes = a_run.screw_order if isinstance(a_run, HelicalRun) else 1
        if len(node_slots) != expected_nodes:
            raise AlignmentError(
                f"Rod building expects {expected_nodes} PoE-bearing node "
                f"slot(s) on this run but found {len(node_slots)}."
            )
        for node in node_slots:
            # lateral connections of a node = its degree minus the
            # connections the run itself consumes (its continuation).
            # Counted from run membership, not assumed to be two: a
            # blueprint node can sit on a run with any local geometry.
            degree = len(topology.slots[node].atoms.indices_from_symbol("X"))
            consumed = sum(1 for nb in adjacency[node] if nb in run_slots)
            if len(rod.arms) != degree - consumed:
                raise AlignmentError(
                    f"Rod {rod.name!r} carries {len(rod.arms)} arms per "
                    f"repeat but node slot {node} of this run expects "
                    f"{degree - consumed} lateral connections."
                )
    # lateral (linker) slots are whatever no rod run claims
    lateral_slots = [s for s in range(len(topology.slots)) if s not in run_slots]
    per_slot = _validate_linkers(topology, linkers, lateral_slots)

    build = _RodBuild(topology, rod, per_slot, runs if len(runs) > 1 else runs[0])

    # optimize: coarse rotation grid per axis family (a full product
    # grid would be 16^families), then refine everything with Nelder-Mead
    start = build.initial_guess()
    for family in range(build.n_families):
        angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)

        def value_at(angle: float, family: int = family) -> float:
            trial = start.copy()
            trial[1 + 2 * family] = angle
            return build.objective(trial)

        start[1 + 2 * family] = min(angles, key=value_at)
    result = minimize(
        build.objective,
        start,
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000},
    )
    scale, theta, z0 = build.unpack(result.x)
    # the polytopic arm assignments were fixed at the blueprint cell;
    # re-solve them now that the cell has moved
    build.refresh_assignments(scale)

    # relieve ditopic-linker ring-plane clashes: rotating each linker
    # about its own arm axis keeps the anchors (hence the bonds) fixed,
    # so maximize the minimum inter-unit contact over one phi per
    # relievable lateral orbit. A ring has pi symmetry, so the search
    # spans [0, pi); grid then refine.
    def relief(phi: np.ndarray) -> float:
        return -build.min_inter_unit_contact(build.evaluate(scale, theta, z0, phi=phi))

    best_phi: np.ndarray | None = None
    if build.n_orbits == 1:
        grid = np.linspace(0.0, np.pi, 24, endpoint=False)
        best_phi = np.array([min(grid, key=lambda a: relief(np.array([a])))])
    elif build.n_orbits > 1:
        best_phi = np.zeros(build.n_orbits)
        for _ in range(4):  # coordinate ascent over the orbits
            for k in range(build.n_orbits):
                grid = np.linspace(0.0, np.pi, 16, endpoint=False)

                def score(a: float, k: int = k) -> float:
                    trial = best_phi.copy()  # type: ignore[union-attr]
                    trial[k] = a
                    return relief(trial)

                best_phi[k] = min(grid, key=score)
    if best_phi is not None:
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
            f"inter-unit bond deviates {worst_bond:.2f} A from its "
            f"covalent target (bond_tolerance={bond_tolerance})."
        )
    worst_rmsd = max((p["rmsd"] for p in placed["placements"]), default=0.0)
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
    if verify_net:
        from autografs.net import verify_net as _verify_net

        _verify_net(framework, topology)
    return framework


def _assemble_graph(build: _RodBuild, result: dict, find_cutoffs, load_uff):
    """The built framework's bond graph, from placed geometry.

    Mirrors ``utils.fragments_to_networkx`` conventions (node attrs,
    intra-unit bonds with orders) with two rod specifics: rod internal
    bonds come from the recorded ``RodFragment.bonds`` (continuations
    wrap between repeats), and every inter-unit bond - rod to linker
    and linker to linker alike - is an explicit edge from the
    optimizer's port pairing. All tags are 0.
    """
    from pymatgen.core.bonds import get_bond_order

    # rod_build marks the framework so tag/anchor-based editing ops
    # (defects, flip, rotate, functionalize) refuse it - see editing._reject_rod.
    # rod_empty_slots records the blueprint slots deliberately left empty,
    # so net verification contracts them out of the blueprint too.
    graph = networkx.Graph(cell=result["cell"], rod_build=True)
    if build.empty_slots:
        graph.graph["rod_empty_slots"] = list(build.empty_slots)
    n_atoms = len(build.rod.positions)

    # rod atoms: typed once on the first repeat, replicated across every
    # repeat of every rod (a multi-rod net places one rod per helical
    # run; each rod-repeat is its own slot so verify_net can tell them
    # apart). Global atom ids match result["rod_positions"] indices, so
    # the optimizer's matching (which uses those) wires up directly.
    rod_molgraph = _typed_repeat_molgraph(build, result)
    rod_types = [rod_molgraph.molecule[i].properties["ufftype"] for i in range(n_atoms)]
    per_rod = build.n_repeats * n_atoms
    for ri in range(build.n_rods):
        for n in range(build.n_repeats):
            start = ri * per_rod + n * n_atoms
            for i in range(n_atoms):
                graph.add_node(
                    start + i,
                    symbol=build.rod.repeat.symbols[i],
                    coord=result["rod_positions"][start + i],
                    tag=0,
                    ufftype=rod_types[i],
                    slot=ri * build.n_repeats + n,
                    sbu=build.rod.name,
                )
    for ri in range(build.n_rods):
        for n in range(build.n_repeats):
            start = ri * per_rod + n * n_atoms
            for a, b, m in build.rod.bonds:
                partner_start = ri * per_rod + ((n + m) % build.n_repeats) * n_atoms
                graph.add_edge(start + a, partner_start + b, bond_order=1.0)

    # linkers: one molgraph per placement for types, bonds and orders
    offset = build.n_rods * per_rod
    anchor_nodes: list[dict[int, int]] = []
    for p_index, placement in enumerate(result["placements"]):
        linker = build.species[placement["species"]]
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
                slot=build.n_rods * build.n_repeats + p_index,
                sbu=linker.fragment.name,
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

    def node_of(owner: tuple[int, ...]) -> int:
        """Graph node behind one paired port."""
        if owner[0] == _ROD_PORT:
            return owner[1]  # rod atom ids are the graph ids
        # a linker anchor row indexes the original SBU (dummies
        # included); map it onto the dummy-stripped molgraph row
        _kind, p_index, anchor_row = owner
        linker = build.species[result["placements"][p_index]["species"]]
        return anchor_nodes[p_index][linker.stripped_rows[anchor_row]]

    # inter-unit bonds from the optimizer's port pairing: rod to linker,
    # and linker to linker wherever the blueprint's laterals touch (#168)
    for owner_a, owner_b in result["bonds"]:
        graph.add_edge(node_of(owner_a), node_of(owner_b), bond_order=1.0)
    return graph


def _placed_linker_molgraph(build: _RodBuild, p_index: int, result: dict):
    """UFF-typed molgraph of one placed linker (dummies included then
    stripped by fragment_to_molgraph)."""
    from autografs.utils import fragment_to_molgraph

    placement = result["placements"][p_index]
    linker = build.species[placement["species"]]
    # the placement transformed every atom of the original linker
    # (real and dummy) together; "coords" holds them all
    molecule = Molecule(
        linker.symbols,
        placement["coords"],
        site_properties={"tags": [0] * len(linker.symbols)},
    )
    return fragment_to_molgraph(Fragment(atoms=molecule, name=linker.fragment.name))
