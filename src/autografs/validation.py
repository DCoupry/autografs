"""Is this built framework a structure that makes sense?

The builder places rigid units on a blueprint and optimizes a cell. It
does that well, but "it built" and "it is a material" are different
claims, and until now the second one had to be assembled by the caller
out of several unrelated calls - some of them off by default.

Measured on 1895 builds: with stock settings the pipeline returns 1105
structures whose inter-unit bonds do not close and 521 whose atoms
overlap, because ``bond_tolerance`` and ``min_distance`` are both
``None`` unless asked for. A further 10 carried free molecules, for
which no check existed anywhere. Even on the best-behaved blueprints
(every slot pinned by site symmetry) only a quarter of builds pass all
three.

So this module composes every post-build check into one verdict:

>>> report = framework.validate()
>>> report.ok
False
>>> print(report)
FAILED: closure 1.42 A > 0.50; contact 0.31 A < 1.50 (2 of 4 checks)

Each check reports its measured value next to the threshold it was
judged against, so a near miss is distinguishable from a catastrophe -
a single boolean would throw away exactly the information needed to
decide whether a relaxation can rescue the structure.

Nothing here changes what ``build`` returns. The gates are opt-in via
``build(..., strict=True)``; this is how you find out either way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import networkx
import numpy as np

if TYPE_CHECKING:
    from autografs.framework import Framework

__all__ = [
    "BuildAssessment",
    "Check",
    "STERIC_BALANCED",
    "STERIC_STRAINED",
    "STRICT_BOND_TOLERANCE",
    "STRICT_MIN_DISTANCE",
    "SlotAssessment",
    "ValidationReport",
    "assess_build",
    "balance_mappings",
    "edge_scales",
    "inter_unit_bond_deviations",
    "scale_spread",
    "validate_framework",
]

#: Worst inter-unit bond deviation a strict build will accept, in
#: Angstrom. Not a physical constant: over builds that already pass the
#: shape gate the worst-bond deviation has median 0.43 A and p90 1.14 A,
#: so this admits the well-closed half and rejects the tail. Override it
#: per call rather than tuning it here.
STRICT_BOND_TOLERANCE = 0.5
#: Closest non-bonded contact a strict build will accept, in Angstrom.
#: Below this the structure is a starting point for a relaxation, not a
#: candidate: real frameworks sit comfortably above it (MOF-5 built from
#: the shipped library measures 2.04 A).
STRICT_MIN_DISTANCE = 1.5
#: Two atoms closer than this are coincident, not bonded. Catches a
#: corrupted relaxation, which can return bonded pairs at 1e-8 A.
COINCIDENT_DISTANCE = 0.3


@dataclass(frozen=True)
class Check:
    """One named verdict, with the number behind it.

    ``value`` and ``threshold`` are kept so callers can see *how far*
    a check missed. ``value`` is None when the quantity could not be
    measured at all, which is distinct from measuring badly and is
    never silently treated as a pass.
    """

    name: str
    passed: bool
    value: float | None = None
    threshold: float | None = None
    detail: str = ""

    def __str__(self) -> str:
        mark = "ok" if self.passed else "FAIL"
        if self.value is None:
            return (
                f"{self.name}: {mark} ({self.detail})"
                if self.detail
                else f"{self.name}: {mark}"
            )
        return f"{self.name}: {mark} {self.value:.3g}" + (
            f" vs {self.threshold:.3g}" if self.threshold is not None else ""
        )


@dataclass(frozen=True)
class ValidationReport:
    """Every post-build check, and the descriptors behind them."""

    checks: tuple[Check, ...] = ()
    descriptors: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when every check passed."""
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def __getitem__(self, name: str) -> Check:
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(name)

    def __str__(self) -> str:
        if self.ok:
            return f"OK ({len(self.checks)} checks passed)"
        bad = "; ".join(str(check) for check in self.failures)
        return f"FAILED: {bad} ({len(self.failures)} of {len(self.checks)} checks)"


def inter_unit_bond_deviations(framework: Framework) -> np.ndarray:
    """How far every inter-unit bond sits from its covalent target.

    The realized closure of a build. Bonds internal to one placed SBU
    are excluded - those are fixed by the fragment, not by the cell -
    but a same-slot bond that *crosses* a boundary is a unit bonded to
    its own periodic image, which is real and is measured. Elements
    absent from the Cordero table have no target and are skipped rather
    than scored against zero, which would report a whole bond length as
    the deviation and read as a catastrophic build.
    """
    from pymatgen.analysis.local_env import CovalentRadius

    graph = framework.graph
    cell = np.asarray(graph.graph["cell"], dtype=float)
    inverse = np.linalg.inv(cell)
    radii = CovalentRadius.radius
    deviations: list[float] = []
    for node_a, node_b in graph.edges():
        data_a = graph.nodes[node_a]
        data_b = graph.nodes[node_b]
        delta = np.asarray(data_a["coord"], float) - np.asarray(data_b["coord"], float)
        crossing = np.round(delta @ inverse)
        if data_a.get("slot") == data_b.get("slot") and not np.any(crossing):
            continue
        symbols = (data_a["symbol"], data_b["symbol"])
        if any(symbol not in radii for symbol in symbols):
            continue
        delta -= crossing @ cell
        target = sum(radii[symbol] for symbol in symbols)
        deviations.append(abs(float(np.linalg.norm(delta)) - target))
    return np.asarray(deviations, dtype=float)


def free_molecule_count(framework: Framework) -> tuple[int, int]:
    """(bond-graph components, how many are free molecules).

    A component with no boundary-crossing bond is 0-periodic: a
    molecule sitting in the cell, not part of the framework. Several
    *periodic* components are legitimate - that is interpenetration -
    so only the 0-periodic ones are a defect. Same convention
    ``deconstruct`` uses when it strips guests.
    """
    graph = framework.graph
    cell = np.asarray(graph.graph["cell"], dtype=float)
    inverse = np.linalg.inv(cell)
    components = list(networkx.connected_components(graph))
    free = 0
    for component in components:
        crossing = False
        for node_a, node_b in graph.subgraph(component).edges():
            delta = np.asarray(graph.nodes[node_a]["coord"], float) - np.asarray(
                graph.nodes[node_b]["coord"], float
            )
            if np.any(np.round(delta @ inverse)):
                crossing = True
                break
        if not crossing:
            free += 1
    return len(components), free


def validate_framework(
    framework: Framework,
    bond_tolerance: float | None = STRICT_BOND_TOLERANCE,
    min_distance: float | None = STRICT_MIN_DISTANCE,
    require_connected: bool = True,
) -> ValidationReport:
    """Run every post-build check and report them together.

    Parameters
    ----------
    bond_tolerance : float or None, optional
        Worst acceptable inter-unit bond deviation (A). None skips the
        closure check.
    min_distance : float or None, optional
        Closest acceptable non-bonded contact (A). None skips the
        overlap check.
    require_connected : bool, optional
        Fail when the framework carries free (0-periodic) molecules.

    Returns
    -------
    ValidationReport
        Checks plus descriptors; ``report.ok`` is the single verdict.
    """
    checks: list[Check] = []
    structure = framework.structure
    n_components, n_free = free_molecule_count(framework)
    descriptors: dict = {
        "n_atoms": len(structure),
        "n_bonds": framework.graph.number_of_edges(),
        "n_components": n_components,
        "n_free_molecules": n_free,
        "volume_per_atom": float(structure.volume / max(len(structure), 1)),
        "n_distinct_sbus": len(set(framework.slots.values())) if framework.slots else 0,
    }

    deviations = inter_unit_bond_deviations(framework)
    worst = float(deviations.max()) if deviations.size else None
    descriptors["worst_bond_deviation"] = worst
    descriptors["median_bond_deviation"] = (
        float(np.median(deviations)) if deviations.size else None
    )
    if bond_tolerance is not None:
        if worst is None:
            # no inter-unit bond could be measured: closure is UNKNOWN,
            # and passing on a measurement that does not exist is how a
            # bare node net scores as a closed framework
            checks.append(
                Check(
                    "closure",
                    False,
                    None,
                    bond_tolerance,
                    "no inter-unit bond could be measured",
                )
            )
        else:
            checks.append(
                Check("closure", worst <= bond_tolerance, worst, bond_tolerance)
            )

    contact = float(framework.min_contact())
    descriptors["min_contact"] = contact
    if min_distance is not None:
        checks.append(Check("contact", contact >= min_distance, contact, min_distance))

    if require_connected:
        checks.append(
            Check(
                "connectivity",
                n_free == 0,
                float(n_free),
                0.0,
                f"{n_free} free molecule(s) of {n_components} component(s)",
            )
        )

    # coincident atoms: never legitimate, and cheap insurance against a
    # corrupted relaxation being carried forward as a result
    closest = _closest_pair(structure)
    descriptors["closest_any_pair"] = closest
    checks.append(
        Check(
            "coincident_atoms",
            closest is None or closest >= COINCIDENT_DISTANCE,
            closest,
            COINCIDENT_DISTANCE,
        )
    )
    return ValidationReport(tuple(checks), descriptors)


#: Above this many symmetry-allowed slot displacements a blueprint's
#: proportions are not fixed by its own symmetry, so the idealized
#: embedding carries arbitrary ones that a single cell scale cannot
#: correct - whatever units fill it.
FREEDOM_MEDIUM = 1
FREEDOM_LOW = 5

#: Edge-scale spread (max/min) below which the units are proportioned
#: consistently enough that one cell scale suits every edge. Corpus
#: measured over 355 builds: below 1.05, 30% come out clash-free;
#: above 1.6, 0.6% do (1 of 160).
STERIC_BALANCED = 1.05
#: Above this the edges disagree so badly about the cell scale that the
#: optimizer must compromise, and the losing edges either stretch their
#: bonds or drive their units into each other.
STERIC_STRAINED = 1.3


def edge_scales(topology, mappings: dict) -> np.ndarray:
    """The cell scale each blueprint edge would need, one per edge.

    Two slots sharing a connection tag have their dummies at the *same*
    point, so the blueprint's centre-to-centre distance across that edge
    is exactly ``slot_arm_A + slot_arm_B``. Filling them demands
    ``sbu_arm_A + sbu_arm_B``. The cell carries ONE scale, so each edge
    wants its own ``s = sbu_sum / slot_sum`` and the optimizer has to
    compromise between them - the losers stretch their bonds or push
    their units together.

    This is the mechanism behind the failure the pipeline actually
    produces. Clashing builds overlap between *bonded* slots (measured:
    344 of 364 sub-1.2 A pairs on ith-d, 262 of 300 on lcw-x), not
    between distant ones, which is what a bulk-versus-void picture would
    predict. Nothing here needs alignment or a cell optimization, so it
    is cheap enough to screen with.

    Returns
    -------
    np.ndarray
        One required scale per paired edge; empty when the blueprint
        has no shared tags among the mapped slots.
    """
    from autografs.alignment import match_directions

    per_slot: dict[int, tuple[list[int], np.ndarray, np.ndarray]] = {}
    for slot_index, sbu in mappings.items():
        if sbu is None:
            continue
        slot = topology.slots[slot_index]
        dummy_idx = [i for i, s in enumerate(slot.atoms) if s.specie.symbol == "X"]
        if not dummy_idx:
            continue
        tags = [int(slot.atoms[i].properties["tags"]) for i in dummy_idx]
        slot_dummies = np.asarray(slot.atoms.cart_coords)[dummy_idx]
        slot_arms = slot_dummies - slot_dummies.mean(axis=0)
        slot_len = np.linalg.norm(slot_arms, axis=1)

        sbu_idx = [i for i, s in enumerate(sbu.atoms) if s.specie.symbol == "X"]
        if len(sbu_idx) != len(dummy_idx):
            continue
        sbu_dummies = np.asarray(sbu.atoms.cart_coords)[sbu_idx]
        sbu_arms = sbu_dummies - sbu_dummies.mean(axis=0)
        sbu_len = np.linalg.norm(sbu_arms, axis=1)
        if len(slot_len) > 2:
            # which SBU arm serves which slot arm; for 1-2 arms the
            # assignment is immaterial (both are the same length)
            units_slot = slot_arms / np.maximum(slot_len[:, None], 1e-12)
            units_sbu = sbu_arms / np.maximum(sbu_len[:, None], 1e-12)
            _, permutation, _ = match_directions(units_slot, units_sbu)
            sbu_len = sbu_len[permutation]
        per_slot[slot_index] = (tags, slot_len, sbu_len)

    by_tag: dict[int, list[tuple[float, float]]] = {}
    for tags, slot_len, sbu_len in per_slot.values():
        for position, tag in enumerate(tags):
            by_tag.setdefault(tag, []).append(
                (float(slot_len[position]), float(sbu_len[position]))
            )
    scales = []
    for entries in by_tag.values():
        if len(entries) != 2:
            # an unpaired dummy constrains nothing
            continue
        slot_sum = entries[0][0] + entries[1][0]
        sbu_sum = entries[0][1] + entries[1][1]
        if slot_sum > 1e-9:
            scales.append(sbu_sum / slot_sum)
    return np.asarray(scales, dtype=float)


def scale_spread(scales: np.ndarray) -> float | None:
    """How badly the edges disagree about the cell scale (max/min)."""
    if scales.size < 2:
        return None
    smallest = float(scales.min())
    if smallest <= 1e-9:
        return None
    return float(scales.max()) / smallest


@dataclass(frozen=True)
class SlotAssessment:
    """How well one SBU suits one slot type, before any build."""

    slot_type: int
    slot_arms: int
    sbu: str
    sbu_arms: int
    rmsd: float | None
    threshold: float

    @property
    def fits(self) -> bool:
        return self.rmsd is not None and self.rmsd <= self.threshold

    @property
    def margin(self) -> float | None:
        """How much shape tolerance is left; negative means it does not fit."""
        return None if self.rmsd is None else self.threshold - self.rmsd

    def __str__(self) -> str:
        if self.rmsd is None:
            return (
                f"slot {self.slot_type} ({self.slot_arms}-c): {self.sbu} has "
                f"{self.sbu_arms} arms - no fit at any threshold"
            )
        return (
            f"slot {self.slot_type} ({self.slot_arms}-c): {self.sbu} "
            f"rmsd {self.rmsd:.3f} of {self.threshold:.2f}"
        )


@dataclass(frozen=True)
class BuildAssessment:
    """Will this (net, SBU) combination give a sensible structure?

    Answered from the blueprint and the units alone - no build, no cell
    optimization - so a screening loop can skip the combinations that
    were never going to work and a user can be told *why* before
    spending the time.

    Three things decide it, all cheap: whether each SBU has the right
    connectivity, how well its arm *directions* match the slot, and
    whether the units are proportioned consistently enough for one cell
    scale to suit every edge (``scale_spread``). The last is what
    predicts packing, and it is the dominant failure mode in practice.

    ``confidence`` is a qualitative class, deliberately not a
    probability: the outcome depends on the specific chemistry as well
    as these descriptors, and a number would imply a per-structure
    guarantee this cannot make.

    - ``infeasible`` - the build will raise; some slot has no SBU of
      the right connectivity.
    - ``low`` - expect to discard it. A slot is over its shape
      threshold, or the blueprint has many free proportions that one
      cell scale cannot set.
    - ``medium`` - no obstruction found, but the blueprint carries
      free proportions; build it and check.
    - ``high`` - every slot fits comfortably and the blueprint is
      fully pinned by its own site symmetry, which is the regime
      pcu/dia/srs/fcu/sql/hcb live in. Still check the packing.
    """

    net: str
    n_slots: int
    n_free: int | None
    is_2d: bool
    slots: tuple[SlotAssessment, ...]
    confidence: str
    reasons: tuple[str, ...] = ()
    #: How badly the blueprint's edges disagree about the cell scale
    #: (max/min of the per-edge requirement). 1.0 is perfect agreement;
    #: None when the blueprint has too few paired edges to compare.
    scale_spread: float | None = None

    @property
    def feasible(self) -> bool:
        """False when the build cannot succeed at all."""
        return self.confidence != "infeasible"

    @property
    def worst_slot(self) -> SlotAssessment | None:
        scored = [s for s in self.slots if s.rmsd is not None]
        if not scored:
            return self.slots[0] if self.slots else None
        return max(scored, key=lambda s: s.rmsd or 0.0)

    def __str__(self) -> str:
        head = f"{self.net}: {self.confidence}"
        if self.reasons:
            head += " (" + "; ".join(self.reasons) + ")"
        return head


def assess_build(
    topology,
    mappings: dict,
    max_rmsd: float | None = None,
) -> BuildAssessment:
    """Judge a (net, SBU) combination without building it.

    Parameters
    ----------
    topology : Topology
        The blueprint.
    mappings : dict[int, Fragment]
        One Fragment per slot index, as ``_validate_mappings`` returns.
        Emptied slots (mapped to None) are skipped.
    max_rmsd : float or None, optional
        Shape threshold each slot is judged against; defaults to the
        sieve's own ``COMPATIBILITY_MAX_RMSD``.

    Returns
    -------
    BuildAssessment
    """
    from autografs.fragment import COMPATIBILITY_MAX_RMSD

    threshold = COMPATIBILITY_MAX_RMSD if max_rmsd is None else max_rmsd
    slots: list[SlotAssessment] = []
    for slot_type, indices in topology.mappings.items():
        index = next((i for i in indices if mappings.get(i) is not None), None)
        if index is None:
            continue
        fragment = mappings[index]
        slots.append(
            SlotAssessment(
                slot_type=index,
                slot_arms=len(slot_type.arm_units),
                sbu=getattr(fragment, "name", "?"),
                sbu_arms=len(fragment.arm_units),
                rmsd=slot_type.match_rmsd(fragment),
                threshold=threshold,
            )
        )

    n_free: int | None
    try:
        from autografs.symmetry import orbit_displacements

        n_free = int(orbit_displacements(topology).n_free)
    except Exception:  # noqa: BLE001 - a descriptor must never block a build
        n_free = None

    spread = scale_spread(edge_scales(topology, mappings))
    reasons: list[str] = []
    misfit = [s for s in slots if s.rmsd is None]
    strained = [s for s in slots if s.rmsd is not None and s.rmsd > threshold]
    if misfit:
        confidence = "infeasible"
        reasons.append(f"{len(misfit)} slot(s) have no SBU of the right connectivity")
    else:
        if strained:
            worst = max(s.rmsd or 0.0 for s in strained)
            reasons.append(
                f"{len(strained)} slot(s) over the shape threshold "
                f"(worst rmsd {worst:.2f} of {threshold:.2f})"
            )
        if n_free is None:
            reasons.append("blueprint freedom could not be determined")
        elif n_free > FREEDOM_LOW:
            reasons.append(
                f"blueprint has {n_free} free proportions one cell scale cannot set"
            )
        elif n_free > FREEDOM_MEDIUM:
            reasons.append(f"blueprint has {n_free} free proportions")
        elif n_free == FREEDOM_MEDIUM:
            reasons.append("blueprint has 1 free proportion")
        if spread is not None and spread > STERIC_STRAINED:
            reasons.append(
                f"units are unevenly proportioned for this net: its edges "
                f"disagree about the cell scale by x{spread:.2f}"
            )
        elif spread is not None and spread > STERIC_BALANCED:
            reasons.append(f"edges disagree about the cell scale by x{spread:.2f}")
        if (
            strained
            or (n_free is not None and n_free > FREEDOM_LOW)
            or (spread is not None and spread > STERIC_STRAINED)
        ):
            confidence = "low"
        elif (
            n_free is None
            or n_free > 0
            or (spread is not None and spread > STERIC_BALANCED)
        ):
            confidence = "medium"
        else:
            confidence = "high"
            reasons.append(
                "every slot fits, the units are evenly proportioned, and "
                "the blueprint is fully pinned"
            )
    return BuildAssessment(
        net=topology.name,
        n_slots=len(topology),
        n_free=n_free,
        is_2d=bool(getattr(topology, "is_2d", False)),
        slots=tuple(slots),
        confidence=confidence,
        reasons=tuple(reasons),
        scale_spread=spread,
    )


def slot_arm_length(slot) -> float:
    """Mean distance from a slot's dummy centroid to its dummies."""
    dummy_idx = [i for i, s in enumerate(slot.atoms) if s.specie.symbol == "X"]
    if not dummy_idx:
        return 0.0
    dummies = np.asarray(slot.atoms.cart_coords)[dummy_idx]
    return float(np.linalg.norm(dummies - dummies.mean(axis=0), axis=1).mean())


def balance_mappings(
    topology,
    candidates: dict,
    targets: int = 41,
) -> tuple[dict, float | None]:
    """Pick, per slot type, the SBU that keeps every edge at one scale.

    The selection fix for the failure ``edge_scales`` measures. An edge's
    required scale is a weighted average of the two slots' own ratios
    ``sbu_arm / slot_arm``, so making those ratios UNIFORM across slot
    types makes every edge agree - which turns a combinatorial search
    into a per-slot choice against a common target ratio.

    Without this the caller picks arbitrarily (the first compatible name,
    typically), and the sieve is no help because it matches arm
    *directions* and never looks at length: for a 2-connected slot it is
    vacuous, since any two arms seen from their own centroid are
    antiparallel. A 30-atom linker and a 6-atom one are equally
    "compatible" with the same edge.

    Parameters
    ----------
    topology : Topology
        The blueprint.
    candidates : dict[slot type, list[Fragment]]
        Per slot type, the fragments worth considering - typically the
        sieve's own output resolved to Fragments.
    targets : int, optional
        How many candidate target ratios to sweep.

    Returns
    -------
    tuple[dict, float | None]
        The chosen {slot type: Fragment} mapping, and its predicted
        edge-scale spread (None when the blueprint has too few paired
        edges to compare).

    Notes
    -----
    The search is over the family "every slot takes the candidate
    closest to a common target ratio", swept over ``targets`` values -
    not over the full product, which is combinatorial on multinodal
    nets. That family contains the uniform-ratio optimum by
    construction, but a hand-picked combination can occasionally beat
    the result by a fraction of a percent. Measured against the pick it
    replaces (first compatible name per slot) over 59 nets: fully valid
    builds 0 -> 11, clash-free 0 -> 12, median closest contact 0.42 ->
    0.86 A, better on 41 nets and worse on 14. It optimizes the
    *predicted* spread, not the realized contact, so it is an
    improvement in distribution rather than a guarantee per net.
    """
    ratios: dict = {}
    for slot_type, fragments in candidates.items():
        arm = slot_arm_length(slot_type)
        if arm <= 1e-9:
            continue
        options = []
        for fragment in fragments:
            dummy_idx = [
                i for i, s in enumerate(fragment.atoms) if s.specie.symbol == "X"
            ]
            if not dummy_idx:
                continue
            dummies = np.asarray(fragment.atoms.cart_coords)[dummy_idx]
            length = float(
                np.linalg.norm(dummies - dummies.mean(axis=0), axis=1).mean()
            )
            options.append((length / arm, fragment))
        if options:
            ratios[slot_type] = options
    if not ratios:
        return {}, None

    def spread_of(mapping: dict) -> float | None:
        resolved = {}
        for slot_type, fragment in mapping.items():
            for index in topology.mappings[slot_type]:
                resolved[index] = fragment
        return scale_spread(edge_scales(topology, resolved))

    everything = [r for options in ratios.values() for r, _ in options]
    grid = np.linspace(min(everything), max(everything), max(targets, 2))
    best_mapping: dict = {}
    best_spread: float | None = None
    for target in grid:
        mapping = {
            slot_type: min(options, key=lambda pair: abs(pair[0] - target))[1]
            for slot_type, options in ratios.items()
        }
        spread = spread_of(mapping)
        if spread is None:
            # nothing to optimize against; any choice is as good
            return mapping, None
        if best_spread is None or spread < best_spread:
            best_mapping, best_spread = mapping, spread

    # DO NOT add a coordinate-descent pass here to squeeze the spread
    # further. It was tried, and it makes the proxy better while making
    # the RESULT worse: over the same 60 nets, driving the spread to its
    # floor moved fully-valid builds 11 -> 9, clash-free 12 -> 10 and the
    # median contact 0.863 -> 0.824 A. Spread is a stand-in for packing,
    # not packing itself - it compares slots by their mean arm length and
    # cannot see how uneven one SBU's own arms are - so its minimum is
    # not the best structure. The uniform-ratio sweep is the criterion
    # with a physical argument behind it; keep the optimizer honest by
    # stopping there.
    return best_mapping, best_spread


def _closest_pair(structure) -> float | None:
    """Closest distance between any two atoms, bonded or not."""
    if len(structure) < 2:
        return None
    centers, points, _, distances = structure.get_neighbor_list(r=COINCIDENT_DISTANCE)
    real = distances[centers != points]
    if real.size:
        return float(real.min())
    matrix = structure.distance_matrix.copy()
    np.fill_diagonal(matrix, math.inf)
    value = float(matrix.min())
    return value if math.isfinite(value) else None
