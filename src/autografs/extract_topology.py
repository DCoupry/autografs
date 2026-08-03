"""Topology extraction from tetrahedral framework structures.

The fast path from an idealized zeolite crystal (T atoms bridged by
oxygen, e.g. the IZA idealized SiO2 CIFs) to a buildable library
Topology: T atoms become 4-connected node slots, bridging oxygens
become 2-connected edge-center slots, and the connection dummies sit
at the T-O midpoints - the quarter points of the T...T edge, exactly
the convention the CGD parser produces for RCSR nets, so an extracted
zeolite and a CGD-imported net are structurally identical down to the
exact identification tier. Slot extraction, point groups, and
crystallographic orbits are cgd.analyze / cgd.orbit_equivalence_classes,
shared with the CGD path rather than re-implemented.

>>> from autografs.extract_topology import topology_from_tetrahedral
>>> topology = topology_from_tetrahedral(Structure.from_file("FAU.cif"), "FAU")
>>> identify_net(topology_quotient_edges(topology), mofgen.topologies)
['fau']

Interrupted frameworks (terminal OH/F on a T site, IZA dash codes) are
rejected: a T atom with fewer than four bridges has no 4-c vertex, and
the extracted net would not be the framework type.

``topology_from_deconstruction`` is the general path (coverage plan
stage 3): any finite deconstruction's own blueprint, one slot per
building unit at its real position, one shared connection point per
cut bond - the self-templated round trip's template. Unlike the
tetrahedral path it makes no idealization at all: the blueprint IS the
crystal's embedding, so a rebuild against it tests whether rigid
representative units regenerate the material, with the idealized
embedding removed from the question entirely.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.core.structure import Molecule, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from autografs.cgd import analyze, orbit_equivalence_classes
from autografs.exceptions import TopologyExtractionError
from autografs.fragment import Fragment
from autografs.topology import Topology

if TYPE_CHECKING:
    from autografs.deconstruct import Deconstruction

__all__ = [
    "rod_topology_from_deconstruction",
    "topology_from_deconstruction",
    "topology_from_tetrahedral",
]

logger = logging.getLogger(__name__)

# a T-O bond in tetrahedral frameworks is 1.5-1.8 A (Si/Al/P/Ge...);
# the next-nearest contact (O-O at ~2.6 A) is comfortably beyond
T_O_CUTOFF = 2.3


def topology_from_tetrahedral(
    structure: Structure, name: str, cutoff: float = T_O_CUTOFF
) -> Topology:
    """Extract a buildable Topology from a tetrahedral framework.

    Parameters
    ----------
    structure : Structure
        The idealized crystal: T atoms (any non-oxygen species) and
        bridging oxygens, nothing else. Cell and coordinates are used
        as given - IZA idealized CIFs are already maximum-symmetry
        embeddings.
    name : str
        Name of the resulting topology (e.g. the framework type code).
    cutoff : float, optional
        T-O bond cutoff in Angstrom.

    Returns
    -------
    Topology
        Node slots on the T atoms, edge-center slots on the oxygens,
        shared dummies at the T-O midpoints, orbits from spglib.

    Raises
    ------
    TopologyExtractionError
        For non-tetrahedral input: an oxygen not bridging exactly two
        T atoms, or a T atom without exactly four bridges (interrupted
        frameworks, dash-coded in the IZA nomenclature).
    """
    structure = structure.copy()
    structure.remove_oxidation_states()
    t_indices = [i for i, site in enumerate(structure) if site.specie.symbol != "O"]
    o_indices = [i for i, site in enumerate(structure) if site.specie.symbol == "O"]
    if not t_indices or not o_indices:
        raise TopologyExtractionError(
            f"{name}: a tetrahedral framework needs T atoms and bridging "
            f"oxygens; got {len(t_indices)} T and {len(o_indices)} O sites."
        )
    t_set = set(t_indices)

    # every oxygen must bridge exactly two T atoms; collect the bridge
    # geometry (the neighbor coords carry the correct periodic image).
    # Species follow the CGD convention: centers encode their
    # coordination as the atomic number (Z=4 nodes, Z=2 edge centers),
    # dummies are X at the T-O midpoints - the quarter points of the
    # T...T edge, exactly where the CGD parser puts them
    node_species = get_el_sp(4)
    edge_species = get_el_sp(2)
    dummy_species = get_el_sp("X")
    species: list = [node_species] * len(t_indices)
    coords: list[np.ndarray] = [structure[i].coords for i in t_indices]
    t_bridges = dict.fromkeys(t_indices, 0)
    for o_index in o_indices:
        site = structure[o_index]
        neighbors = [
            neighbor
            for neighbor in structure.get_neighbors(site, r=cutoff)
            if neighbor.index in t_set
        ]
        if len(neighbors) != 2:
            raise TopologyExtractionError(
                f"{name}: an oxygen bridges {len(neighbors)} T atom(s) "
                "instead of two - a terminal or over-coordinated oxygen "
                "(interrupted or non-tetrahedral framework)."
            )
        species.append(edge_species)
        coords.append(site.coords)
        for neighbor in neighbors:
            t_bridges[neighbor.index] += 1
            species.append(dummy_species)
            coords.append((site.coords + neighbor.coords) / 2.0)
    under = {i: n for i, n in t_bridges.items() if n != 4}
    if under:
        example = structure[min(under)].specie.symbol
        raise TopologyExtractionError(
            f"{name}: {len(under)} T atom(s) (e.g. {example}) do not carry "
            "exactly four oxygen bridges - an interrupted or "
            "non-tetrahedral framework, which has no 4-coordinated net."
        )

    net = Structure(
        structure.lattice,
        species,
        [structure.lattice.get_fractional_coords(c) for c in coords],
    )
    fragments = analyze(net)
    # center indices inside net: all T first, then one He per oxygen
    # every third site (each O appended as He, X, X)
    n_t = len(t_indices)
    centers = list(range(n_t)) + [n_t + 3 * k for k in range(len(o_indices))]
    equivalence_classes = orbit_equivalence_classes(net, centers)
    try:
        spacegroup = SpacegroupAnalyzer(
            structure, symprec=1e-3
        ).get_space_group_number()
    except Exception:  # noqa: BLE001 - spglib failure is metadata loss only
        spacegroup = None
    logger.info(
        f"\t[x] extracted {name}: {len(t_indices)} T slots, "
        f"{len(o_indices)} edge centers, spacegroup {spacegroup}."
    )
    return Topology(
        name=name,
        slots=fragments,
        cell=net.lattice,
        equivalence_classes=equivalence_classes or None,
        spacegroup_number=spacegroup,
        is_2d=False,
    )


def topologies_from_tetrahedral_cifs(
    paths: dict[str, Path], cutoff: float = T_O_CUTOFF
) -> dict[str, Topology]:
    """Convert a batch of tetrahedral-framework CIFs to Topologies.

    The workhorse behind ``autografs-topologies --use_iza``: each CIF
    (e.g. an IZA idealized SiO2 framework) goes through
    ``topology_from_tetrahedral``; entries that are interrupted,
    non-tetrahedral, or unparseable are skipped with a per-entry
    reason, mirroring ``cgd.read_cgd_data``'s error accounting.

    Parameters
    ----------
    paths : dict[str, Path]
        CIF path per topology name (e.g. official framework codes).
    cutoff : float, optional
        T-O bond cutoff in Angstrom.

    Returns
    -------
    dict[str, Topology]
        The successfully converted entries.
    """
    converted: dict[str, Topology] = {}
    failures: dict[str, str] = {}
    for name in sorted(paths):
        try:
            structure = Structure.from_file(paths[name])
            converted[name] = topology_from_tetrahedral(
                structure, name=name, cutoff=cutoff
            )
        except TopologyExtractionError as exc:
            failures[name] = str(exc)
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    logger.info(
        f"Converted {len(converted)}/{len(paths)} tetrahedral CIFs "
        f"({len(failures)} skipped)."
    )
    for name, reason in sorted(failures.items()):
        logger.info(f"    - {name}: {reason}")
    return converted


def topology_from_deconstruction(
    result: Deconstruction, name: str = "self"
) -> tuple[Topology, dict[int, str]]:
    """The structure's own blueprint, plus its identity slot mapping.

    Erects a Topology directly from a deconstruction's
    ``BlueprintRecipe``: one slot per building unit, centered at the
    unit's real (home-gauge) centroid, with one X dummy per cut bond at
    the real bond midpoint. Each cut's two dummy expressions - the same
    physical point in each end's gauge - share one tag, which is
    exactly the pairing convention ``alignment.prepare_build`` consumes,
    and their fractional difference is the cut's integer voltage by
    construction.

    Slot center species encode the connectivity (Z = number of
    connections, the CGD ``NODE`` convention). The spacegroup is set to
    1 (triclinic): a real crystal's symmetry is approximate, and P1
    gives the cell optimizer full freedom rather than a constraint the
    embedding may not satisfy; crystallographic orbits are attempted on
    the erected net and fall back to one class per slot.

    Returns ``(topology, mapping)`` where ``mapping`` is the identity
    assignment ``{slot index: fragment name}`` - each slot takes its
    own unit's (deduplicated, representative) fragment.

    Raises
    ------
    TopologyExtractionError
        When the deconstruction has no blueprint recipe (rod-containing
        structures) or a unit exceeds the library's connectivity range.
    """
    recipe = result.blueprint
    if recipe is None or result.rod_units:
        raise TopologyExtractionError(
            "Rod-containing deconstructions need a slot-run blueprint "
            "(the recipe carries the rod's points of extension in "
            "rod_poe; the run-consuming constructor is the next "
            "increment of the coverage plan)."
        )
    lattice = result.structure.lattice
    per_unit_cuts: dict[int, list[tuple[int, tuple]]] = {}
    for cut_index, (unit_a, unit_b, mid_a, mid_b) in enumerate(recipe.cuts):
        per_unit_cuts.setdefault(unit_a, []).append((cut_index, mid_a))
        per_unit_cuts.setdefault(unit_b, []).append((cut_index, mid_b))

    slots: list[Fragment] = []
    mapping: dict[int, str] = {}
    for k, unit in enumerate(result.units):
        connections = per_unit_cuts.get(k, [])
        if not connections:
            raise TopologyExtractionError(
                f"Unit {k} ({unit.name}) carries no cut bond; a "
                "disconnected unit has no slot."
            )
        if len(connections) > 118:
            raise TopologyExtractionError(
                f"Unit {k} carries {len(connections)} connections; no "
                "element encodes that connectivity."
            )
        center = np.asarray(recipe.centers[k], dtype=float)
        species = [get_el_sp(len(connections))] + ["X"] * len(connections)
        frac_coords = [center] + [
            np.asarray(mid, dtype=float) for _index, mid in connections
        ]
        carts = [lattice.get_cartesian_coords(fc) for fc in frac_coords]
        tags = [0] + [cut_index + 1 for cut_index, _mid in connections]
        molecule = Molecule(species, carts, site_properties={"tags": tags})
        slots.append(Fragment(atoms=molecule, name=f"slot_{k}"))
        mapping[k] = unit.name

    # orbits on the erected net, so symmetric self-templates group
    # their slots; a distorted P1 crystal degrades gracefully to one
    # class per slot
    all_species: list = []
    all_frac: list[np.ndarray] = []
    for slot in slots:
        for site in slot.atoms:
            all_species.append(site.specie)
            all_frac.append(lattice.get_fractional_coords(site.coords))
    try:
        erected = Structure(
            lattice, all_species, np.array(all_frac), coords_are_cartesian=False
        )
        centers_idx = []
        offset = 0
        for slot in slots:
            centers_idx.append(offset)
            offset += len(slot.atoms)
        classes = orbit_equivalence_classes(erected, centers_idx)
    except Exception:  # noqa: BLE001 - orbits are an optimization only
        classes = []
    if len(classes) != len(slots):
        classes = list(range(len(slots)))

    return (
        Topology(
            name=name,
            slots=slots,
            cell=lattice,
            equivalence_classes=classes,
            spacegroup_number=1,
            is_2d=False,
        ),
        mapping,
    )


def rod_topology_from_deconstruction(result: Deconstruction, name: str = "self-rod"):
    """The rod structure's own blueprint, its run, and its mappings.

    The slot-run counterpart of ``topology_from_deconstruction``
    (coverage plan, rod self-templates): the single rod's points of
    extension become node slots along the crystal's own axis, finite
    units become lateral slots, and the run handed back is built
    directly from the rod's measured geometry - no run detection, no
    library. ``build_rod_framework(topology, rod_fragment, laterals,
    run=run)`` then reuses the validated rod builder unchanged.

    Returns
    -------
    tuple
        ``(topology, run, lateral_mapping)`` where ``run`` is a
        ``SlotRun`` (screwless rod) or ``HelicalRun`` (its screw from
        the rod's own canonical repeat) and ``lateral_mapping`` maps
        lateral slot index -> fragment name (the identity assignment).

    Raises
    ------
    TopologyExtractionError
        For structures without exactly one rod and one periodic
        component (the multi-rod and catenated-rod cases await their
        own increment), or when a cut cannot be attributed to a point
        of extension unambiguously.
    """
    from autografs.net import HelicalRun, SlotRun
    from autografs.rods import rod_fragment as _rod_fragment

    recipe = result.blueprint
    if recipe is None or not recipe.rod_poe:
        raise TopologyExtractionError("No rod recipe on this deconstruction.")
    if len(result.rod_units) != 1 or result.n_periodic_components != 1:
        raise TopologyExtractionError(
            "Rod self-templates currently take exactly one rod and one "
            "periodic component."
        )
    (rod_index, poe_entries), *_ = recipe.rod_poe.items()
    rod_unit = result.rod_units[0]
    lattice = result.structure.lattice

    # attribute each rod-end cut to its point of extension: the
    # midpoint sits half a bond from its own anchor and full bond
    # lengths from any other PoE atom, so nearest-position wins; an
    # ambiguous case is refused, not guessed
    poe_carts = {
        atom: lattice.get_cartesian_coords(np.asarray(pos, dtype=float))
        for atom, pos in poe_entries
    }
    unit_is_rod = [center is None for center in recipe.centers]

    def poe_of(mid_frac: tuple) -> int:
        mid = lattice.get_cartesian_coords(np.asarray(mid_frac, dtype=float))
        ranked = sorted(
            poe_carts, key=lambda atom: float(np.linalg.norm(poe_carts[atom] - mid))
        )
        if len(ranked) > 1:
            best = float(np.linalg.norm(poe_carts[ranked[0]] - mid))
            runner = float(np.linalg.norm(poe_carts[ranked[1]] - mid))
            if runner - best < 0.2:
                raise TopologyExtractionError(
                    f"Ambiguous cut-to-PoE attribution ({best:.2f} vs {runner:.2f} A)."
                )
        return ranked[0]

    poe_cuts: dict[int, list[tuple[int, tuple]]] = {atom: [] for atom, _ in poe_entries}
    lateral_cuts: dict[int, list[tuple[int, tuple]]] = {}
    for cut_index, (unit_a, unit_b, mid_a, mid_b) in enumerate(recipe.cuts):
        for unit, mid in ((unit_a, mid_a), (unit_b, mid_b)):
            if unit_is_rod[unit]:
                poe_cuts[poe_of(mid)].append((cut_index, mid))
            else:
                lateral_cuts.setdefault(unit, []).append((cut_index, mid))

    slots: list[Fragment] = []
    lateral_mapping: dict[int, str] = {}
    # recipe indices follow the RAW unit partition (rod included);
    # result.units is the compacted finite list in the same order
    finite_raw = [index for index, is_rod in enumerate(unit_is_rod) if not is_rod]
    for k, building_unit in zip(finite_raw, result.units, strict=True):
        connections = lateral_cuts.get(k, [])
        if not connections:
            raise TopologyExtractionError(
                f"Lateral unit {k} ({building_unit.name}) carries no cut bond."
            )
        center = np.asarray(recipe.centers[k], dtype=float)
        species = [get_el_sp(len(connections))] + ["X"] * len(connections)
        frac_coords = [center] + [
            np.asarray(mid, dtype=float) for _index, mid in connections
        ]
        carts = [lattice.get_cartesian_coords(fc) for fc in frac_coords]
        tags = [0] + [cut_index + 1 for cut_index, _mid in connections]
        slots.append(
            Fragment(
                atoms=Molecule(species, carts, site_properties={"tags": tags}),
                name=f"slot_{k}",
            )
        )
        lateral_mapping[len(slots) - 1] = building_unit.name

    run_slots: list[int] = []
    for atom, pos in poe_entries:
        connections = poe_cuts[atom]
        if not connections:
            raise TopologyExtractionError(f"Point of extension {atom} carries no cut.")
        center = lattice.get_cartesian_coords(np.asarray(pos, dtype=float))
        species = [get_el_sp(len(connections))] + ["X"] * len(connections)
        carts = [center] + [
            lattice.get_cartesian_coords(np.asarray(mid, dtype=float))
            for _index, mid in connections
        ]
        tags = [0] + [cut_index + 1 for cut_index, _mid in connections]
        slots.append(
            Fragment(
                atoms=Molecule(species, carts, site_properties={"tags": tags}),
                name=f"poe_{atom}",
            )
        )
        run_slots.append(len(slots) - 1)

    topology = Topology(
        name=name,
        slots=slots,
        cell=lattice,
        equivalence_classes=list(range(len(slots))),
        spacegroup_number=1,
        is_2d=False,
    )

    fragment = _rod_fragment(result.structure, rod_unit, name="self_rod")
    direction = (
        int(rod_unit.generator[0]),
        int(rod_unit.generator[1]),
        int(rod_unit.generator[2]),
    )
    period = float(rod_unit.repeat_length)
    if fragment.repeat.screw_order > 1:
        axis_point = np.mean([poe_carts[atom] for atom, _ in poe_entries], axis=0)
        run: SlotRun | HelicalRun = HelicalRun(
            direction=direction,
            slots=tuple(run_slots),
            period=period,
            screw_angle=float(fragment.repeat.screw_angle),
            screw_order=int(fragment.repeat.screw_order),
            axis_point=(
                float(axis_point[0]),
                float(axis_point[1]),
                float(axis_point[2]),
            ),
        )
    else:
        run = SlotRun(direction=direction, slots=tuple(run_slots), period=period)
    return topology, run, lateral_mapping, fragment
