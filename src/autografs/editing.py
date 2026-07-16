"""
Post-build editing of Framework objects.

The 2.x line could keep editing a structure after it was built:
rotating a placed linker, carving statistical defects out of a
supercell, grafting functional groups onto the framework itself. This
module reimplements those operations on the 3.x bond graph.

Everything here works on the per-atom provenance recorded at build
time (node attributes ``slot`` and ``sbu``: which placed SBU an atom
belongs to) plus the tag convention (a positive ``tag`` marks an
anchor atom, one half of an inter-SBU bond). All operations return a
new Framework and leave the input untouched.

Geometry conventions: node coordinates are unwrapped cartesian, so a
bond's periodic image shift is recoverable by minimum-image rounding
of the fractional endpoint difference - the supercell and defect code
rely on that to keep boundary-crossing bonds exact.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable

import networkx
import numpy as np
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import EconNN
from pymatgen.core.bonds import get_bond_length, get_bond_order
from pymatgen.core.structure import FunctionalGroups, Molecule

import autografs.utils
from autografs.framework import Framework

__all__ = [
    "rotate_sbu",
    "flip_sbu",
    "make_supercell",
    "make_defects",
    "functionalizable_sites",
    "functionalize",
]

logger = logging.getLogger(__name__)

# maximum out-of-plane deviation (Angstrom) for the anchors of a
# >2-connected SBU to count as coplanar (mirrorable through their plane)
FLIP_PLANARITY_TOLERANCE = 0.01

# terminal atoms bonded above this order are not functionalizable
# (replacing one end of a double bond is not a substitution)
_SINGLE_BOND_MAX_ORDER = 1.2


# ----------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------


def _require_provenance(framework: Framework) -> None:
    """Editing needs to know which atoms belong to which placed SBU."""
    if any("slot" not in data for _, data in framework.graph.nodes(data=True)):
        raise ValueError(
            "The framework graph carries no slot provenance; post-build "
            "editing needs the per-atom 'slot'/'sbu' attributes recorded "
            "by builds of autografs >= 3.1. Rebuild the framework to edit it."
        )


def _copy_graph(graph: networkx.Graph) -> networkx.Graph:
    """Copy a framework graph with private node dicts and coord arrays."""
    new = networkx.Graph(**graph.graph)
    for node, data in graph.nodes(data=True):
        copied = dict(data)
        copied["coord"] = np.array(data["coord"], dtype=float)
        new.add_node(node, **copied)
    new.add_edges_from((u, v, dict(d)) for u, v, d in graph.edges(data=True))
    return new


def _relabel(graph: networkx.Graph) -> networkx.Graph:
    """Relabel nodes to contiguous 0..n-1, preserving sorted order.

    Framework views index the pymatgen Structure by sorted node id, so
    every editing operation must hand back a gap-free graph.
    """
    mapping = {old: new for new, old in enumerate(sorted(graph))}
    return networkx.relabel_nodes(graph, mapping, copy=True)


def _slot_nodes(framework: Framework, slot: int) -> list[int]:
    """Node ids of one placed SBU, with a helpful error when unknown."""
    _require_provenance(framework)
    nodes = [
        node for node, data in framework.graph.nodes(data=True) if data["slot"] == slot
    ]
    if not nodes:
        raise ValueError(
            f"No placed SBU with slot index {slot}; available slots: "
            f"{autografs.utils.format_mappings(framework.slots)}"
        )
    return nodes


def _anchor_nodes(framework: Framework, nodes: list[int]) -> list[int]:
    """The connection (anchor) atoms among an SBU's nodes: tag > 0."""
    return [n for n in nodes if framework.graph.nodes[n].get("tag", 0) > 0]


def _axis_rotation(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation matrix around a unit axis."""
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.asarray(
        np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)
    )


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Proper rotation taking unit vector source onto unit vector target."""
    cosine = float(source @ target)
    axis = np.cross(source, target)
    sine = float(np.linalg.norm(axis))
    if sine < 1e-12:
        if cosine > 0.0:
            return np.eye(3)
        # antiparallel: rotate half a turn around any perpendicular
        reference = (
            np.array([1.0, 0.0, 0.0])
            if abs(source[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        perpendicular = np.cross(source, reference)
        perpendicular /= np.linalg.norm(perpendicular)
        return _axis_rotation(perpendicular, np.pi)
    return _axis_rotation(axis / sine, float(np.arctan2(sine, cosine)))


def _bond_image_shift(
    coord_a: np.ndarray, coord_b: np.ndarray, inv_cell: np.ndarray
) -> np.ndarray:
    """Integer image shift s such that coord_b + s @ cell bonds coord_a.

    Coordinates are unwrapped cartesian: within a fragment the shift is
    zero, and for a tag-pair bond through a boundary the fractional
    difference sits near an integer (the bond length is small against
    the cell), so minimum-image rounding recovers the crossing exactly.
    """
    return np.asarray(np.round((coord_a - coord_b) @ inv_cell).astype(int))


def _cap_ufftype(symbol: str) -> str:
    """UFF4MOF type for a monovalent capping element."""
    prefix = symbol if len(symbol) == 2 else f"{symbol}_"
    candidates = autografs.utils._uff_types_for_prefix(prefix)
    if not candidates:
        raise ValueError(f"No UFF4MOF type for capping element {symbol!r}.")
    return min(candidates, key=lambda t: t.coordination).symbol


# ----------------------------------------------------------------------
# rotation / flipping of placed SBUs
# ----------------------------------------------------------------------


def rotate_sbu(framework: Framework, slot: int, theta: float) -> Framework:
    """Rotate a placed 2-connected SBU around its anchor-anchor axis.

    See Framework.rotate for the user-facing documentation.
    """
    nodes = _slot_nodes(framework, slot)
    anchors = _anchor_nodes(framework, nodes)
    if len(anchors) != 2:
        raise ValueError(
            f"Slot {slot} ({framework.graph.nodes[nodes[0]]['sbu']}) has "
            f"{len(anchors)} connection points; rotation around a bond "
            "axis needs exactly 2. Use flip() for planar-anchored SBUs."
        )
    graph = _copy_graph(framework.graph)
    origin = np.asarray(graph.nodes[anchors[0]]["coord"], dtype=float)
    axis = np.asarray(graph.nodes[anchors[1]]["coord"], dtype=float) - origin
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise ValueError(f"Slot {slot} has coincident anchors; no rotation axis.")
    rotation = _axis_rotation(axis / norm, theta)
    for node in nodes:
        coord = graph.nodes[node]["coord"]
        graph.nodes[node]["coord"] = (coord - origin) @ rotation.T + origin
    return Framework(graph, name=framework.name)


def flip_sbu(framework: Framework, slot: int) -> Framework:
    """Mirror a placed SBU through a plane that fixes its anchors.

    See Framework.flip for the user-facing documentation.
    """
    nodes = _slot_nodes(framework, slot)
    anchors = _anchor_nodes(framework, nodes)
    coords = np.array([framework.graph.nodes[n]["coord"] for n in nodes], dtype=float)
    if len(anchors) < 2:
        raise ValueError(
            f"Slot {slot} has {len(anchors)} connection points; flipping "
            "needs at least 2 to define a mirror that preserves them."
        )
    anchor_coords = np.array(
        [framework.graph.nodes[n]["coord"] for n in anchors], dtype=float
    )
    if len(anchors) == 2:
        origin = anchor_coords[0]
        axis = anchor_coords[1] - origin
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            raise ValueError(f"Slot {slot} has coincident anchors; no mirror axis.")
        axis /= norm
        # mirror through the plane containing the axis and the atom
        # farthest from it: deterministic, and holds that atom fixed
        offsets = coords - origin
        perpendicular = offsets - np.outer(offsets @ axis, axis)
        distances = np.linalg.norm(perpendicular, axis=1)
        farthest = int(np.argmax(distances))
        if distances[farthest] < 1e-6:
            raise ValueError(
                f"Slot {slot} is strictly linear; flipping it is the identity."
            )
        normal = np.cross(axis, perpendicular[farthest])
        normal /= np.linalg.norm(normal)
    else:
        # >2 anchors: mirror through their common plane, if they have one
        origin = anchor_coords.mean(axis=0)
        _, singular, vt = np.linalg.svd(anchor_coords - origin)
        normal = vt[2]
        deviation = float(np.abs((anchor_coords - origin) @ normal).max())
        if deviation > FLIP_PLANARITY_TOLERANCE:
            raise ValueError(
                f"Slot {slot} anchors are not coplanar (deviation "
                f"{deviation:.3f} A); no anchor-preserving mirror exists."
            )
    mirror = np.eye(3) - 2.0 * np.outer(normal, normal)
    graph = _copy_graph(framework.graph)
    for node, coord in zip(nodes, coords, strict=True):
        graph.nodes[node]["coord"] = (coord - origin) @ mirror + origin
    return Framework(graph, name=framework.name)


# ----------------------------------------------------------------------
# interpenetration / catenation
# ----------------------------------------------------------------------

# high-symmetry displacement candidates tried by offset="auto": body
# center first (the classic dia-c / pcu-c catenation vector), then
# face and edge centers
_CATENATION_CANDIDATES: tuple[tuple[float, float, float], ...] = (
    (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.0, 0.5),
)


def _min_internet_contact(framework: Framework, n_atoms: int, cutoff: float) -> float:
    """Smallest distance between atoms of different interpenetrated nets.

    Copies are blocks of n_atoms consecutive node ids, so the net of
    an atom is its id // n_atoms.
    """
    import math

    centers, points, _, distances = framework.structure.get_neighbor_list(r=cutoff)
    if len(distances) == 0:
        return math.inf
    different_net = (centers // n_atoms) != (points // n_atoms)
    if not different_net.any():
        return math.inf
    return float(distances[different_net].min())


def interpenetrate(
    framework: Framework,
    n: int = 2,
    offset: tuple[float, float, float] | str = "auto",
) -> Framework:
    """Generate an n-fold interpenetrated (catenated) framework.

    See Framework.interpenetrate for the user-facing documentation.
    """
    if n < 2:
        raise ValueError(f"Interpenetration needs n >= 2 nets, got {n}.")
    if isinstance(offset, str):
        if offset != "auto":
            raise ValueError(
                f"offset must be a fractional 3-vector or 'auto', got {offset!r}."
            )
        candidates = _CATENATION_CANDIDATES
    else:
        candidates = (tuple(float(x) for x in offset),)  # type: ignore[assignment]
        if len(candidates[0]) != 3:
            raise ValueError(
                f"offset must have three fractional components, got {offset!r}."
            )
    n_atoms = len(framework.graph)
    best: tuple[float, Framework] | None = None
    for candidate in candidates:
        catenated = _displaced_copies(framework, n, candidate)
        contact = _min_internet_contact(catenated, n_atoms, cutoff=6.0)
        if best is None or contact > best[0]:
            best = (contact, catenated)
    assert best is not None  # candidates is never empty
    contact, catenated = best
    logger.info(
        f"Interpenetrated {framework.name!r} {n}-fold; closest inter-net "
        f"contact {contact:.2f} A."
    )
    return catenated


def replicated_graph(
    framework: Framework,
    shifts: Iterable[np.ndarray],
    cell: np.ndarray | None = None,
    copy_edges: bool = True,
) -> networkx.Graph:
    """Disjoint displaced copies of a framework's bond graph.

    Copy k is displaced by ``shifts[k]`` (cartesian) and its node ids
    offset by ``k * n_atoms``. The tag-uniqueness and slot-provenance
    invariants live here, and only here: positive anchor tags shift by
    ``k * tag_base`` and slot ids by ``k * slot_base``, so tag-pair
    semantics and placed-SBU identity survive replication the same way
    for supercells, interpenetration, and layer stacking.

    Parameters
    ----------
    framework : Framework
        Source framework; unchanged.
    shifts : iterable of np.ndarray
        One cartesian displacement per copy.
    cell : np.ndarray or None, optional
        Cell matrix of the combined graph; the source cell when None.
    copy_edges : bool, optional
        Copy each bond within its own copy (default). Callers that
        rewire bonds across copies (supercells) pass False and add
        the edges themselves.

    Returns
    -------
    networkx.Graph
        The combined graph, ready to wrap in a Framework.
    """
    graph = framework.graph
    n_atoms = len(graph)
    combined = networkx.Graph(cell=framework.cell.copy() if cell is None else cell)
    tag_base = max((d["tag"] for _, d in graph.nodes(data=True)), default=0)
    slot_base = (
        max((d.get("slot", 0) for _, d in graph.nodes(data=True)), default=0) + 1
    )
    for k, shift in enumerate(shifts):
        for node, data in graph.nodes(data=True):
            copied = dict(data)
            copied["coord"] = np.asarray(data["coord"], dtype=float) + shift
            if copied["tag"] > 0:
                copied["tag"] += k * tag_base
            if "slot" in copied:
                copied["slot"] += k * slot_base
            combined.add_node(node + k * n_atoms, **copied)
        if copy_edges:
            for i, j, data in graph.edges(data=True):
                combined.add_edge(i + k * n_atoms, j + k * n_atoms, **dict(data))
    return combined


def _displaced_copies(
    framework: Framework, n: int, offset: tuple[float, float, float]
) -> Framework:
    """n copies of the framework, copy k displaced by k * offset."""
    cell = framework.cell
    shifts = [(k * np.asarray(offset, dtype=float)) @ cell for k in range(n)]
    combined = replicated_graph(framework, shifts)
    return Framework(combined, name=f"{framework.name}_cat{n}")


# ----------------------------------------------------------------------
# supercells
# ----------------------------------------------------------------------


def make_supercell(
    framework: Framework, scale: int | tuple[int, int, int]
) -> Framework:
    """Replicate a framework into an exact supercell graph.

    See Framework.supercell for the user-facing documentation.
    """
    _require_provenance(framework)
    if isinstance(scale, int):
        scale = (scale, scale, scale)
    multipliers = tuple(int(m) for m in scale)
    if len(multipliers) != 3 or any(m < 1 for m in multipliers):
        raise ValueError(f"Supercell multipliers must be positive ints, got {scale}.")
    graph = framework.graph
    cell = framework.cell
    inv_cell = np.linalg.inv(cell)
    n_atoms = graph.number_of_nodes()
    images = list(itertools.product(*(range(m) for m in multipliers)))
    image_index = {image: k for k, image in enumerate(images)}
    new_cell = np.asarray(multipliers, dtype=float)[:, None] * cell

    tag_base = max((d["tag"] for _, d in graph.nodes(data=True)), default=0)

    # nodes: one copy per image (edges are rewired across copies below,
    # and every replicated tag is reassigned pairwise below)
    shifts = [np.asarray(image, dtype=float) @ cell for image in images]
    supercell = replicated_graph(framework, shifts, cell=new_cell, copy_edges=False)

    # edges: a bond from u to v's image at shift s maps, for u placed in
    # image m, onto v placed in image (m + s) mod multipliers - bonds
    # that crossed the old boundary now run to the neighboring copy
    for u, v, data in graph.edges(data=True):
        shift = _bond_image_shift(
            graph.nodes[u]["coord"], graph.nodes[v]["coord"], inv_cell
        )
        for k, image in enumerate(images):
            partner = tuple((np.asarray(image) + shift) % multipliers)
            supercell.add_edge(
                u + k * n_atoms,
                v + image_index[partner] * n_atoms,
                **dict(data),
            )

    # tags: each original anchor pair yields one pair per image of its
    # first member, tagged uniquely; unpaired tags replicate per image
    nodes_by_tag: dict[int, list[int]] = {}
    for node, data in graph.nodes(data=True):
        if data["tag"] > 0:
            nodes_by_tag.setdefault(data["tag"], []).append(node)
    for tag, tagged in sorted(nodes_by_tag.items()):
        if len(tagged) == 2:
            a, b = tagged
            shift = _bond_image_shift(
                graph.nodes[a]["coord"], graph.nodes[b]["coord"], inv_cell
            )
            for k, image in enumerate(images):
                partner = tuple((np.asarray(image) + shift) % multipliers)
                new_tag = tag + k * tag_base
                supercell.nodes[a + k * n_atoms]["tag"] = new_tag
                supercell.nodes[b + image_index[partner] * n_atoms]["tag"] = new_tag
        else:
            for node in tagged:
                for k in range(len(images)):
                    supercell.nodes[node + k * n_atoms]["tag"] = tag + k * tag_base

    name = f"{framework.name}_{multipliers[0]}x{multipliers[1]}x{multipliers[2]}"
    return Framework(supercell, name=name)


# ----------------------------------------------------------------------
# statistical defects
# ----------------------------------------------------------------------


def make_defects(
    framework: Framework,
    fraction: float | None = None,
    slots: Iterable[int] | None = None,
    sbu: str | None = None,
    cap: str | None = "H",
    seed: int | None = None,
) -> Framework:
    """Remove whole placed SBUs and cap the dangling anchors.

    See Framework.defects for the user-facing documentation.
    """
    _require_provenance(framework)
    placed = framework.slots
    if (fraction is None) == (slots is None):
        raise ValueError("Give exactly one of 'fraction' and 'slots'.")
    if slots is not None:
        if sbu is not None:
            raise ValueError("The 'sbu' filter only applies to fraction sampling.")
        to_remove = sorted(set(slots))
        unknown = [s for s in to_remove if s not in placed]
        if unknown:
            raise ValueError(f"Unknown slot indices: {unknown}.")
    else:
        assert fraction is not None
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be within [0, 1], got {fraction}.")
        candidates = [s for s, name in placed.items() if sbu is None or name == sbu]
        if not candidates:
            raise ValueError(
                f"No placed SBU named {sbu!r}; present: {sorted(set(placed.values()))}."
            )
        count = round(fraction * len(candidates))
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(candidates), size=count, replace=False)
        to_remove = sorted(candidates[int(i)] for i in chosen)
    if len(to_remove) == len(placed):
        raise ValueError("Refusing to remove every placed SBU.")
    if not to_remove:
        logger.warning("Defect selection is empty; returning an unchanged copy.")

    graph = _copy_graph(framework.graph)
    removed_slots = set(to_remove)
    removed_nodes = {
        node for node, data in graph.nodes(data=True) if data["slot"] in removed_slots
    }
    # cap every surviving anchor whose partner disappears, along the
    # direction of the removed bond (minimum image, so a defect on the
    # far side of the boundary caps in the right direction)
    inv_cell = np.linalg.inv(framework.cell)
    next_node = max(graph) + 1
    if cap is not None:
        cap_ufftype = _cap_ufftype(cap)
        for survivor, removed in list(graph.edges(removed_nodes)):
            if survivor in removed_nodes:
                survivor, removed = removed, survivor
            if survivor in removed_nodes:
                continue  # bond between two removed SBUs
            coord = np.asarray(graph.nodes[survivor]["coord"], dtype=float)
            partner = np.asarray(graph.nodes[removed]["coord"], dtype=float)
            shift = _bond_image_shift(coord, partner, inv_cell)
            direction = partner + shift @ framework.cell - coord
            direction /= np.linalg.norm(direction)
            length = float(
                get_bond_length(graph.nodes[survivor]["symbol"], cap, bond_order=1)
            )
            graph.add_node(
                next_node,
                symbol=cap,
                coord=coord + length * direction,
                tag=0,
                ufftype=cap_ufftype,
                slot=graph.nodes[survivor]["slot"],
                sbu=graph.nodes[survivor]["sbu"],
            )
            graph.add_edge(survivor, next_node, bond_order=1.0)
            next_node += 1
    graph.remove_nodes_from(removed_nodes)
    # anchors left unpaired (their partner is gone) lose their tag
    surviving_tags: dict[int, int] = {}
    for _, data in graph.nodes(data=True):
        if data["tag"] > 0:
            surviving_tags[data["tag"]] = surviving_tags.get(data["tag"], 0) + 1
    for _, data in graph.nodes(data=True):
        if data["tag"] > 0 and surviving_tags[data["tag"]] == 1:
            data["tag"] = 0
    return Framework(_relabel(graph), name=f"{framework.name}_defective")


# ----------------------------------------------------------------------
# post-build functionalization
# ----------------------------------------------------------------------


def functionalizable_sites(
    framework: Framework, symbol: str = "H", sbu: str | None = None
) -> list[int]:
    """Terminal, single-bonded, non-anchor atoms of the given element.

    See Framework.functionalizable_sites for the user-facing docs.
    """
    graph = framework.graph
    sites = []
    for node, data in graph.nodes(data=True):
        if data["symbol"] != symbol:
            continue
        if sbu is not None and data.get("sbu") != sbu:
            continue
        if data.get("tag", 0) > 0:
            continue
        neighbors = list(graph.neighbors(node))
        if len(neighbors) != 1:
            continue
        order = graph.edges[node, neighbors[0]].get("bond_order", 1.0)
        if order > _SINGLE_BOND_MAX_ORDER:
            continue
        sites.append(node)
    return sorted(sites)


def _load_functional_group(
    functional_group: str | Molecule,
) -> tuple[Molecule, int, int]:
    """The group molecule plus its dummy and head (attachment) indices."""
    if isinstance(functional_group, str):
        try:
            group = FunctionalGroups[functional_group].copy()
        except KeyError:
            raise ValueError(
                f"Unknown functional group {functional_group!r}; available: "
                f"{sorted(FunctionalGroups)} (or pass a Molecule with one 'X')."
            ) from None
    else:
        group = functional_group.copy()
    dummies = list(group.indices_from_symbol("X"))
    if len(dummies) != 1:
        raise ValueError(
            f"A functional group needs exactly one dummy attachment point "
            f"('X'); got {len(dummies)}."
        )
    dummy = dummies[0]
    real = [i for i in range(len(group)) if i != dummy]
    if not real:
        raise ValueError("A functional group needs at least one real atom.")
    coords = np.asarray(group.cart_coords)
    distances = np.linalg.norm(coords[real] - coords[dummy], axis=1)
    head = real[int(np.argmin(distances))]
    return group, dummy, head


def functionalize(
    framework: Framework,
    index: int | Iterable[int],
    functional_group: str | Molecule,
) -> Framework:
    """Replace terminal atoms of the framework with a functional group.

    See Framework.functionalize for the user-facing documentation.
    """
    _require_provenance(framework)
    indices = [index] if isinstance(index, int) else sorted(set(index))
    if not indices:
        raise ValueError("No site indices given.")
    group, dummy, head = _load_functional_group(functional_group)
    group_coords = np.asarray(group.cart_coords)
    real_indices = [i for i in range(len(group)) if i != dummy]
    head_symbol = group[head].specie.symbol
    # direction the group's head-from-dummy axis must be rotated onto:
    # the dummy marks where the parent atom sits
    group_axis = group_coords[head] - group_coords[dummy]
    group_axis /= np.linalg.norm(group_axis)

    graph = _copy_graph(framework.graph)
    # the grafted local environment is congruent for every site with
    # the same parent element (the group is rigid and the head sits at
    # the tabulated bond length), so UFF types and intra-group bond
    # orders are computed once per parent element, not once per site
    typing_cache: dict[str, tuple[list[str], list[tuple[int, int, float]]]] = {}
    for site in indices:
        if site not in graph:
            raise ValueError(f"No atom with index {site}.")
        data = graph.nodes[site]
        neighbors = list(graph.neighbors(site))
        if data.get("tag", 0) > 0:
            raise ValueError(
                f"Atom {site} is a framework connection point; replacing it "
                "would break inter-SBU bonding."
            )
        if len(neighbors) != 1:
            raise ValueError(
                f"Atom {site} ({data['symbol']}) has {len(neighbors)} bonds; "
                "only terminal (single-bonded) atoms can be functionalized."
            )
        parent = neighbors[0]
        parent_data = graph.nodes[parent]
        parent_symbol = parent_data["symbol"]
        parent_coord = np.asarray(parent_data["coord"], dtype=float)
        direction = np.asarray(data["coord"], dtype=float) - parent_coord
        direction /= np.linalg.norm(direction)
        rotation = _rotation_between(group_axis, direction)
        length = float(get_bond_length(parent_symbol, head_symbol, bond_order=1))
        head_position = parent_coord + length * direction
        placed = (
            group_coords[real_indices] - group_coords[head]
        ) @ rotation.T + head_position

        if parent_symbol not in typing_cache:
            # type the new atoms in their local environment (parent
            # included so the head's coordination is right); bond
            # orders as at build
            local = Molecule(
                [parent_symbol] + [group[i].specie.symbol for i in real_indices],
                np.vstack([parent_coord, placed]),
            )
            strategy = EconNN(
                tol=autografs.utils.BOND_TOLERANCE,
                use_fictive_radius=True,
                cutoff=autografs.utils.BOND_CUTOFF,
            )
            molgraph = MoleculeGraph.from_local_env_strategy(local, strategy=strategy)
            uff_lib, uff_symbs = autografs.utils.load_uff_lib(local)
            mmtypes = autografs.utils.find_mmtypes(
                molgraph=molgraph, uff_lib=uff_lib, uff_symbs=uff_symbs
            )
            cutoffs = autografs.utils.find_element_cutoffs(uff_lib, uff_symbs)
            local_symbols = [s.specie.symbol for s in local]
            group_bonds: list[tuple[int, int, float]] = []
            for i in range(len(local)):
                for connected in molgraph.get_connected_sites(i):
                    j, distance = connected.index, connected.dist
                    if i == 0 or j == 0 or j <= i:
                        continue  # parent bond handled below; count each once
                    order = get_bond_order(
                        local_symbols[i],
                        local_symbols[j],
                        distance,
                        tol=0.2,
                        default_bl=cutoffs[(local_symbols[i], local_symbols[j])],
                    )
                    group_bonds.append((i, j, float(order)))
            typing_cache[parent_symbol] = (mmtypes, group_bonds)
        mmtypes, group_bonds = typing_cache[parent_symbol]

        graph.remove_node(site)
        base = max(graph) + 1
        for k, group_index in enumerate(real_indices):
            graph.add_node(
                base + k,
                symbol=group[group_index].specie.symbol,
                coord=placed[k],
                tag=0,
                ufftype=mmtypes[k + 1],
                slot=parent_data["slot"],
                sbu=parent_data["sbu"],
            )
        head_node = base + real_indices.index(head)
        graph.add_edge(parent, head_node, bond_order=1.0)
        for i, j, order in group_bonds:
            graph.add_edge(base + i - 1, base + j - 1, bond_order=order)
    return Framework(_relabel(graph), name=framework.name)
