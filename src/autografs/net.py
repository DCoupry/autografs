"""
Net verification: does a built framework realize its blueprint?

The build gates check geometry (``max_rmsd``) and overlap
(``min_distance``), but neither verifies that the output actually
realizes the requested topology - mis-paired tags or a bond through
the wrong periodic image would pass both while building a different
net. This module closes that gap by comparing labeled quotient graphs.

Both a Topology and a built Framework reduce to the same object: one
node per slot, and one edge per inter-slot bond carrying an integer
*voltage* (the periodic image offset between the two slots' home-cell
representatives). Because every built atom records which blueprint
slot it fills (the ``slot`` provenance attribute), the two quotient
graphs share their node labels, and verification is an exact multiset
comparison of canonicalized edges - no graph isomorphism search.

>>> mof = mofgen.build(topology, mappings, verify_net=True)   # gated
>>> mof.verify_net(topology)                                  # explicit
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from autografs.exceptions import NetMismatchError

if TYPE_CHECKING:
    from autografs.framework import Framework
    from autografs.topology import Topology

__all__ = [
    "topology_quotient_edges",
    "framework_quotient_edges",
    "verify_net",
]

logger = logging.getLogger(__name__)

# fractional coordinates are rounded to this many decimals before the
# integer/fractional split, so a center sitting a float-noise away
# from a cell boundary wraps identically on both sides
_WRAP_DECIMALS = 6

Edge = tuple[int, int, tuple[int, int, int]]


def _split_image(frac: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split fractional coords into integer image and wrapped remainder."""
    stabilized = np.round(np.asarray(frac, dtype=float), _WRAP_DECIMALS)
    image = np.floor(stabilized)
    return image.astype(int), stabilized - image


def _canonical(slot_a: int, slot_b: int, voltage: np.ndarray) -> Edge:
    """Direction-independent form of one quotient edge.

    An edge from a to b with voltage v equals the edge from b to a
    with voltage -v; self-edges (a slot bonded to its own image)
    canonicalize the voltage sign lexicographically.
    """
    v0, v1, v2 = (int(x) for x in voltage)
    v = (v0, v1, v2)
    negated = (-v0, -v1, -v2)
    if slot_a < slot_b:
        return (slot_a, slot_b, v)
    if slot_b < slot_a:
        return (slot_b, slot_a, negated)
    return (slot_a, slot_a, max(v, negated))


def _require_provenance(framework: Framework) -> None:
    if any("slot" not in data for _, data in framework.graph.nodes(data=True)):
        raise NetMismatchError(
            "The framework graph carries no slot provenance; net "
            "verification needs the per-atom 'slot' attributes recorded "
            "at build time."
        )


def _topology_slot_images(topology: Topology) -> dict[int, np.ndarray]:
    """Home-cell image of every blueprint slot's dummy centroid.

    These are the shared gauge for both quotient graphs: the builder
    places each SBU with its dummy centroid exactly at the blueprint
    slot center, so using the blueprint's integer/fractional split on
    both sides removes the per-node gauge freedom that would otherwise
    let float noise at a cell boundary flip a voltage.
    """
    inv_cell = np.linalg.inv(topology.cell.matrix)
    images: dict[int, np.ndarray] = {}
    for slot_index, slot in enumerate(topology.slots):
        dummy_idx = [
            i for i, site in enumerate(slot.atoms) if site.specie.symbol == "X"
        ]
        dummy_frac = np.asarray(slot.atoms.cart_coords)[dummy_idx] @ inv_cell
        images[slot_index], _ = _split_image(dummy_frac.mean(axis=0))
    return images


def topology_quotient_edges(topology: Topology) -> Counter[Edge]:
    """The blueprint's labeled quotient graph as an edge multiset.

    Nodes are slot indices; one edge per shared blueprint dummy (the
    same tag on two slots), with the voltage worked out from the
    dummies' coincident positions and the slots' home-cell images.
    """
    inv_cell = np.linalg.inv(topology.cell.matrix)
    images = _topology_slot_images(topology)
    tag_sites: dict[int, list[tuple[int, np.ndarray]]] = {}
    for slot_index, slot in enumerate(topology.slots):
        dummy_idx = [
            i for i, site in enumerate(slot.atoms) if site.specie.symbol == "X"
        ]
        dummy_frac = np.asarray(slot.atoms.cart_coords)[dummy_idx] @ inv_cell
        for i, dummy in zip(dummy_idx, dummy_frac, strict=True):
            tag = int(slot.atoms[i].properties["tags"])
            tag_sites.setdefault(tag, []).append((slot_index, dummy))
    edges: Counter[Edge] = Counter()
    for tag in sorted(tag_sites):
        entries = tag_sites[tag]
        if len(entries) < 2:
            continue
        # >2 slots on one tag is malformed input; prepare_build warns
        # and bonds the first two, so the quotient must mirror that
        (slot_a, frac_a), (slot_b, frac_b) = entries[:2]
        shift = np.round(frac_a - frac_b).astype(int)
        voltage = images[slot_b] + shift - images[slot_a]
        edges[_canonical(slot_a, slot_b, voltage)] += 1
    return edges


def framework_quotient_edges(
    framework: Framework, images: dict[int, np.ndarray] | None = None
) -> Counter[Edge]:
    """The built framework's labeled quotient graph as an edge multiset.

    Nodes are the ``slot`` provenance ids; one edge per inter-slot
    bond, with the voltage recovered from the bond's minimum-image
    crossing and the slots' home-cell images (coordinates are
    unwrapped cartesian, so both are exact integer roundings).

    Parameters
    ----------
    framework : Framework
        The built framework.
    images : dict[int, np.ndarray] or None, optional
        Home-cell image per slot id. verify_net passes the blueprint's
        (see _topology_slot_images) so both sides share one gauge;
        when omitted, images are derived from each placed SBU's atom
        centroid - fine standalone, but a centroid within float noise
        of a cell boundary may wrap differently than the blueprint's.
    """
    graph = framework.graph
    _require_provenance(framework)
    inv_cell = np.linalg.inv(framework.cell)
    if images is None:
        by_slot: dict[int, list[int]] = {}
        for node, data in graph.nodes(data=True):
            by_slot.setdefault(data["slot"], []).append(node)
        images = {}
        for slot, nodes in by_slot.items():
            center = np.mean(
                [np.asarray(graph.nodes[n]["coord"], dtype=float) for n in nodes],
                axis=0,
            )
            images[slot], _ = _split_image(center @ inv_cell)
    edges: Counter[Edge] = Counter()
    for u, v in graph.edges():
        slot_u = graph.nodes[u]["slot"]
        slot_v = graph.nodes[v]["slot"]
        if slot_u == slot_v:
            continue
        delta = np.asarray(graph.nodes[u]["coord"], dtype=float) - np.asarray(
            graph.nodes[v]["coord"], dtype=float
        )
        shift = np.round(delta @ inv_cell).astype(int)
        edges[_canonical(slot_u, slot_v, images[slot_v] + shift - images[slot_u])] += 1
    return edges


def verify_net(framework: Framework, topology: Topology) -> None:
    """Check that a built framework realizes its blueprint topology.

    Compares the labeled quotient graphs (slots + inter-slot bonds
    with periodic image voltages) as exact edge multisets - possible
    because the build records which blueprint slot every atom fills.
    Only meaningful for as-built frameworks: supercells, stacks and
    defective frameworks intentionally change the quotient graph.

    Parameters
    ----------
    framework : Framework
        The built framework (untouched by post-build editing).
    topology : Topology
        The blueprint it was built on.

    Raises
    ------
    NetMismatchError
        If the slot sets or the bonded quotient edges differ - the
        framework does not realize the blueprint's net.
    """
    _require_provenance(framework)
    built_slots = set(framework.slots)
    blueprint_slots = set(range(len(topology)))
    if built_slots != blueprint_slots:
        raise NetMismatchError(
            f"Framework {framework.name!r} fills slots "
            f"{sorted(built_slots)} but topology {topology.name!r} has "
            f"slots {sorted(blueprint_slots)}."
        )
    blueprint = topology_quotient_edges(topology)
    # both quotient graphs use the blueprint's slot images: one shared
    # gauge, so equal nets compare equal edge-for-edge (see
    # _topology_slot_images)
    built = framework_quotient_edges(framework, images=_topology_slot_images(topology))
    if built == blueprint:
        return
    missing = blueprint - built
    extra = built - blueprint
    details = []
    if missing:
        details.append(f"missing inter-SBU bonds: {sorted(missing.elements())}")
    if extra:
        details.append(f"unexpected inter-SBU bonds: {sorted(extra.elements())}")
    raise NetMismatchError(
        f"Framework {framework.name!r} does not realize topology "
        f"{topology.name!r} ({'; '.join(details)}). Edges are "
        "(slot_a, slot_b, periodic image voltage)."
    )
