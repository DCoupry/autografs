"""
Net verification and identification on labeled quotient graphs.

Verification: does a built framework realize its blueprint? The build
gates check geometry (``max_rmsd``) and overlap (``min_distance``),
but neither verifies that the output actually realizes the requested
topology - mis-paired tags or a bond through the wrong periodic image
would pass both while building a different net. Comparing labeled
quotient graphs closes that gap.

Both a Topology and a built Framework reduce to the same object: one
node per slot, and one edge per inter-slot bond carrying an integer
*voltage* (the periodic image offset between the two slots' home-cell
representatives). Because every built atom records which blueprint
slot it fills (the ``slot`` provenance attribute), the two quotient
graphs share their node labels, and verification is an exact multiset
comparison of canonicalized edges - no graph isomorphism search.

>>> mof = mofgen.build(topology, mappings, verify_net=True)   # gated
>>> mof.verify_net(topology)                                  # explicit

Identification: which library net does an *unlabeled* quotient graph
realize? Deconstruction (autografs.deconstruct) produces a quotient
graph with arbitrary node ids, so the multiset comparison above does
not apply. Instead each net is reduced to a cell-choice-invariant
signature - the multiset of per-vertex coordination sequences of the
underlying periodic graph, with 1-coordinated vertices (caps) pruned
and 2-coordinated vertices (edge centers, ditopic linkers) contracted
first - and matched against signatures computed the same way from the
topology library. This is the CrystalNets/ToposPro approach:
coordination sequences are not a proof of isomorphism, but they are an
almost-unique invariant across known net archives, and matching is
against a fixed library, so collisions surface as multiple candidates
rather than silent errors.

>>> identify_net(framework_quotient_edges(mof), mofgen.topologies)
['pcu']
"""

from __future__ import annotations

import functools
import logging
import math
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

import numpy as np

from autografs.exceptions import NetMismatchError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from autografs.framework import Framework
    from autografs.topology import Topology

__all__ = [
    "topology_quotient_edges",
    "framework_quotient_edges",
    "verify_net",
    "coordination_sequences",
    "net_signature",
    "identify_net",
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


# ---------------------------------------------------------------------
# net identification
# ---------------------------------------------------------------------

# shells of the coordination sequence used in net signatures. Ten is
# the CrystalNets/ToposPro convention: empirically enough to separate
# every net in the known archives
SIGNATURE_SHELLS = 10

Voltage = tuple[int, int, int]
# node -> outgoing (neighbor, voltage) half-edges, one per edge end,
# so len(adjacency[n]) is n's degree in the periodic graph
_Adjacency = dict[int, list[tuple[int, Voltage]]]


def _negate(voltage: Voltage) -> Voltage:
    return (-voltage[0], -voltage[1], -voltage[2])


def _add(a: Voltage, b: Voltage) -> Voltage:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _adjacency(edges: Counter[Edge]) -> _Adjacency:
    """Directed half-edge adjacency of a quotient edge multiset.

    Every quotient edge contributes one half-edge at each end (a
    self-edge contributes two at its node, one per voltage sign), so
    vertex degrees in the periodic graph fall out as list lengths.
    """
    adjacency: _Adjacency = defaultdict(list)
    for (node_a, node_b, voltage), multiplicity in edges.items():
        for _ in range(multiplicity):
            adjacency[node_a].append((node_b, voltage))
            adjacency[node_b].append((node_a, _negate(voltage)))
    return dict(adjacency)


def _prune_and_contract(adjacency: _Adjacency, contract: bool = True) -> _Adjacency:
    """Reduce a quotient graph to its topology-bearing vertices.

    Iteratively removes 1-coordinated vertices (capping ligands and
    other dead ends carry no net information) and, when ``contract``
    is set, contracts 2-coordinated vertices into a single edge
    joining their neighbors (edge centers in blueprints and ditopic
    linkers in deconstructed frameworks are the same edge of the
    underlying net). A vertex whose two half-edges form a self-loop
    cannot be contracted and is left in place, so a pure-cycle
    component terminates the loop.
    """
    adjacency = {node: list(halves) for node, halves in adjacency.items()}
    changed = True
    while changed:
        changed = False
        for node in list(adjacency):
            halves = adjacency.get(node)
            if halves is None or node in {n for n, _ in halves}:
                continue
            if len(halves) == 1:
                ((neighbor, voltage),) = halves
                adjacency[neighbor].remove((node, _negate(voltage)))
                del adjacency[node]
                changed = True
            elif len(halves) == 2 and contract:
                (nb_a, v_a), (nb_b, v_b) = halves
                # the contracted edge runs nb_a -> node -> nb_b, so its
                # voltage chains the two hops
                adjacency[nb_a].remove((node, _negate(v_a)))
                adjacency[nb_b].remove((node, _negate(v_b)))
                adjacency[nb_a].append((nb_b, _add(_negate(v_a), v_b)))
                adjacency[nb_b].append((nb_a, _add(_negate(v_b), v_a)))
                del adjacency[node]
                changed = True
    return adjacency


def coordination_sequences(
    edges: Counter[Edge], shells: int = SIGNATURE_SHELLS
) -> dict[int, tuple[int, ...]]:
    """Coordination sequence of every quotient vertex.

    The coordination sequence of a vertex counts the vertices of the
    *periodic* graph (the infinite unfolding of the quotient graph) at
    each graph distance 1..shells - the classic per-vertex topology
    invariant of crystal chemistry. Computed by breadth-first search
    over (vertex, periodic image) states.

    Parameters
    ----------
    edges : Counter[Edge]
        Labeled quotient graph as produced by topology_quotient_edges
        or framework_quotient_edges.
    shells : int, optional
        Number of coordination shells to expand.

    Returns
    -------
    dict[int, tuple[int, ...]]
        Vertex id to its coordination sequence, one count per shell.
    """
    adjacency = _adjacency(edges)
    sequences: dict[int, tuple[int, ...]] = {}
    for source in adjacency:
        origin = (source, (0, 0, 0))
        visited: set[tuple[int, Voltage]] = {origin}
        frontier = [origin]
        counts = []
        for _ in range(shells):
            new_frontier = []
            for node, image in frontier:
                for neighbor, voltage in adjacency[node]:
                    state = (neighbor, _add(image, voltage))
                    if state not in visited:
                        visited.add(state)
                        new_frontier.append(state)
            counts.append(len(new_frontier))
            frontier = new_frontier
        sequences[source] = tuple(counts)
    return sequences


Signature = tuple[tuple[tuple[int, ...], int], ...]


def net_signature(
    edges: Counter[Edge], shells: int = SIGNATURE_SHELLS, contract: bool = True
) -> Signature:
    """Cell-choice-invariant signature of a quotient graph.

    1-coordinated vertices are pruned and (by default) 2-coordinated
    vertices contracted first (see _prune_and_contract), then the
    multiset of coordination sequences is collected with its counts
    reduced by their gcd: a supercell of the same net multiplies every
    count by the same integer, so reduced counts compare equal across
    cell choices. Not a proof of isomorphism - distinct nets sharing
    all coordination sequences would collide - but an almost-unique
    invariant across the known net archives (the CrystalNets result).

    Parameters
    ----------
    edges : Counter[Edge]
        Labeled quotient graph.
    shells : int, optional
        Coordination sequence depth.
    contract : bool, optional
        With contraction (default) the signature identifies the
        underlying net regardless of how its edges are subdivided.
        Without it, 2-coordinated vertices (blueprint edge centers,
        ditopic linkers) count as vertices of their own, which
        separates a net from its edge-decorated derivatives - the
        finer of the two matching tiers in identify_net.

    Returns
    -------
    Signature
        Sorted tuple of (coordination sequence, reduced count) pairs;
        empty when nothing remains after pruning.
    """
    reduced = _prune_and_contract(_adjacency(edges), contract=contract)
    quotient: Counter[Edge] = Counter()
    for node, halves in reduced.items():
        for neighbor, voltage in halves:
            key = _canonical(node, neighbor, np.asarray(voltage))
            quotient[key] += 1
    # every non-loop edge was seen from both ends, every self-loop
    # from both voltage signs (canonicalized to one key)
    quotient = Counter({edge: mult // 2 for edge, mult in quotient.items()})
    counts = Counter(coordination_sequences(quotient, shells=shells).values())
    if not counts:
        return ()
    divisor = math.gcd(*counts.values())
    return tuple(sorted((seq, mult // divisor) for seq, mult in counts.items()))


@functools.lru_cache(maxsize=8192)
def _topology_signature_cached(
    topology: Topology, shells: int, contract: bool
) -> Signature:
    """Signature of a library topology, cached per Topology object.

    Topology neither defines __eq__ nor __hash__, so the cache is
    keyed by object identity - correct because LazyTopologyLibrary
    caches materialized topologies for the session.
    """
    return net_signature(
        topology_quotient_edges(topology), shells=shells, contract=contract
    )


def _degree_profile(degrees: list[int]) -> tuple[tuple[int, int], ...]:
    """Reduced multiset of topology-bearing vertex degrees.

    Degrees 1 and 2 are dropped (pruned/contracted away by the
    signature) and the remaining counts divided by their gcd, giving a
    supercell-invariant prefilter that is much cheaper than a
    signature: it needs only per-vertex connectivities, which the
    topology library exposes without materializing Topology objects.
    """
    counts = Counter(d for d in degrees if d > 2)
    if not counts:
        return ()
    divisor = math.gcd(*counts.values())
    return tuple(sorted((degree, mult // divisor) for degree, mult in counts.items()))


def identify_net(
    edges: Counter[Edge],
    topologies: Mapping[str, Topology],
    shells: int = SIGNATURE_SHELLS,
) -> list[str]:
    """Names of the library nets matching a quotient graph's signature.

    Candidates are prefiltered on the reduced degree multiset of their
    topology-bearing vertices (read from the library's raw payload
    when available, so the full RCSR library is scanned without
    materializing it), then matched in two tiers: first on the exact
    uncontracted signature (2-coordinated vertices counted, so a net
    and its edge-decorated derivatives stay distinct), and only if
    nothing matches exactly, on the contracted signature (the
    underlying net, however its edges are subdivided).

    Parameters
    ----------
    edges : Counter[Edge]
        Labeled quotient graph of the structure to identify.
    topologies : Mapping[str, Topology]
        Topology library to match against (e.g. Autografs.topologies).
    shells : int, optional
        Coordination sequence depth; both sides use the same value.

    Returns
    -------
    list[str]
        Sorted names of the matching nets - usually one, empty when
        nothing in the library matches, several only if the library
        holds nets that share all coordination sequences.
    """
    contracted = net_signature(edges, shells=shells, contract=True)
    if not contracted:
        return []
    # the first coordination shell of a vertex is its degree; the
    # profile is contraction-invariant, so it prefilters both tiers
    target_profile = _degree_profile(
        [seq[0] for seq, mult in contracted for _ in range(mult)]
    )
    raw_items = getattr(topologies, "raw_items", None)
    if raw_items is not None:
        candidates = [
            name
            for name, payload in raw_items()
            if _degree_profile(
                [slot["species"].count("X") for slot in payload.get("slots", [])]
            )
            == target_profile
        ]
    else:
        candidates = [
            name
            for name, topology in topologies.items()
            if _degree_profile(
                [len(slot.atoms.indices_from_symbol("X")) for slot in topology.slots]
            )
            == target_profile
        ]
    exact = net_signature(edges, shells=shells, contract=False)
    exact_matches = sorted(
        name
        for name in candidates
        if _topology_signature_cached(topologies[name], shells, False) == exact
    )
    if exact_matches:
        return exact_matches
    return sorted(
        name
        for name in candidates
        if _topology_signature_cached(topologies[name], shells, True) == contracted
    )
