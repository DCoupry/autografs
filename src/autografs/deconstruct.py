"""
Framework deconstruction: an experimental structure back to SBUs + net.

The inverse of the build pipeline. A periodic structure (CIF file or
pymatgen Structure) is reduced to library-ready building blocks and
the labeled quotient graph of its underlying net:

1. bond detection with the same EconNN strategy the builder uses;
2. free guests (0-periodic components) removed;
3. atoms clustered into building units - the *metal-oxo* convention
   for MOFs (metal clusters keep their inorganic coordination sphere
   and oxo-acid binding groups: carboxylate, phosphonate, sulfonate,
   so bonds are cut at the carboxylate-to-backbone carbon-carbon bond
   and at metal-to-donor bonds), or *branch-point* analysis for
   metal-free frameworks (see below);
4. every cut bond becomes a dummy atom ("X") at the bond midpoint on
   both sides, yielding Fragments the builder can consume directly;
5. building units collapse to quotient graph vertices, cut bonds to
   voltage-labeled edges, and the net is identified against the
   topology library by coordination-sequence signature
   (autografs.net.identify_net).

>>> result = mofgen.deconstruct("IRMOF-1.cif")
>>> result.net_candidates
['pcu']
>>> result.fragments
{'node_C6O13Zn4_6X': ..., 'linker_C6H4_2X': ...}
>>> result.write_xyz("harvested_sbus.xyz")   # feed back into Autografs

Metal-free frameworks (COFs) take a branch-point path instead of
metal-oxo clustering: rigid ring systems and non-ring atoms collapse
to super-vertices, and a super-vertex's external connection count
decides its role (>=3 a node, 2 a linker body, 1 a cap). Ring
perception runs as a bounded zero-voltage cycle search on the periodic
graph, so it needs no global unwrap. This is the *single-node*
convention; a node bundled differently in the original SBU library
(e.g. a triphenylamine core split at its central atom) may come back
more finely divided, but the recovered net is the same.

Interpenetrated (catenated) structures are handled: each periodic
subframework is a separate connected component, identified on its own,
with the fold reported as ``n_periodic_components`` and the per-net
results in ``subframework_nets``.

Scope: frameworks with molecular (0-periodic) building units. Rod MOFs
and 1-periodic (chain) units raise DeconstructionError.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import networkx
import numpy as np
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import EconNN
from pymatgen.core.structure import Molecule, Structure

from autografs.exceptions import DeconstructionError
from autografs.fragment import Fragment

# _canonical and _split_image are net's gauge conventions; the unit
# quotient graph below must share them exactly, so it uses the same
# helpers rather than re-deriving them
from autografs.net import Edge, _canonical, _split_image, identify_net
from autografs.utils import (
    BOND_CUTOFF,
    BOND_TOLERANCE,
    find_element_cutoffs,
    load_uff_lib,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from autografs.topology import Topology

__all__ = [
    "BuildingUnit",
    "Deconstruction",
    "deconstruct",
    "merge_fragment",
    "write_fragments_xyz",
]

logger = logging.getLogger(__name__)

# bonds detected by EconNN are kept only up to this multiple of the
# UFF radius sum: the sum estimates the bond length itself, so real
# bonds sit within a few percent of it, while EconNN's spurious picks
# (isolated guests grabbing the nearest atom, long metal-metal
# contacts) overshoot it by half an angstrom or more
BOND_LENGTH_SLACK = 1.25

# arm-direction RMSD under which two extracted units of equal formula
# and connectivity are treated as the same building block. Stricter
# than the sieve's COMPATIBILITY_MAX_RMSD: this asserts identity, not
# buildability
DEDUPLICATION_MAX_RMSD = 0.1


@dataclass
class BuildingUnit:
    """One placed building unit of a deconstructed structure.

    Attributes
    ----------
    name : str
        Name of the Fragment type this unit is an instance of (a key
        of Deconstruction.fragments).
    kind : str
        "node" (metal cluster), "linker" (organic unit with 2+
        connections) or "cap" (organic unit with a single connection,
        e.g. a bound solvent molecule or modulator residue).
    atom_indices : list[int]
        Site indices into Deconstruction.structure.
    n_connections : int
        Number of cut bonds (dummies of the fragment).
    """

    name: str
    kind: str
    atom_indices: list[int]
    n_connections: int


@dataclass
class Deconstruction:
    """Result of deconstructing a periodic structure.

    Attributes
    ----------
    structure : Structure
        The analyzed structure, free guests removed.
    fragments : dict[str, Fragment]
        Deduplicated building blocks, dummy-capped and library-ready.
    units : list[BuildingUnit]
        Every placed unit, referencing its fragment type by name.
    quotient_edges : Counter[Edge]
        Labeled quotient graph of the framework: one vertex per unit
        (indexed as in ``units``), one voltage-labeled edge per cut
        bond.
    net_candidates : list[str]
        Library nets realized by the framework: the consensus (shared
        candidates) across all interpenetrated subframeworks. Empty
        when identification was skipped, found nothing, or the
        subframeworks disagree (their individual results are then in
        ``subframework_nets``).
    n_periodic_components : int
        Number of catenated (interpenetrated) subframeworks - the fold
        of the interpenetration (1 for a non-catenated framework).
    subframework_nets : list[list[str]]
        Net candidates for each periodic subframework independently,
        one list per component. For genuine n-fold interpenetration
        every entry is the same net; differing entries flag distinct
        interpenetrating nets.
    guest_formulas : list[str]
        Compositions of the removed 0-periodic components.
    """

    structure: Structure
    fragments: dict[str, Fragment]
    units: list[BuildingUnit]
    quotient_edges: Counter[Edge]
    net_candidates: list[str] = field(default_factory=list)
    n_periodic_components: int = 1
    subframework_nets: list[list[str]] = field(default_factory=list)
    guest_formulas: list[str] = field(default_factory=list)

    @property
    def is_catenated(self) -> bool:
        """True when the structure has more than one interpenetrated net."""
        return self.n_periodic_components > 1

    def write_xyz(self, path: str | Path) -> Path:
        """Write the fragments as a multi-structure XYZ SBU library.

        The output follows the shipped library convention (an atom
        count line, a comment line carrying ``name=``, dummies as
        ``X``) and loads back with Autografs(xyzfile=...) or
        autografs.utils.xyz_to_sbu.
        """
        return write_fragments_xyz(self.fragments, path)

    def __repr__(self) -> str:
        kinds = Counter(unit.kind for unit in self.units)
        summary = ", ".join(
            f"{count} {kind}(s)" for kind, count in sorted(kinds.items())
        )
        net = ", ".join(self.net_candidates) if self.net_candidates else "unidentified"
        fold = f"; {self.n_periodic_components}-fold" if self.is_catenated else ""
        return f"Deconstruction({summary}; net: {net}{fold})"


def write_fragments_xyz(fragments: dict[str, Fragment], path: str | Path) -> Path:
    """Write a fragment library as a multi-structure XYZ file.

    The output follows the shipped library convention (an atom count
    line, a comment line carrying ``name=``, dummies as ``X``) and
    loads back with Autografs(xyzfile=...) or autografs.utils.xyz_to_sbu.
    """
    lines: list[str] = []
    for name, fragment in fragments.items():
        atoms = fragment.atoms
        lines.append(str(len(atoms)))
        lines.append(f'name={name} pbc="F F F"')
        for site in atoms:
            x, y, z = site.coords
            lines.append(f"{site.specie.symbol:<2} {x:>15.8f} {y:>15.8f} {z:>15.8f}")
    out_path = Path(path)
    out_path.write_text("\n".join(lines) + "\n")
    logger.info(f"\t[x] wrote {len(fragments)} fragments to {out_path}")
    return out_path


def merge_fragment(
    library: dict[str, Fragment], instance: Fragment, base_name: str
) -> str:
    """Add ``instance`` to ``library`` under a deduplicated name.

    An instance geometrically identical (arm-direction RMSD within
    DEDUPLICATION_MAX_RMSD) to an existing entry of the same base name
    reuses that entry's name; a genuinely different shape sharing the
    base name gets a numeric suffix. The library is mutated in place;
    the resolved name is returned. Used both within one deconstruction
    and to merge fragments harvested across many structures.

    Identity here is composition + connection count + arm directions,
    which is what buildability depends on - it is NOT full chemical
    identity: positional isomers with the same formula and
    near-identical arm geometry merge into one entry. Provenance of a
    merged fragment therefore reads "same building block for the
    builder", not "same molecule".
    """
    name = base_name
    suffix = 1
    while name in library:
        if library[name].has_compatible_symmetry(
            instance, max_rmsd=DEDUPLICATION_MAX_RMSD
        ):
            return name
        suffix += 1
        name = f"{base_name}_{suffix}"
    typed = instance.copy()
    typed.name = name
    library[name] = typed
    return name


def _hill_formula(symbols: Iterable[str]) -> str:
    """Hill-order formula string without separators (C6H4O2Zn)."""
    counts = Counter(symbols)
    ordered = [s for s in ("C", "H") if s in counts]
    ordered += sorted(set(counts) - {"C", "H"})
    return "".join(f"{s}{counts[s]}" if counts[s] > 1 else s for s in ordered)


def _structure_bond_graph(structure: Structure) -> networkx.MultiGraph:
    """Periodic bond graph of a structure, EconNN + UFF fictive radii.

    Nodes are site indices; edges carry ``to_jimage``, the periodic
    image offset from u to v, matching pymatgen's StructureGraph
    convention. The strategy mirrors the builder's molecular bond
    detection so a round-tripped framework reconnects identically.
    """
    strategy = EconNN(tol=BOND_TOLERANCE, use_fictive_radius=True, cutoff=BOND_CUTOFF)
    graph = StructureGraph.from_local_env_strategy(structure, strategy)
    # EconNN is adaptive: an isolated guest atom bonds to whatever is
    # nearest, however far away. Capping detected bonds at the UFF
    # radius sums (the builder's own cutoffs) drops those, along with
    # spurious long metal-metal contacts
    cutoffs = find_element_cutoffs(*load_uff_lib(structure))
    frac = structure.frac_coords
    lattice = structure.lattice
    symbols = [site.specie.symbol for site in structure]
    bonds = networkx.MultiGraph()
    bonds.add_nodes_from(range(len(structure)))
    for u, v, data in graph.graph.edges(data=True):
        jimage = tuple(int(x) for x in data["to_jimage"])
        vector = lattice.get_cartesian_coords(frac[v] + jimage) - (
            lattice.get_cartesian_coords(frac[u])
        )
        distance = float(np.linalg.norm(vector))
        if distance > BOND_LENGTH_SLACK * cutoffs[(symbols[u], symbols[v])]:
            continue
        # canonical orientation low-index -> high-index: an undirected
        # networkx edge yields its data from either end, so the image
        # offset must have one agreed-upon direction (see _jimage_from)
        if u > v:
            u, v = v, u
            jimage = (-jimage[0], -jimage[1], -jimage[2])
        bonds.add_edge(u, v, to_jimage=jimage)
    return bonds


def _jimage_from(u: int, v: int, jimage: tuple[int, int, int]) -> np.ndarray:
    """Image offset of v relative to u for a canonically stored edge."""
    offset = np.asarray(jimage, dtype=int)
    return offset if u <= v else -offset


def _component_periodicity(
    bonds: networkx.MultiGraph, component: set[int]
) -> tuple[int, dict[int, np.ndarray]]:
    """Periodic rank of a connected component, plus unwrap images.

    A spanning breadth-first search assigns every atom the periodic
    image that keeps it bonded to its tree parent; the rank of the
    image mismatches over the remaining edges is the component's
    dimensionality (0 = molecular guest, 3 = full framework). The
    image assignment doubles as the unwrap used to build molecular
    fragments from wrapped coordinates.
    """
    start = min(component)
    images: dict[int, np.ndarray] = {start: np.zeros(3, dtype=int)}
    frontier = [start]
    mismatches: list[np.ndarray] = []
    seen_edges: set[tuple[int, int, int]] = set()
    while frontier:
        new_frontier = []
        for u in frontier:
            for _, v, key, data in bonds.edges(u, keys=True, data=True):
                edge_id = (min(u, v), max(u, v), key)
                if edge_id in seen_edges:
                    continue
                seen_edges.add(edge_id)
                image_v = images[u] + _jimage_from(u, v, data["to_jimage"])
                if v in images:
                    mismatch = image_v - images[v]
                    if mismatch.any():
                        mismatches.append(mismatch)
                else:
                    images[v] = image_v
                    new_frontier.append(v)
        frontier = new_frontier
    rank = 0
    if mismatches:
        rank = int(np.linalg.matrix_rank(np.array(mismatches)))
    return rank, images


def _is_metal(structure: Structure, index: int) -> bool:
    element = structure[index].specie
    return bool(getattr(element, "is_metal", False))


def _node_atoms(structure: Structure, bonds: networkx.MultiGraph) -> set[int]:
    """Atoms belonging to metal-oxo nodes.

    The metal-oxo convention, matching the shipped SBU library:

    - metal atoms;
    - non-carbon atoms bonded to a metal but to no carbon (oxo,
      hydroxo, aqua oxygens, bridging halides, ammine nitrogens);
    - oxo-acid binding groups: a C, S or P bonded to two or more
      oxygens of which at least one touches a metal is absorbed with
      all its oxygens (carboxylate, sulfonate, phosphonate) - for
      carbon only when it has at most one carbon neighbor, which is
      the bond to the linker backbone where the cut happens;
    - hydrogens all of whose neighbors are node atoms (mu-OH, water).

    Donor atoms with a carbon neighbor (pyridine N, thiolate S,
    phenolate O) stay with the organic unit: those cuts happen at the
    metal-donor bond.
    """
    symbols = [site.specie.symbol for site in structure]
    neighbors: dict[int, set[int]] = {i: set(bonds.neighbors(i)) for i in bonds.nodes}
    node_atoms = {i for i in bonds.nodes if _is_metal(structure, i)}
    if not node_atoms:
        raise DeconstructionError(
            "No metal atoms found: metal-oxo deconstruction needs at "
            "least one metal. Metal-free frameworks (COFs) require "
            "branch-point analysis and are not supported yet."
        )
    # inorganic coordination sphere
    for i in bonds.nodes:
        if i in node_atoms or symbols[i] in ("C", "H"):
            continue
        bonded_symbols = {symbols[j] for j in neighbors[i]}
        if "C" not in bonded_symbols and neighbors[i] & node_atoms:
            node_atoms.add(i)
    # oxo-acid binding groups
    for i in bonds.nodes:
        if i in node_atoms or symbols[i] not in ("C", "S", "P"):
            continue
        oxygens = {j for j in neighbors[i] if symbols[j] == "O"}
        if len(oxygens) < 2:
            continue
        if symbols[i] == "C":
            carbons = [j for j in neighbors[i] if symbols[j] == "C"]
            if len(carbons) > 1:
                continue
        metal_bound = any(
            any(_is_metal(structure, k) for k in neighbors[j]) for j in oxygens
        )
        if metal_bound:
            node_atoms.add(i)
            node_atoms |= oxygens
    # hydrogen mop-up (mu-OH, aqua, carboxylic acid protons)
    for i in bonds.nodes:
        if symbols[i] == "H" and neighbors[i] and neighbors[i] <= node_atoms:
            node_atoms.add(i)
    return node_atoms


def _unit_partition(bonds: networkx.MultiGraph, node_atoms: set[int]) -> list[set[int]]:
    """Building units: metal clusters, then organic components."""
    units: list[set[int]] = []
    for atom_set in (node_atoms, set(bonds.nodes) - node_atoms):
        subgraph = bonds.subgraph(atom_set)
        units.extend(
            set(component) for component in networkx.connected_components(subgraph)
        )
    return units


# longest molecular ring the branch-point analysis perceives; 6-rings
# (benzene, triazine, boroxine) and their fused/bridged relatives all
# fall well under this, and the bound keeps the per-bond cycle search
# cheap on the periodic graph
BOND_RING_MAX = 8

_Voltage = tuple[int, int, int]


def _ring_bonds(
    bonds: networkx.MultiGraph, heavy: set[int], max_ring: int = BOND_RING_MAX
) -> set[tuple[int, int, _Voltage]]:
    """Heavy-atom bonds that lie on a molecular (zero-voltage) ring.

    A bond is a ring bond when its endpoints are joined by an
    alternate path of at most ``max_ring`` - 1 further hops whose
    voltages sum to the bond's own voltage - i.e. a small cycle that
    closes within one unit cell rather than wrapping the lattice. The
    search runs in the periodic graph's infinite unfolding (states are
    (atom, image) pairs), so it needs no global unwrap and never
    splits a ring that straddles a cell boundary.
    """
    adjacency: dict[int, list[tuple[int, _Voltage]]] = defaultdict(list)
    edges: list[tuple[int, int, _Voltage]] = []
    for u, v, data in bonds.edges(data=True):
        if u not in heavy or v not in heavy:
            continue
        offset = _jimage_from(u, v, data["to_jimage"])
        voltage = (int(offset[0]), int(offset[1]), int(offset[2]))
        adjacency[u].append((v, voltage))
        adjacency[v].append((u, (-voltage[0], -voltage[1], -voltage[2])))
        edges.append((u, v, voltage))
    ring_bonds: set[tuple[int, int, _Voltage]] = set()
    for u, v, voltage in edges:
        back = (-voltage[0], -voltage[1], -voltage[2])
        target = (v, voltage)
        seen = {(u, (0, 0, 0))}
        frontier: deque[tuple[tuple[int, _Voltage], int]] = deque([((u, (0, 0, 0)), 0)])
        found = False
        while frontier and not found:
            (atom, image), depth = frontier.popleft()
            if depth >= max_ring - 1:
                continue
            for neighbor, step in adjacency[atom]:
                # remove the bond under test (and any parallel copy) so
                # only a genuine detour can reach the target
                if (atom == u and neighbor == v and step == voltage) or (
                    atom == v and neighbor == u and step == back
                ):
                    continue
                new_image = (
                    image[0] + step[0],
                    image[1] + step[1],
                    image[2] + step[2],
                )
                state = (neighbor, new_image)
                if state == target:
                    found = True
                    break
                if state not in seen:
                    seen.add(state)
                    frontier.append((state, depth + 1))
        if found:
            ring_bonds.add((u, v, voltage))
    return ring_bonds


def _organic_units(
    structure: Structure, bonds: networkx.MultiGraph
) -> tuple[list[set[int]], set[int]]:
    """Branch-point clustering for metal-free frameworks (COFs).

    The single-node convention: rigid ring systems and non-ring atoms
    are collapsed to super-vertices, and a super-vertex's external
    connection count decides its role - three or more make it a *node*
    (a branch point of the underlying net), two make it part of a
    *linker* strut, one a terminal *cap*. Cuts fall on the bonds
    leaving a node, so a linker keeps every 2-connected fragment along
    its length (a biphenyl bridge stays one unit).

    Returns the atom partition and the set of atoms belonging to nodes
    (kept parallel to the metal-oxo path so the rest of the pipeline is
    shared).
    """
    symbols = [site.specie.symbol for site in structure]
    heavy = {i for i in bonds.nodes if symbols[i] != "H"}
    if not heavy:
        raise DeconstructionError(
            "The framework has no heavy atoms to cluster into building units."
        )

    # rigid ring systems become single super-vertices; every other
    # heavy atom is its own super-vertex
    ring_graph = networkx.Graph()
    ring_graph.add_nodes_from(heavy)
    for u, v, _ in _ring_bonds(bonds, heavy):
        ring_graph.add_edge(u, v)
    super_of: dict[int, int] = {}
    super_atoms: dict[int, set[int]] = {}
    for super_id, component in enumerate(networkx.connected_components(ring_graph)):
        super_atoms[super_id] = set(component)
        for atom in component:
            super_of[atom] = super_id

    # external connections per super-vertex (bonds crossing super-vertices)
    super_degree: Counter[int] = Counter()
    for u, v, _ in bonds.edges(data=True):
        if u in heavy and v in heavy and super_of[u] != super_of[v]:
            super_degree[super_of[u]] += 1
            super_degree[super_of[v]] += 1
    node_supers = {sid for sid in super_atoms if super_degree[sid] >= 3}

    units: list[set[int]] = []
    node_unit_indices: set[int] = set()
    for sid in node_supers:
        node_unit_indices.add(len(units))
        units.append(set(super_atoms[sid]))
    # linkers/caps: connected runs of non-node super-vertices, joined by
    # the super-edges that do not touch a node
    linker_graph = networkx.Graph()
    linker_graph.add_nodes_from(sid for sid in super_atoms if sid not in node_supers)
    for u, v, _ in bonds.edges(data=True):
        if u in heavy and v in heavy:
            su, sv = super_of[u], super_of[v]
            if su != sv and su not in node_supers and sv not in node_supers:
                linker_graph.add_edge(su, sv)
    for component in networkx.connected_components(linker_graph):
        atoms: set[int] = set()
        for sid in component:
            atoms |= super_atoms[sid]
        units.append(atoms)

    # hydrogens ride with their heavy neighbour's unit
    atom_unit = {atom: k for k, unit in enumerate(units) for atom in unit}
    for i in bonds.nodes:
        if symbols[i] != "H":
            continue
        heavy_neighbours = [j for j in bonds.neighbors(i) if j in heavy]
        if heavy_neighbours:
            units[atom_unit[heavy_neighbours[0]]].add(i)
        else:
            units.append({i})

    node_atoms: set[int] = set()
    for k in node_unit_indices:
        node_atoms |= units[k]
    return units, node_atoms


def _unwrap_unit(
    structure: Structure, bonds: networkx.MultiGraph, unit: set[int]
) -> dict[int, np.ndarray]:
    """Periodic images making a unit a connected molecular cluster.

    Raises
    ------
    DeconstructionError
        If the unit is bonded to its own periodic image (a rod or
        layer, which has no molecular fragment representation).
    """
    rank, images = _component_periodicity(bonds.subgraph(unit), unit)
    if rank > 0:
        symbols = Counter(structure[i].specie.symbol for i in sorted(unit)[:50])
        raise DeconstructionError(
            f"Building unit {dict(symbols)} is {rank}-periodic (a rod or "
            "layer). Rod-MOF deconstruction is not supported yet."
        )
    return images


def deconstruct(
    source: Structure | str | Path,
    topologies: Mapping[str, Topology] | None = None,
) -> Deconstruction:
    """Deconstruct a periodic structure into SBUs and its net.

    Parameters
    ----------
    source : Structure or str or Path
        A pymatgen Structure, or the path of any structure file
        pymatgen reads (CIF included).
    topologies : Mapping[str, Topology] or None, optional
        Topology library to identify the net against (e.g.
        Autografs.topologies). When None, net identification is
        skipped and ``net_candidates`` stays empty.

    Returns
    -------
    Deconstruction
        Building blocks, placed units, quotient graph and net
        candidates. See the class documentation.

    Raises
    ------
    DeconstructionError
        For disordered input, structures without metals, structures
        with no periodic component, or rod-like building units.

    Examples
    --------
    >>> result = deconstruct("IRMOF-1.cif", topologies=mofgen.topologies)
    >>> result.net_candidates
    ['pcu']
    """
    if isinstance(source, (str, Path)):
        structure = Structure.from_file(str(source))
    else:
        structure = source.copy()
    if not structure.is_ordered:
        raise DeconstructionError(
            "The structure has partially occupied sites; deconstruction "
            "needs an ordered structure. Resolve the disorder first."
        )
    structure.remove_oxidation_states()
    if "X" in {site.specie.symbol for site in structure}:
        raise DeconstructionError(
            "The structure contains dummy atoms ('X'); deconstruction "
            "expects a real crystal structure."
        )
    # Guest removal, iterated to a guest-free bond graph. EconNN adapts
    # bond detection to each atom's environment, so removing guests can
    # change the bonds detected among the remaining atoms; looping until
    # no 0-periodic component is left guarantees the interpenetration
    # fold and the per-subframework split both come from the one final
    # graph. Isolated atoms are stripped before each bond pass because
    # EconNN crashes on an atom with no neighbor in reach (and such
    # atoms are 0-periodic guests by definition).
    guest_formulas: list[str] = []
    while True:
        isolated = [
            i for i, nn in enumerate(structure.get_all_neighbors(BOND_CUTOFF)) if not nn
        ]
        if len(isolated) == len(structure):
            raise DeconstructionError(
                "No periodic component found: the structure is a molecular "
                "crystal, not a framework."
            )
        if isolated:
            guest_formulas += [structure[i].specie.symbol for i in isolated]
            structure.remove_sites(isolated)
        bonds = _structure_bond_graph(structure)
        components = [set(c) for c in networkx.connected_components(bonds)]
        guest_components = [
            c for c in components if _component_periodicity(bonds, c)[0] == 0
        ]
        if not guest_components:
            break
        guest_atoms = set().union(*guest_components)
        if len(guest_atoms) == len(structure):
            raise DeconstructionError(
                "No periodic component found: the structure is a molecular "
                "crystal, not a framework."
            )
        guest_formulas += [
            _hill_formula(structure[i].specie.symbol for i in c)
            for c in guest_components
        ]
        logger.info(
            f"\t[x] removed {len(guest_atoms)} guest atoms "
            f"({', '.join(sorted(set(guest_formulas)))})."
        )
        keep = sorted(set(range(len(structure))) - guest_atoms)
        structure = Structure.from_sites([structure[i] for i in keep])
    # every remaining component is a periodic subframework; more than
    # one means interpenetration, each identified independently below
    periodic_components = components
    n_periodic = len(periodic_components)
    if n_periodic > 1:
        logger.info(f"\t[x] {n_periodic} interpenetrated subframeworks detected.")

    # metal-oxo clustering for MOFs; branch-point analysis for
    # metal-free frameworks (COFs), which have no metal to anchor cuts
    if any(_is_metal(structure, i) for i in bonds.nodes):
        node_atoms = _node_atoms(structure, bonds)
        units = _unit_partition(bonds, node_atoms)
    else:
        units, node_atoms = _organic_units(structure, bonds)
    atom_to_unit = {atom: k for k, unit in enumerate(units) for atom in unit}

    # cut bonds join two different units by construction
    cuts: list[tuple[int, int, tuple[int, int, int]]] = []
    for u, v, data in bonds.edges(data=True):
        if atom_to_unit[u] != atom_to_unit[v]:
            offset = _jimage_from(u, v, data["to_jimage"])
            cuts.append((u, v, (int(offset[0]), int(offset[1]), int(offset[2]))))

    frac = structure.frac_coords
    lattice = structure.lattice
    unwraps = [_unwrap_unit(structure, bonds, unit) for unit in units]

    # dummy positions per unit: midpoint of every cut bond, expressed
    # in the unit's own unwrap frame
    unit_dummies: dict[int, list[np.ndarray]] = defaultdict(list)
    for u, v, jimage in cuts:
        shift = np.asarray(jimage, dtype=int)
        for atom, partner, partner_shift in ((u, v, shift), (v, u, -shift)):
            unit_index = atom_to_unit[atom]
            atom_frac = frac[atom] + unwraps[unit_index][atom]
            # the partner sits one bond away from this atom's unwrapped
            # position: its cell coordinate shifted by the bond's image
            # plus this atom's unwrap
            partner_frac = frac[partner] + partner_shift + unwraps[unit_index][atom]
            midpoint = lattice.get_cartesian_coords((atom_frac + partner_frac) / 2.0)
            unit_dummies[unit_index].append(midpoint)

    # molecular fragments, one per unit instance
    instances: list[Fragment] = []
    for k, unit in enumerate(units):
        ordered = sorted(unit)
        species = [structure[i].specie.symbol for i in ordered]
        coords = [
            lattice.get_cartesian_coords(frac[i] + unwraps[k][i]) for i in ordered
        ]
        species += ["X"] * len(unit_dummies[k])
        coords += list(unit_dummies[k])
        instances.append(Fragment(atoms=Molecule(species, coords), name=f"unit_{k}"))

    # deduplicate instances into named fragment types
    fragments: dict[str, Fragment] = {}
    built_units: list[BuildingUnit] = []
    for k, (unit, instance) in enumerate(zip(units, instances, strict=True)):
        n_connections = len(unit_dummies[k])
        # units are entirely node atoms or entirely organic by construction
        if unit <= node_atoms:
            kind = "node"
        elif n_connections >= 2:
            kind = "linker"
        else:
            kind = "cap"
        real_formula = _hill_formula(structure[i].specie.symbol for i in unit)
        base_name = f"{kind}_{real_formula}_{n_connections}X"
        name = merge_fragment(fragments, instance, base_name)
        built_units.append(
            BuildingUnit(
                name=name,
                kind=kind,
                atom_indices=sorted(unit),
                n_connections=n_connections,
            )
        )

    # unit-level labeled quotient graph, built directly from the cut
    # list: one vertex per unit, one voltage-labeled edge per cut bond.
    # Building it as a simple unit graph would collapse parallel cuts
    # joining the same atom pair through different periodic images (a
    # one-atom bridge between a unit and its partner's own image),
    # silently dropping quotient edges. The gauge matches
    # autografs.net: a unit's home-cell image is the integer split of
    # its unwrapped centroid, and the cut's voltage follows exactly
    # from its image offset and the two endpoint unwraps.
    unit_images: dict[int, np.ndarray] = {}
    for k, unit in enumerate(units):
        centroid = np.mean([frac[i] + unwraps[k][i] for i in sorted(unit)], axis=0)
        unit_images[k], _ = _split_image(centroid)

    def unit_quotient(atoms: set[int] | None = None) -> Counter[Edge]:
        edges: Counter[Edge] = Counter()
        for u, v, jimage in cuts:
            # a cut joins two bonded units, so both endpoints share a
            # periodic component: one endpoint decides the restriction
            if atoms is not None and u not in atoms:
                continue
            unit_u = atom_to_unit[u]
            unit_v = atom_to_unit[v]
            shift = np.asarray(jimage, dtype=int) + (
                unwraps[unit_u][u] - unwraps[unit_v][v]
            )
            voltage = unit_images[unit_v] + shift - unit_images[unit_u]
            edges[_canonical(unit_u, unit_v, voltage)] += 1
        return edges

    quotient_edges = unit_quotient()

    # identify each interpenetrated subframework independently
    subframework_nets: list[list[str]] = []
    if topologies is not None:
        subframework_nets = [
            identify_net(unit_quotient(component), topologies)
            for component in periodic_components
        ]

    net_candidates: list[str] = []
    if topologies is not None:
        # consensus: the candidates common to every subframework
        common = set(subframework_nets[0])
        for nets in subframework_nets[1:]:
            common &= set(nets)
        net_candidates = sorted(common)
        if n_periodic > 1 and not net_candidates:
            logger.warning(
                f"interpenetrated subframeworks identify different nets "
                f"({subframework_nets}); no consensus net."
            )
        logger.info(
            f"\t[x] net candidates: {', '.join(net_candidates) or 'none found'}"
            f"{f' ({n_periodic}-fold)' if n_periodic > 1 else ''}."
        )

    return Deconstruction(
        structure=structure,
        fragments=fragments,
        units=built_units,
        quotient_edges=quotient_edges,
        net_candidates=net_candidates,
        n_periodic_components=n_periodic,
        subframework_nets=subframework_nets,
        guest_formulas=sorted(guest_formulas),
    )
