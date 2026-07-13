"""
Framework deconstruction: an experimental structure back to SBUs + net.

The inverse of the build pipeline. A periodic structure (CIF file or
pymatgen Structure) is reduced to library-ready building blocks and
the labeled quotient graph of its underlying net:

1. bond detection with the same EconNN strategy the builder uses;
2. free guests (0-periodic components) removed;
3. atoms clustered into building units under the *metal-oxo*
   convention: metal clusters keep their inorganic coordination
   sphere and oxo-acid binding groups (carboxylate, phosphonate,
   sulfonate), so bonds are cut at the carboxylate-to-backbone
   carbon-carbon bond and at metal-to-donor bonds of N/S/O-donor
   linkers - exactly the granularity of the shipped SBU library;
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

Scope: frameworks with molecular (0-periodic) building units and at
least one metal atom. Rod MOFs (1-periodic units) and metal-free
frameworks (COFs, where cutting requires branch-point analysis
instead of metal detection) raise DeconstructionError.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
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
from autografs.framework import Framework
from autografs.net import Edge, framework_quotient_edges, identify_net
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
        Library nets matching the quotient graph's coordination
        sequence signature; empty when identification was skipped or
        found nothing.
    n_periodic_components : int
        Number of catenated (interpenetrated) subframeworks.
    guest_formulas : list[str]
        Compositions of the removed 0-periodic components.
    """

    structure: Structure
    fragments: dict[str, Fragment]
    units: list[BuildingUnit]
    quotient_edges: Counter[Edge]
    net_candidates: list[str] = field(default_factory=list)
    n_periodic_components: int = 1
    guest_formulas: list[str] = field(default_factory=list)

    def write_xyz(self, path: str | Path) -> Path:
        """Write the fragments as a multi-structure XYZ SBU library.

        The output follows the shipped library convention (an atom
        count line, a comment line carrying ``name=``, dummies as
        ``X``) and loads back with Autografs(xyzfile=...) or
        autografs.utils.xyz_to_sbu.
        """
        lines: list[str] = []
        for name, fragment in self.fragments.items():
            atoms = fragment.atoms
            lines.append(str(len(atoms)))
            lines.append(f'name={name} pbc="F F F"')
            for site in atoms:
                x, y, z = site.coords
                lines.append(
                    f"{site.specie.symbol:<2} {x:>15.8f} {y:>15.8f} {z:>15.8f}"
                )
        out_path = Path(path)
        out_path.write_text("\n".join(lines) + "\n")
        logger.info(f"\t[x] wrote {len(self.fragments)} fragments to {out_path}")
        return out_path

    def __repr__(self) -> str:
        kinds = Counter(unit.kind for unit in self.units)
        summary = ", ".join(
            f"{count} {kind}(s)" for kind, count in sorted(kinds.items())
        )
        net = ", ".join(self.net_candidates) if self.net_candidates else "unidentified"
        return f"Deconstruction({summary}; net: {net})"


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
    # EconNN crashes on an atom with no neighbor in reach, and such
    # atoms are 0-periodic guests by definition: strip them first
    guest_formulas: list[str] = []
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

    # free guests: 0-periodic connected components
    guest_atoms: set[int] = set()
    n_periodic = 0
    for component in networkx.connected_components(bonds):
        rank, _ = _component_periodicity(bonds, set(component))
        if rank == 0:
            guest_atoms |= set(component)
            guest_formulas.append(
                _hill_formula(structure[i].specie.symbol for i in component)
            )
        else:
            n_periodic += 1
    if n_periodic == 0:
        raise DeconstructionError(
            "No periodic component found: the structure is a molecular "
            "crystal, not a framework."
        )
    if guest_atoms:
        logger.info(
            f"\t[x] removed {len(guest_atoms)} guest atoms "
            f"({', '.join(sorted(set(guest_formulas)))})."
        )
        keep = sorted(set(range(len(structure))) - guest_atoms)
        structure = Structure.from_sites([structure[i] for i in keep])
        bonds = _structure_bond_graph(structure)
    if n_periodic > 1:
        logger.warning(
            f"{n_periodic} interpenetrated subframeworks detected; "
            "building units are extracted from all of them and net "
            "identification assumes they realize the same net."
        )

    node_atoms = _node_atoms(structure, bonds)
    units = _unit_partition(bonds, node_atoms)
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
        name = base_name
        suffix = 1
        while name in fragments:
            if fragments[name].has_compatible_symmetry(
                instance, max_rmsd=DEDUPLICATION_MAX_RMSD
            ):
                break
            suffix += 1
            name = f"{base_name}_{suffix}"
        if name not in fragments:
            typed = instance.copy()
            typed.name = name
            fragments[name] = typed
        built_units.append(
            BuildingUnit(
                name=name,
                kind=kind,
                atom_indices=sorted(unit),
                n_connections=n_connections,
            )
        )

    # unit-level quotient graph, reusing the framework machinery: a
    # graph whose "atoms" carry unit ids as slot provenance and
    # unit-unwrapped cartesian coordinates reduces to exactly the
    # quotient graph of the deconstructed net
    unit_graph = networkx.Graph(cell=np.asarray(lattice.matrix, dtype=float))
    for k, unit in enumerate(units):
        for i in sorted(unit):
            unit_graph.add_node(
                i,
                slot=k,
                coord=lattice.get_cartesian_coords(frac[i] + unwraps[k][i]),
                symbol=structure[i].specie.symbol,
            )
    for u, v, _ in cuts:
        unit_graph.add_edge(u, v)
    quotient_edges = framework_quotient_edges(Framework(unit_graph, name="units"))

    net_candidates: list[str] = []
    if topologies is not None:
        net_candidates = identify_net(quotient_edges, topologies)
        logger.info(
            f"\t[x] net candidates: {', '.join(net_candidates) or 'none found'}."
        )

    return Deconstruction(
        structure=structure,
        fragments=fragments,
        units=built_units,
        quotient_edges=quotient_edges,
        net_candidates=net_candidates,
        n_periodic_components=n_periodic,
        guest_formulas=sorted(guest_formulas),
    )
