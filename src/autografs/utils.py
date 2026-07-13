"""
Utility functions for AuToGraFS framework generation.

This module provides utility functions for file I/O, molecular graph
manipulation, force field parameterization, and structure visualization.

Functions
---------
format_mappings
    Format slot-to-SBU mappings for logging output.
get_xyz_names
    Extract fragment names from multi-structure XYZ files.
xyz_to_sbu
    Load Secondary Building Units from an XYZ file.
load_uff_lib
    Load UFF force field parameters for a molecule.
find_element_cutoffs
    Calculate bond distance cutoffs from UFF radii.
find_mmtypes
    Determine UFF atom types from molecular connectivity.
fragment_to_molgraph
    Convert a Fragment to a pymatgen MoleculeGraph.
fragments_to_networkx
    Combine fragments into a single networkx Graph.
view_graph
    Visualize a molecular graph using ASE.
networkx_to_gulp
    Export a molecular graph to GULP input format.
"""

from __future__ import annotations

import functools
import itertools
import logging
from collections import defaultdict
from collections.abc import Iterable
from itertools import count, groupby
from pathlib import Path

import networkx
import numpy as np
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import EconNN
from pymatgen.core.bonds import get_bond_order
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.xyz import XYZ

from autografs.data.uff4mof import UFF4MOF, UFFType
from autografs.fragment import Fragment

__all__ = [
    "format_mappings",
    "get_xyz_names",
    "xyz_to_sbu",
    "load_uff_lib",
    "find_element_cutoffs",
    "find_mmtypes",
    "fragment_to_molgraph",
    "fragments_to_networkx",
    "view_graph",
    "networkx_to_gulp",
]

logger = logging.getLogger(__name__)

# Constants for molecular analysis
BOND_TOLERANCE = 0.3  # Tolerance for bond detection in EconNN
BOND_CUTOFF = 10.0  # Maximum cutoff distance for bond detection


def format_indices(iterable: Iterable[int]) -> str:
    """Format a consecutive run of indices as "first-last" (or "only")."""
    lst = list(iterable)
    if len(lst) > 1:
        return f"{lst[0]}-{lst[-1]}"
    return f"{lst[0]}"


def _format_runs(indices: list[int]) -> str:
    """Format sorted indices with consecutive runs grouped (e.g. "0-2,5")."""
    counter = count()
    # n - next(counter) is constant along a consecutive run
    grouped = groupby(sorted(indices), lambda n: n - next(counter))
    return ",".join(format_indices(g) for _, g in grouped)


def format_mappings(mappings: dict[int, str]) -> str:
    """Format slot-to-SBU mappings into a readable string.

    Converts a dictionary of slot indices to SBU names into a compact,
    human-readable format suitable for logging.

    Parameters
    ----------
    mappings : dict[int, str]
        Dictionary mapping slot indices to SBU names.

    Returns
    -------
    str
        Formatted string with consecutive indices grouped (e.g., "0-3").

    Examples
    --------
    >>> mappings = {0: "SBU_A", 1: "SBU_A", 2: "SBU_A", 3: "SBU_B"}
    >>> print(format_mappings(mappings))
    SBU_A : 0-2; SBU_B : 3
    """
    new_dict: defaultdict[str, list[int]] = defaultdict(list)
    for slot, sbu_name in mappings.items():
        new_dict[sbu_name].append(slot)
    out_mappings = []
    for sbu_name, slots in new_dict.items():
        out_mappings.append(f"{sbu_name} : {_format_runs(slots)}")
    return "; ".join(out_mappings)


def get_xyz_names(path: str) -> list[str]:
    """Extract fragment names from a multi-structure XYZ file.

    Parses comment lines in XYZ format files to extract names defined
    with the "name=" tag. Structures without names are labeled "Unnamed".

    Parameters
    ----------
    path : str
        Path to the XYZ file containing one or more molecular structures.

    Returns
    -------
    list[str]
        List of names corresponding to each structure in the file.

    Examples
    --------
    >>> names = get_xyz_names("sbus.xyz")
    >>> print(names)  # ['Benzene_linear', 'Zn_paddlewheel', ...]
    """

    def is_count(line: str) -> bool:
        return line.strip().isdigit()

    names = []
    with open(path) as xyz_file:
        lines = xyz_file.readlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if not is_count(lines[i]):
            raise ValueError(
                f"Malformed XYZ file {path}: expected an atom count on "
                f"line {i + 1}, got {stripped!r}."
            )
        # a count line directly followed by another count line (or EOF)
        # is a stray with no molecule behind it; pymatgen's XYZ reader
        # skips those, so the name walk must too to stay in step
        if i + 1 >= len(lines) or is_count(lines[i + 1]):
            i += 1
            continue
        comment = lines[i + 1]
        name = next(
            (
                token.removeprefix("name=")
                for token in comment.split()
                if token.startswith("name=")
            ),
            "Unnamed",
        )
        names.append(name)
        i += 2 + int(stripped)
    return names


def xyz_to_sbu(path: str) -> dict[str, Fragment]:
    """Load Secondary Building Units from an XYZ file.

    Reads a multi-structure XYZ file and creates Fragment objects for
    each structure. Dummy atoms ("X") define connection points.

    Point group analysis is NOT run here: compatibility and alignment
    are geometric (arm directions), so the symbol is pure metadata and
    each Fragment computes it lazily on first access. This keeps
    library loading at XYZ-parsing speed.

    Parameters
    ----------
    path : str
        Path to the XYZ file containing SBU structures.

    Returns
    -------
    dict[str, Fragment]
        Dictionary mapping SBU names to Fragment objects.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is not valid XYZ (bad atom counts, unparseable
        coordinates, or mismatched structure/comment counts).

    Examples
    --------
    >>> sbus = xyz_to_sbu("custom_sbus.xyz")
    >>> print(sbus.keys())  # dict_keys(['SBU_1', 'SBU_2', ...])
    """
    try:
        xyz = XYZ.from_file(path)
    except (IndexError, ValueError) as exc:
        # pymatgen surfaces malformed content as bare IndexError/ValueError
        raise ValueError(f"Malformed XYZ file '{path}': {exc}") from exc
    names = get_xyz_names(path)
    molecules = xyz.all_molecules
    if len(molecules) != len(names):
        raise ValueError(
            f"Malformed XYZ file '{path}': parsed {len(molecules)} "
            f"structure(s) but found {len(names)} comment line(s)"
        )
    return {
        name: Fragment(atoms=molecule, name=name)
        for molecule, name in zip(molecules, names, strict=True)
    }


@functools.cache
def _uff_types_for_prefix(prefix: str) -> tuple[UFFType, ...]:
    """All UFF4MOF types whose symbol starts with the element prefix."""
    return tuple(t for t in UFF4MOF if t.symbol.startswith(prefix))


def load_uff_lib(mol: Molecule | Structure) -> tuple[tuple[UFFType, ...], list[str]]:
    """Load UFF force field parameters relevant to a molecule.

    Extracts UFF4MOF parameters for elements present in the molecule,
    used for determining atom types and bond length cutoffs.

    Parameters
    ----------
    mol : Molecule or Structure
        A pymatgen Molecule (or Structure, for deconstruction).

    Returns
    -------
    tuple[tuple[UFFType, ...], list[str]]
        A tuple containing:
        - UFF parameter entries for the relevant elements.
        - List of UFF symbol prefixes for each atom in the molecule.
    """
    uff_symbs = [
        s.symbol if len(s.symbol) == 2 else f"{s.symbol}_" for s in mol.species
    ]
    uff_lib = tuple(
        entry
        for prefix in sorted(set(uff_symbs))
        for entry in _uff_types_for_prefix(prefix)
    )
    return uff_lib, uff_symbs


def find_element_cutoffs(
    uff_lib: tuple[UFFType, ...], uff_symbs: list[str]
) -> dict[tuple[str, str], float]:
    """Calculate bond distance cutoffs from UFF atomic radii.

    Computes maximum bonding distances for all element pairs using the
    sum of their UFF radii.

    Parameters
    ----------
    uff_lib : tuple[UFFType, ...]
        UFF parameter entries including radii.
    uff_symbs : list[str]
        List of UFF symbol prefixes for atoms in the molecule.

    Returns
    -------
    dict[tuple[str, str], float]
        Dictionary mapping element pairs to bond distance cutoffs.

    Examples
    --------
    >>> uff_lib, uff_symbs = load_uff_lib(molecule)
    >>> cutoffs = find_element_cutoffs(uff_lib, uff_symbs)
    >>> print(cutoffs[("C", "N")])  # Bond cutoff for C-N
    """
    # largest radius among each element's UFF types
    max_radius: dict[str, float] = {}
    for entry in uff_lib:
        prefix = entry.symbol[:2]
        max_radius[prefix] = max(max_radius.get(prefix, 0.0), entry.radius)
    cuts = {}
    # element pairs, not atom pairs: uff_symbs is per-atom, and pairing
    # atoms would make this quadratic in atom count for identical output
    elements = sorted(set(uff_symbs))
    for e0, e1 in itertools.product(elements, elements):
        cuts[(e0.strip("_"), e1.strip("_"))] = max_radius[e0] + max_radius[e1]
    return cuts


def find_mmtypes(
    molgraph: MoleculeGraph, uff_lib: tuple[UFFType, ...], uff_symbs: list[str]
) -> list[str]:
    """Determine UFF atom types from molecular connectivity.

    Assigns UFF atom types based on coordination number and bond angles,
    following the UFF4MOF parameterization scheme.

    Parameters
    ----------
    molgraph : MoleculeGraph
        A pymatgen MoleculeGraph with connectivity information.
    uff_lib : tuple[UFFType, ...]
        UFF parameter entries to choose from.
    uff_symbs : list[str]
        List of UFF symbol prefixes for atoms in the molecule.

    Returns
    -------
    list[str]
        List of assigned UFF atom type symbols, one per atom, in atom order.
    """
    mmtypes = []
    for i, symb in enumerate(uff_symbs):
        conn = molgraph.get_connected_sites(i)
        ncoord = len(conn)
        atom_compat = [t for t in uff_lib if t.symbol.startswith(symb)]
        if len(atom_compat) == 1:
            mmtypes.append(atom_compat[0].symbol)
            continue
        if ncoord >= 2:
            c0 = molgraph.molecule.sites[i].coords
            # get the two closest sites
            s1, s2 = sorted(conn, key=lambda k: k.dist)[:2]
            c1, c2 = s1.site.coords, s2.site.coords
            c01 = c1 - c0
            c02 = c2 - c0
            cosine_angle = np.dot(c01, c02) / (
                np.linalg.norm(c01) * np.linalg.norm(c02)
            )
            # clip to the valid arccos domain: collinear neighbors can
            # give |cos| marginally above 1.0 from floating point error
            angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        else:
            angle = 180.0
        coordinations_compat = [t for t in atom_compat if t.coordination == ncoord]
        if len(coordinations_compat) == 1:
            mmtypes.append(coordinations_compat[0].symbol)
            continue
        if not coordinations_compat:
            # problem with the coordinations. use angles
            coordinations_compat = atom_compat
        # TODO: if < 10% diff in angle error, use radii error
        best = min(coordinations_compat, key=lambda t: abs(t.angle - angle))
        mmtypes.append(best.symbol)
    return mmtypes


def fragment_to_molgraph(fragment: Fragment) -> MoleculeGraph:
    """Convert a Fragment to a pymatgen MoleculeGraph.

    Creates a molecular connectivity graph from a Fragment, with UFF
    atom types assigned and dummy atoms removed. Tags from dummies are
    transferred to their connected atoms.

    Parameters
    ----------
    fragment : Fragment
        The molecular fragment to convert.

    Returns
    -------
    MoleculeGraph
        A pymatgen MoleculeGraph with connectivity and UFF types.
    """
    mol = fragment.atoms.copy()
    dummies_idx = fragment.atoms.indices_from_symbol("X")
    mol.replace_species({"X": "H"})
    # setting up UFF type analysis
    uff_lib, uff_symbs = load_uff_lib(mol)
    # obtaining cutoffs from the maximum UFF radius
    strategy = EconNN(tol=BOND_TOLERANCE, use_fictive_radius=True, cutoff=BOND_CUTOFF)
    mg = MoleculeGraph.from_local_env_strategy(mol, strategy=strategy)
    # add mmtypes
    mmtypes = find_mmtypes(molgraph=mg, uff_lib=uff_lib, uff_symbs=uff_symbs)
    for i, mmtype in enumerate(mmtypes):
        mg.molecule[i].properties["ufftype"] = mmtype
    # transfer tags from dummies
    if "tags" in mg.molecule.site_properties:
        for dummy_idx in dummies_idx:
            tag = mg.molecule[dummy_idx].properties["tags"]
            for site in mg.get_connected_sites(dummy_idx):
                mg.molecule[site.index].properties["tags"] = tag
    # remove dummies
    mg.remove_nodes(list(dummies_idx))
    return mg


def fragments_to_networkx(
    fragments: list[Fragment],
    cell: np.ndarray | None = None,
    slots: list[int] | None = None,
) -> networkx.Graph:
    """Combine multiple fragments into a single networkx Graph.

    Creates a molecular graph representing the full framework structure,
    with inter-fragment bonds formed between atoms with matching tags.

    Parameters
    ----------
    fragments : list[Fragment]
        List of aligned Fragment objects to combine.
    cell : np.ndarray or None, optional
        3x3 cell matrix for periodic structures. Stored as graph attribute.
    slots : list[int] or None, optional
        Topology slot index per fragment; falls back to the fragment's
        position in the list. Recorded per node so post-build editing
        (rotation, defects, functionalization) can recover which atoms
        belong to which placed SBU.

    Returns
    -------
    networkx.Graph
        Molecular graph with node attributes (symbol, coord, tag,
        ufftype, slot, sbu) and edge attributes (bond_order).

    Examples
    --------
    >>> graph = fragments_to_networkx(aligned_fragments, cell=topology.cell.matrix)
    >>> print(f"Atoms: {graph.number_of_nodes()}, Bonds: {graph.number_of_edges()}")
    """
    if slots is None:
        slots = list(range(len(fragments)))
    full_graph = networkx.Graph(cell=cell)
    subgraphs = [fragment_to_molgraph(f) for f in fragments]
    offset = 0
    for slot, fragment, subgraph in zip(slots, fragments, subgraphs, strict=True):
        # obtaining non-standard cutoffs from the maximum UFF radius
        bond_lengths = find_element_cutoffs(*load_uff_lib(subgraph.molecule))
        this_len = len(subgraph.molecule)
        # add the nodes
        species = [s.symbol for s in subgraph.molecule.species]
        coords = subgraph.molecule.cart_coords
        tags = subgraph.molecule.site_properties["tags"]
        mmtypes = subgraph.molecule.site_properties["ufftype"]
        for i, (s, c, t, m) in enumerate(
            zip(species, coords, tags, mmtypes, strict=True)
        ):
            full_graph.add_node(
                i + offset,
                symbol=s,
                coord=c,
                tag=t,
                ufftype=m,
                slot=slot,
                sbu=fragment.name,
            )
        for i in range(this_len):
            for j, ij_dist in [
                (s.index, s.dist) for s in subgraph.get_connected_sites(i)
            ]:
                # evaluate bond orders
                uff_bl = bond_lengths[(species[i], species[j])]
                bo = get_bond_order(
                    species[i], species[j], ij_dist, tol=0.2, default_bl=uff_bl
                )
                full_graph.add_edge(i + offset, j + offset, bond_order=bo)
        offset += this_len
    # interfragment edges: atoms sharing a positive tag are the two
    # sides of a blueprint dummy and bond together
    nodes_by_tag: dict[int, list[int]] = defaultdict(list)
    for node, data in full_graph.nodes(data=True):
        if data["tag"] > 0:
            nodes_by_tag[data["tag"]].append(node)
    for tag, nodes in nodes_by_tag.items():
        if len(nodes) == 2:
            full_graph.add_edge(nodes[0], nodes[1], bond_order=1.0)
        elif len(nodes) > 2:
            logger.warning(
                f"Tag {tag} present on {len(nodes)} atoms; bonding the first two."
            )
            full_graph.add_edge(nodes[0], nodes[1], bond_order=1.0)
    return full_graph


def view_graph(graph: networkx.Graph) -> None:
    """Visualize a molecular graph using ASE's viewer.

    Opens an interactive 3D viewer displaying the structure stored in
    the networkx graph with periodic boundary conditions.

    Parameters
    ----------
    graph : networkx.Graph
        Molecular graph with 'cell' graph attribute and node attributes
        'symbol' and 'coord'.

    Notes
    -----
    Requires ASE to be installed with a working visualization backend.
    """
    from ase import Atom, Atoms
    from ase.visualize import view

    at = Atoms(cell=graph.graph["cell"], pbc=(True, True, True))
    # sorted node id order, like every Framework view: insertion order
    # is not guaranteed to match node ids after graph editing
    for node in sorted(graph):
        d = graph.nodes[node]
        at.append(Atom(d["symbol"], d["coord"]))
    view(at)


def networkx_to_gulp(
    graph: networkx.Graph, name: str = "MOF", write_to_file: bool = True
) -> str:
    """Export a molecular graph to GULP input format.

    Generates a GULP input file for geometry optimization using the
    UFF4MOF force field, including cell parameters, atomic coordinates,
    and bond connectivity.

    Parameters
    ----------
    graph : networkx.Graph
        Molecular graph with cell, coordinates, and UFF type information.
    name : str, optional
        Base name for output files. Default is "MOF".
    write_to_file : bool, optional
        If True, writes to ``{name}.gin`` in current directory. Default is True.

    Returns
    -------
    str
        The complete GULP input file as a string.

    Examples
    --------
    >>> gulp_input = networkx_to_gulp(graph, name="my_mof", write_to_file=True)
    >>> # Creates my_mof.gin in current directory
    """
    logger.info("Creating Gulp file from graph.")
    lines = []

    # Header and cell vectors
    cell = graph.graph["cell"]
    lines.append("opti conp molmec noautobond cartesian noelectrostatics ocell")
    lines.append("vectors")
    lines.append(f"{cell[0][0]:>.3f} {cell[0][1]:>.3f} {cell[0][2]:>.3f}")
    lines.append(f"{cell[1][0]:>.3f} {cell[1][1]:>.3f} {cell[1][2]:>.3f}")
    lines.append(f"{cell[2][0]:>.3f} {cell[2][1]:>.3f} {cell[2][2]:>.3f}")
    lines.append("cartesian")

    # Build atom type mapping; sorted so the species labels (C0, C1...)
    # are reproducible run to run instead of following set hash order
    mmset = sorted({(d["symbol"], d["ufftype"]) for _, d in graph.nodes(data=True)})
    mmdict = {u: f"{s}{i}" for i, (s, u) in enumerate(mmset)}

    # Atomic coordinates, in sorted node id order: the connect records
    # below refer to atoms by node id + 1, so line i must be node i
    for node in sorted(graph):
        d = graph.nodes[node]
        x, y, z = d["coord"]
        s = mmdict[d["ufftype"]]
        lines.append(f"{s:<4} core {x:>15.8f} {y:>15.8f} {z:>15.8f}")

    lines.append("")

    # Bond connectivity
    for b0, b1, bd in graph.edges(data=True):
        bond_order = bd.get("bond_order", 1.0)
        if bond_order < 0.9:
            bo = "half"
        elif bond_order < 1.1:
            bo = "single"
        elif bond_order < 1.8:
            bo = "aromatic"
        elif bond_order < 2.1:
            bo = "double"
        else:
            bo = "triple"
        lines.append(f"connect {b0 + 1:<4} {b1 + 1:<4} {bo}")

    # Species mapping
    lines.append("")
    lines.append("species")
    for u, s in mmdict.items():
        lines.append(f"{s:<5} {u:<5}")

    # Footer
    lines.append("")
    lines.append("library uff4mof")
    lines.append("")
    lines.append(f"output movie xyz {name}.xyz")
    lines.append(f"output cif {name}.cif")

    out_string = "\n".join(lines)

    # Write to file
    if write_to_file:
        output_path = Path.cwd() / f"{name}.gin"
        logger.info(f" [x] Saved to {output_path}")
        output_path.write_text(out_string)

    return out_string
