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

import itertools
import logging
import os
import re
import warnings
from collections import defaultdict
from collections.abc import Iterable
from itertools import count, groupby

import networkx
import numpy as np
import pandas
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import EconNN
from pymatgen.core.bonds import get_bond_order
from pymatgen.core.structure import Molecule
from pymatgen.io.xyz import XYZ
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

import autografs.data
from autografs.structure import Fragment


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# suppress false positive SettingWithCopyWarning
pandas.options.mode.chained_assignment = None


def format_indices(iterable: Iterable[int]) -> str:
    lst = list(iterable)
    if len(lst) > 1:
        return f"{lst[0]}-{lst[1]}"
    else:
        return f"{lst[0]}"


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
    new_dict = defaultdict(list)
    for k, v in mappings.items():
        new_dict[v].append(k)
    out_mappings = []
    for k, v in new_dict.items():
        v = sorted(v)
        grouped_indices = groupby(v, lambda n, c=count(): n - next(c))
        v = ",".join(format_indices(g) for _, g in grouped_indices)
        out_mappings.append(f"{k} : {v}")
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
    names = []
    white_space = r"[ \t\r\f\v]"
    natoms_line = white_space + r"*\d+" + white_space + r"*\n"
    with open(path, "r") as xyz_file:
        data = re.split(natoms_line, xyz_file.read())
        comment_lines = [
            list(y)[0] for x, y in itertools.groupby(data, lambda z: z == "") if not x
        ]
        for cl in comment_lines:
            if "name=" in cl:
                name = cl.split("name=")[1].split(" ")[0]
                names.append(name)
            else:
                names.append("Unnamed")
    return names


def xyz_to_sbu(path: str) -> dict[str, Fragment]:
    """Load Secondary Building Units from an XYZ file.

    Reads a multi-structure XYZ file and creates Fragment objects for
    each structure. Dummy atoms ("X") define connection points and are
    used to determine the point group symmetry.

    Parameters
    ----------
    path : str
        Path to the XYZ file containing SBU structures.

    Returns
    -------
    dict[str, Fragment]
        Dictionary mapping SBU names to Fragment objects.

    Examples
    --------
    >>> sbus = xyz_to_sbu("custom_sbus.xyz")
    >>> print(sbus.keys())  # dict_keys(['SBU_1', 'SBU_2', ...])
    """
    xyz = XYZ.from_file(path)
    names = get_xyz_names(path)
    sbu = {}
    for molecule, name in zip(xyz.all_molecules, names):
        dummies_idx = molecule.indices_from_symbol("X")
        symmetric_mol = Molecule(
            [
                "H",
            ]
            * len(dummies_idx),
            [molecule[idx].coords for idx in dummies_idx],
            charge=len(dummies_idx),
        )
        symmetry = PointGroupAnalyzer(symmetric_mol, tolerance=0.1)
        sbu[name] = Fragment(atoms=molecule, symmetry=symmetry, name=name)
    return sbu


def load_uff_lib(mol: Molecule) -> tuple[pandas.DataFrame, list[str]]:
    """Load UFF force field parameters relevant to a molecule.

    Extracts UFF4MOF parameters for elements present in the molecule,
    used for determining atom types and bond length cutoffs.

    Parameters
    ----------
    mol : Molecule
        A pymatgen Molecule object.

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        A tuple containing:
        - DataFrame with UFF parameters for relevant elements.
        - List of UFF symbol prefixes for each atom in the molecule.
    """
    uff_symbs = [
        s.symbol if len(s.symbol) == 2 else f"{s.symbol}_" for s in mol.species
    ]
    path = os.path.join(autografs.data.__path__[0], "uff4mof.csv")
    full_lib = pandas.read_csv(path)
    uff_lib = pandas.concat(
        [full_lib[full_lib.symbol.str.startswith(s)] for s in set(uff_symbs)]
    )
    return uff_lib, uff_symbs


def find_element_cutoffs(
    uff_lib: pandas.DataFrame, uff_symbs: list[str]
) -> dict[tuple[str, str], float]:
    """Calculate bond distance cutoffs from UFF atomic radii.

    Computes maximum bonding distances for all element pairs using the
    sum of their UFF radii.

    Parameters
    ----------
    uff_lib : pandas.DataFrame
        DataFrame containing UFF parameters including radii.
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
    radii = uff_lib[["symbol", "radius"]]
    radii.symbol = radii.symbol.str[:2]
    radii = radii.groupby("symbol").max()
    cuts = {}
    for e0, e1 in itertools.product(uff_symbs, uff_symbs):
        r0 = radii.loc[e0, "radius"]
        r1 = radii.loc[e1, "radius"]
        cuts[(e0.strip("_"), e1.strip("_"))] = r0 + r1
    return cuts


def find_mmtypes(
    molgraph: MoleculeGraph, uff_lib: pandas.DataFrame, uff_symbs: list[str]
) -> list[str]:
    """Determine UFF atom types from molecular connectivity.

    Assigns UFF atom types based on coordination number and bond angles,
    following the UFF4MOF parameterization scheme.

    Parameters
    ----------
    molgraph : MoleculeGraph
        A pymatgen MoleculeGraph with connectivity information.
    uff_lib : pandas.DataFrame
        DataFrame containing UFF parameters.
    uff_symbs : list[str]
        List of UFF symbol prefixes for atoms in the molecule.

    Returns
    -------
    list[str]
        List of assigned UFF atom type symbols.
    """
    mmtypes = []
    for i, symb in enumerate(uff_symbs):
        conn = molgraph.get_connected_sites(i)
        ncoord = len(conn)
        atom_compat = uff_lib[uff_lib.symbol.str.startswith(symb)]
        molgraph.molecule[i].properties["ufftype"] = atom_compat.symbol.values[0]
        if len(atom_compat) == 1:
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
            angle = np.degrees(np.arccos(cosine_angle))
        else:
            angle = 180.0
        coordinations_compat = atom_compat[atom_compat.coordination == ncoord]
        if len(coordinations_compat) == 1:
            molgraph.molecule[i].properties["ufftype"] = (
                coordinations_compat.symbol.values[0]
            )
            continue
        elif len(coordinations_compat) == 0:
            # problem with the coordinations. use angles
            coordinations_compat = atom_compat
        coordinations_compat["angle_diff"] = coordinations_compat.angle - angle
        best_angle = coordinations_compat.sort_values(by="angle_diff").iloc[0]
        # TODO: if < 10% diff in angle error, use radii error
        mmtypes.append(best_angle.symbol)
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
    strategy = EconNN(tol=0.3, use_fictive_radius=True, cutoff=10.0)
    mg = MoleculeGraph.with_local_env_strategy(mol, strategy=strategy)
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
    fragments: list[Fragment], cell: np.ndarray | None = None
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

    Returns
    -------
    networkx.Graph
        Molecular graph with node attributes (symbol, coord, tag, ufftype)
        and edge attributes (bond_order).

    Examples
    --------
    >>> graph = fragments_to_networkx(aligned_fragments, cell=topology.cell.matrix)
    >>> print(f"Atoms: {graph.number_of_nodes()}, Bonds: {graph.number_of_edges()}")
    """
    full_graph = networkx.Graph(cell=cell)
    subgraphs = [fragment_to_molgraph(f) for f in fragments]
    offset = 0
    for subgraph in subgraphs:
        # obtaining non-standard cutoffs from the maximum UFF radius
        bond_lengths = find_element_cutoffs(*load_uff_lib(subgraph.molecule))
        this_len = len(subgraph.molecule)
        # add the nodes
        species = [s.symbol for s in subgraph.molecule.species]
        coords = subgraph.molecule.cart_coords
        tags = subgraph.molecule.site_properties["tags"]
        mmtypes = subgraph.molecule.site_properties["ufftype"]
        for i, (s, c, t, m) in enumerate(zip(species, coords, tags, mmtypes)):
            full_graph.add_node(i + offset, symbol=s, coord=c, tag=t, ufftype=m)
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
    # interfragment edges
    for (n0, d0), (n1, d1) in itertools.combinations(full_graph.nodes(data=True), 2):
        if d0["tag"] == d1["tag"] > 0:
            full_graph.add_edge(n0, n1)
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
    for _, d in graph.nodes(data=True):
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
    out_string = ""
    out_string += "opti conp molmec noautobond cartesian noelectrostatics ocell\n"
    # out_string += 'maxcyc 1000\nswitch bfgs gnorm 0.5\n'
    cell = graph.graph["cell"]
    out_string += "vectors\n"
    out_string += "{0:>.3f} {1:>.3f} {2:>.3f}\n".format(*cell[0])
    out_string += "{0:>.3f} {1:>.3f} {2:>.3f}\n".format(*cell[1])
    out_string += "{0:>.3f} {1:>.3f} {2:>.3f}\n".format(*cell[2])
    out_string += "{0}\n".format("cartesian")
    mmset = list(set([(d["symbol"], d["ufftype"]) for _, d in graph.nodes(data=True)]))
    mmdict = {u: f"{s}{i}" for i, (s, u) in enumerate(mmset)}
    for _, d in graph.nodes(data=True):
        x, y, z = d["coord"]
        s = mmdict[d["ufftype"]]
        out_string += f"{s:<4} core {x:>15.8f} {y:>15.8f} {z:>15.8f}\n"
    out_string += "\n"
    for b0, b1, bd in graph.edges(data=True):
        # this line is for the dummy to dummy bonds
        if "bond_order" not in bd:
            bd["bond_order"] = 1.0
        # the strings depend on the bond order number here
        if bd["bond_order"] < 0.9:
            bo = "half"
        elif 0.9 < bd["bond_order"] < 1.1:
            bo = "single"
        elif 1.1 < bd["bond_order"] < 1.8:
            bo = "aromatic"
        elif 1.8 < bd["bond_order"] < 2.1:
            bo = "double"
        else:
            bo = "triple"
        out_string += f"connect {b0+1:<4} {b1+1:<4} {bo}\n"
    out_string += "\nspecies\n"
    for u, s in mmdict.items():
        out_string += f"{s:<5} {u:<5}\n"
    out_string += "\nlibrary uff4mof\n"
    out_string += "\n"
    out_string += f"output movie xyz {name}.xyz\n"
    out_string += f"output cif {name}.cif\n"
    # write to the file
    if write_to_file:
        logger.info(f" [x] Saved to {name}.gin")
        with open(os.path.join(os.getcwd(), f"{name}.gin"), "w") as gin:
            gin.write(out_string)
    return out_string
