"""
Module docstrings are similar to class docstrings. Instead of classes and class methods being documented,
it’s now the module and any functions found within. Module docstrings are placed at the top of the file
even before any imports. Module docstrings should include the following:

A brief description of the module and its purpose
A list of any classes, exception, functions, and any other objects exported by the module
"""
import itertools
import os
import re
import logging
from collections import defaultdict
from typing import List, Optional, Tuple, Dict, Iterable
from itertools import groupby, count

import networkx
import numpy
import pandas
import pymatgen
import warnings
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
        return f'{lst[0]}-{lst[1]}'
    else:
        return f'{lst[0]}'


def format_mappings(mappings: Dict[int, str]) -> str:
    """Helper function concatenating the mappings into a
    logging-friendly format for readability
    TODO
    """
    new_dict = defaultdict(list)
    for k, v in mappings.items():
        new_dict[v].append(k)
    out_mappings = []
    for k, v in new_dict.items():
        v = sorted(v)
        grouped_indices = groupby(v, lambda n, c=count(): n - next(c))
        v = ','.join(format_indices(g) for _, g in grouped_indices)
        out_mappings.append(f"{k} : {v}")
    return "; ".join(out_mappings)


def get_xyz_names(path: str) -> List[str]:
    """
    [summary]

    Parameters
    ----------
    path : str
        [description]

    Returns
    -------
    str
        [description]
    """
    names = []
    white_space = r"[ \t\r\f\v]"
    natoms_line = white_space + r"*\d+" + white_space + r"*\n"
    with open(path, "r") as xyz_file:
        data = re.split(natoms_line, xyz_file.read())
        comment_lines = [list(y)[0] for x, y in itertools.groupby(data, lambda z: z == "") if not x]
        for cl in comment_lines:
            if "name=" in cl:
                name = cl.split("name=")[1].split(" ")[0]
                names.append(name)
            else:
                names.append("Unnamed")
    return names


def xyz_to_sbu(path: str) -> Dict[str, Fragment]:
    """
    [summary]

    Parameters
    ----------
    path : str
        [description]

    Returns
    -------
    Fragment
        [description]
    """
    xyz = XYZ.from_file(path)
    names = get_xyz_names(path)
    sbu = {}
    for molecule, name in zip(xyz.all_molecules, names):
        dummies_idx = molecule.indices_from_symbol("X")
        symmetric_mol = Molecule(["H", ] * len(dummies_idx),
                                 [molecule[idx].coords for idx in dummies_idx],
                                 charge=len(dummies_idx))
        symmetry = PointGroupAnalyzer(symmetric_mol, tolerance=0.1)
        sbu[name] = Fragment(atoms=molecule, symmetry=symmetry, name=name)
    return sbu


def load_uff_lib(mol: Molecule) -> Tuple[pandas.DataFrame, List[str]]:
    """
    [summary]

    Parameters
    ----------
    mol : Molecule
        [description]

    Returns
    -------
    Tuple[pandas.DataFrame, List[str]]
        [description]
    """
    uff_symbs = [s.symbol if len(s.symbol)==2 else f"{s.symbol}_" for s in mol.species]
    path = os.path.join(autografs.data.__path__[0], "uff4mof.csv")
    full_lib = pandas.read_csv(path)
    uff_lib = pandas.concat([full_lib[full_lib.symbol.str.startswith(s)] for s in set(uff_symbs)])
    return uff_lib, uff_symbs


def find_element_cutoffs(uff_lib: pandas.DataFrame,
                         uff_symbs: List[str]
                         ) -> Dict[Tuple[str, str], float]:
    """
    [summary]

    Parameters
    ----------
    uff_lib : pandas.DataFrame
        [description]
    uff_symbs : List[str]
        [description]

    Returns
    -------
    pandas.DataFrame
        [description]
    """
    radii = uff_lib[["symbol", "radius"]]
    radii.symbol = radii.symbol.str[:2]
    radii = radii.groupby("symbol").max()
    cuts = {}
    for e0, e1 in itertools.product(uff_symbs, uff_symbs):
        r0 = radii.loc[e0, "radius"]
        r1 = radii.loc[e1, "radius"]
        cuts[(e0.strip("_"), e1.strip("_"))] =  r0 + r1
    return cuts


def find_mmtypes(molgraph: MoleculeGraph,
                 uff_lib: pandas.DataFrame,
                 uff_symbs: List[str]
                 ) -> List[str]:
    """
    [summary]

    Parameters
    ----------
    molgraph : MoleculeGraph
        [description]
    uff_lib : pandas.DataFrame
        [description]
    uff_symbs : List[str]
        [description]

    Returns
    -------
    List[str]
        [description]
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
            cosine_angle = numpy.dot(c01, c02) / (numpy.linalg.norm(c01) * numpy.linalg.norm(c02))
            angle = numpy.degrees(numpy.arccos(cosine_angle))
        else:
            angle = 180.0
        coordinations_compat = atom_compat[atom_compat.coordination == ncoord]
        if len(coordinations_compat) == 1:
            molgraph.molecule[i].properties["ufftype"] = coordinations_compat.symbol.values[0]
            continue
        elif len(coordinations_compat) == 0:
            # problem with the coordinations. use angles
            coordinations_compat = atom_compat
        coordinations_compat["angle_diff"] = coordinations_compat.angle - angle
        best_angle = coordinations_compat.sort_values(by='angle_diff').iloc[0]
        # TODO: if < 10% diff in angle error, use radii error
        mmtypes.append(best_angle.symbol)
    return mmtypes


def fragment_to_molgraph(fragment: Fragment) -> MoleculeGraph:
    """
    [summary]

    Parameters
    ----------
    fragment : Fragment
        [description]

    Returns
    -------
    pymatgen.analysis.graphs.MoleculeGraph
        [description]
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


def fragments_to_networkx(fragments: List[Fragment],
                          cell: Optional[numpy.ndarray] = None
                          ) -> networkx.Graph:
    """
    [summary]

    Parameters
    ----------
    fragments : List[Fragment]
        [description]
    cell : Optional[numpy.ndarray], optional
        [description], by default None

    Returns
    -------
    networkx.Graph
        [description]
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
            for j, ij_dist in [(s.index, s.dist) for s in subgraph.get_connected_sites(i)]:
                # evaluate bond orders
                uff_bl = bond_lengths[(species[i], species[j])]
                bo = get_bond_order(species[i], species[j], ij_dist, tol=0.2, default_bl=uff_bl)
                full_graph.add_edge(i + offset, j + offset, bond_order=bo)
        offset += this_len
    # interfragment edges
    for (n0, d0), (n1, d1) in itertools.combinations(full_graph.nodes(data=True), 2):
        if d0["tag"] == d1["tag"] > 0:
            full_graph.add_edge(n0, n1)
    return full_graph


def view_graph(graph: networkx.Graph) -> None:
    """
    [summary]

    Parameters
    ----------
    graph : networkx.Graph
        [description]
    """
    from ase import Atom, Atoms
    from ase.visualize import view
    at = Atoms(cell=graph.graph["cell"], pbc=(True, True, True))
    for _, d in graph.nodes(data=True):
        at.append(Atom(d["symbol"], d["coord"]))
    view(at)


def networkx_to_gulp(graph: networkx.Graph,
                     name: str = "MOF",
                     write_to_file: bool = True
                     ) -> str:
    """
    [summary]

    Parameters
    ----------
    graph : networkx.Graph
        [description]
    name : str, optional
        [description], by default "MOF"
    write_to_file : bool, optional
        [description], by default True

    Returns
    -------
    str
        [description]
    """
    logger.info("Creating Gulp file from graph.")
    out_string = ""
    out_string += 'opti conp molmec noautobond cartesian noelectrostatics ocell\n'
    # out_string += 'maxcyc 1000\nswitch bfgs gnorm 0.5\n'
    cell = graph.graph["cell"]
    out_string += 'vectors\n'
    out_string += '{0:>.3f} {1:>.3f} {2:>.3f}\n'.format(*cell[0])
    out_string += '{0:>.3f} {1:>.3f} {2:>.3f}\n'.format(*cell[1])
    out_string += '{0:>.3f} {1:>.3f} {2:>.3f}\n'.format(*cell[2])
    out_string += '{0}\n'.format('cartesian')
    mmset = list(set([(d["symbol"], d["ufftype"]) for _, d in graph.nodes(data=True)]))
    mmdict = {u: f"{s}{i}" for i, (s, u) in enumerate(mmset)}
    for _, d in graph.nodes(data=True):
        x, y, z = d["coord"]
        s = mmdict[d["ufftype"]]
        out_string += f"{s:<4} core {x:>15.8f} {y:>15.8f} {z:>15.8f}\n"
    out_string += '\n'
    for b0, b1, bd in graph.edges(data=True):
        # this line is for the dummy to dummy bonds
        if "bond_order" not in bd: bd["bond_order"] = 1.0
        # the strings depend on the bond order number here
        if bd["bond_order"] < 0.9: bo = "half"
        elif 0.9 < bd["bond_order"] < 1.1: bo = "single"
        elif 1.1 < bd["bond_order"] < 1.8: bo = "aromatic"
        elif 1.8 < bd["bond_order"] < 2.1: bo = "double"
        else: bo = "triple"
        out_string += f"connect {b0+1:<4} {b1+1:<4} {bo}\n"
    out_string += '\nspecies\n'
    for u, s in mmdict.items():
        out_string += f"{s:<5} {u:<5}\n"
    out_string += '\nlibrary uff4mof\n'
    out_string += '\n'
    out_string += f'output movie xyz {name}.xyz\n'
    out_string += f'output cif {name}.cif\n'
    # write to the file
    if write_to_file:
        logger.info(f" [x] Saved to {name}.gin")
        with open(os.path.join(os.getcwd(), f"{name}.gin"), "w") as gin:
            gin.write(out_string)
    return out_string
