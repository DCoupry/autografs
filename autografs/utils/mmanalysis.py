#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
# Copyright : see accompanying license files for details

__author__  = "Damien Coupry"
__credits__ = ["Prof. Matthew Addicoat"]
__license__ = "MIT"
__maintainer__ = "Damien Coupry"
__version__ = '2.0.4'
__status__  = "beta"

import numpy
import os
from scipy.sparse import csgraph

from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase import Atom
import ase
from itertools import combinations
from ase.visualize import view

from autografs.utils import __data__

def is_metal(symbols : list) -> bool:
    """Check wether symbols in a list are metals"""
    symbols = numpy.array([symbols]).flatten()
    metals = ['Li','Be','Al','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
              'Ga','Ge','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In',
              'Sn','Sb','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho',
              'Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg',
              'Tl','Pb','Bi','Po','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am',
              'Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs',
              'Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']
    return numpy.isin(symbols,metals)

def is_alkali(symbols : list) -> bool:
    """Check wether symbols in a list are alkali"""
    symbols = numpy.array([symbols]).flatten()
    alkali = ['Li','Be','Na','Mg','K','Ca','Rb','Sr','Cs','Ba']
    return numpy.isin(symbols,alkali)


def read_uff_library(library : str = "uff4mof") -> numpy.ndarray:
    """Return the UFF library as a numpy array"""
    uff_file = os.path.join(__data__,"uff/{0}.csv".format(library))
    with open(uff_file,"r") as lib:
        lines = [l.split(",") for l in lib.read().splitlines() if not l.startswith("#")]
        #symbol,radius,angle,coordination
        ufflib   = {s:numpy.array([r,a,c],dtype=numpy.float32) for s,r,a,c in lines}
    return ufflib

def get_bond_matrix(sbu : ase.Atoms) -> numpy.ndarray:
    """Guesses the bond order in neighbourlist based on covalent radii 
    the radii for BO > 1 are extrapolated by removing 0.1 Angstroms by order 
    see Beatriz Cordero, Veronica Gomez, Ana E. Platero-Prats, Marc Reves, Jorge Echeverria,
    Eduard Cremades, Flavia Barragan and Santiago Alvarez (2008). 
    "Covalent radii revisited". 
    Dalton Trans. (21): 2832-2838 
    http://dx.doi.org/10.1039/b801115j
    """
    # first guess
    bonds     = numpy.zeros((len(sbu),len(sbu)))
    symbols   = numpy.array(sbu.get_chemical_symbols())
    numbers   = numpy.array(sbu.get_atomic_numbers())
    positions = numpy.array(sbu.get_positions())
    BO1 = numpy.array([covalent_radii[n] if n>0 else 0.35 for n in numbers])
    BO2 = BO1 - 0.15
    BO3 = BO2 - 0.15
    nl1 = NeighborList(cutoffs=BO1,bothways=True,self_interaction=False,skin=0.1)
    nl2 = NeighborList(cutoffs=BO2,bothways=True,self_interaction=False,skin=0.1)
    nl3 = NeighborList(cutoffs=BO3,bothways=True,self_interaction=False,skin=0.1)
    nl1.build(sbu); nl2.build(sbu); nl3.build(sbu)
    for atom in sbu:
        i1, _ = nl1.get_neighbors(atom.index)
        i2, _ = nl2.get_neighbors(atom.index)
        i3, _ = nl3.get_neighbors(atom.index)
        bonds[atom.index,i1] = 1.0
        bonds[atom.index,i2] = 2.0
        bonds[atom.index,i3] = 3.0
    # cleanup with particular cases
    # identify particular atoms
    hydrogens = numpy.where(symbols=="H")[0]
    metals    = numpy.where(is_metal(symbols))[0]
    alkali    = numpy.where(is_alkali(symbols))[0]
    # the rest is dubbed "organic"
    organic   = numpy.ones(bonds.shape)
    organic[hydrogens,:] = False
    organic[metals   ,:] = False
    organic[alkali   ,:] = False
    organic[:,hydrogens] = False
    organic[:,metals   ] = False
    organic[:,alkali   ] = False    
    organic = numpy.where(organic)[0]
    # Hydrogen has BO of 1
    bonds_h = bonds[hydrogens]
    bonds_h[bonds_h>1.0] = 1.0
    bonds[hydrogens,:] = bonds_h
    bonds[:,hydrogens] = bonds_h.T
    #Metal-Metal bonds: if no special case, nominal bond
    ix  = numpy.ix_(metals,metals)
    bix = bonds[ix]
    bix[numpy.nonzero(bix)] = 0.25
    bonds[ix] = bix
    # no H-Metal bonds
    ix = numpy.ix_(metals,hydrogens)
    bonds[ix] = 0.0
    ix = numpy.ix_(hydrogens,metals)
    bonds[ix] = 0.0
    # no alkali-alkali bonds
    ix = numpy.ix_(alkali,alkali)
    bonds[ix] = 0.0
    # no alkali-metal bonds
    ix = numpy.ix_(metals,alkali)
    bonds[ix] = 0.0
    ix = numpy.ix_(alkali,metals)
    bonds[ix] = 0.0    
    # metal-organic is coordination bond
    ix  = numpy.ix_(metals,organic)
    bix = bonds[ix]
    bix[numpy.nonzero(bix)] = 0.5
    bonds[ix] = bix
    ix = numpy.ix_(organic,metals)
    bix = bonds[ix]
    bix[numpy.nonzero(bix)] = 0.5
    bonds[ix] = bix
    # aromaticity and rings
    rings = []
    # first, use the compressed sparse graph object
    # we only care about organic bonds and not hydrogens
    graph_bonds = numpy.array(bonds>=1.0,dtype=float)
    graph_bonds[hydrogens,:] = 0.0
    graph_bonds[:,hydrogens] = 0.0
    graph = csgraph.csgraph_from_dense(graph_bonds)
    for sg in graph.indices:
        subgraph = graph[sg]
        for i,j in combinations(subgraph.indices,2):
            t0 = csgraph.breadth_first_tree(graph, i_start=i, directed=False)
            t1 = csgraph.breadth_first_tree(graph, i_start=j, directed=False)
            t0i = t0.indices
            t1i = t1.indices
            ring = sorted(set(list(t0i)+list(t1i)+[i,j,sg]))
            # some conditions
            seen        = (ring in rings)
            isring      = (sorted(t0i[1:])==sorted(t1i[1:]))
            bigenough   = (len(ring)>=5)
            smallenough = (len(ring)<=10)
            if isring and not seen and bigenough and smallenough:
                rings.append(ring)
    # we now have a list of all the shortest rings within 
    # the molecular graph. If planar, the ring might be aromatic
    aromatic_epsilon = 0.1
    aromatic = []
    for ring in rings:
        homocycle   = (symbols[ring]=="C").all()
        heterocycle = numpy.in1d(symbols[ring],numpy.array(["C","S","N","O"])).all()
        if (homocycle and (len(ring)%2)==0) or heterocycle:
            ring_positions = positions[ring]
            # small function for coplanarity
            coplanar = all([numpy.linalg.det(numpy.array(x[:3])-x[3])<aromatic_epsilon
                            for x in combinations(ring_positions,4)])
            if coplanar:
                aromatic.append(ring)
    # aromatic bond fixing
    aromatic  = numpy.array(aromatic).ravel()
    ix        = numpy.ix_(aromatic,aromatic)
    bix       = bonds[ix]
    bix[numpy.nonzero(bix)] = 1.5
    bonds[ix] = bix
    # hydrogen bonds
    # TODO
    return bonds

def uff_symbol(atom : ase.Atoms) -> str:
    """Returns the first twol letters of a UFF parameters"""
    sym  = atom.symbol
    if len(sym) == 1 :
        sym =''.join([sym,'_']) 
    return sym

def best_angle(a       : int,
               sbu     : ase.Atoms,
               indices : numpy.ndarray) -> float:
    """Calculates the most common angle around an atom"""
    # linear case
    if len(indices)<=1:
        da = 180.0
    else:
        angles = numpy.array([sbu.get_angle(a1,a,a3,mic=True) 
                              for a1,a3 in combinations(indices,2)]).reshape(-1,1)
        if angles.shape[0]>1:
            # do some clustering on the angles, keep most frequent
            from scipy.cluster.hierarchy import fclusterdata as cluster
            clusters = cluster(angles, 10.0, criterion='distance')
            counts   = numpy.bincount(clusters)
            da = angles[numpy.where(clusters==numpy.argmax(counts))].mean()
        else:
            da = angles[0,0]
    return da

def best_radius(a       : int,
                sbu     : ase.Atoms,
                indices : numpy.ndarray,
                ufflib  : numpy.ndarray) -> float:
    """Return the radius, according to the neighbors of an atom"""
    if len(indices)==0:
        d1 = 0.7
    else:
        # the average of covalent radii will be used for distances
        others  = [uff_symbol(at) for at in sbu[indices] if at.symbol!="X"]
        if len(others)==0:
            d1 = 0.7
        else:
            d1      = numpy.array([numpy.array([v[0] for k,v in ufflib.items() 
                                   if k.startswith(s)]).mean() for s in others]).mean()
    # get the distances also
    d0  = sbu.get_distances(a,indices,mic=True).mean()
    dx  = d0-d1 
    return dx

def best_type(dx     : float,
              da     : float,
              dc     : int  ,
              ufflib : numpy.ndarray,
              types  : list) -> str:
    """Chooses the best UFF type according to neighborhood."""
    mincost = 1000.0
    mintyp  = None 
    for typ in types:
        xx,aa,cc = ufflib[typ]
        cost_x = ((dx-xx)**2)/2.50
        cost_a = ((da-aa)**2)/180.0
        cost_c = ((dc-cc)**2)/4.0
        cost   =  cost_x+cost_a+cost_c
        if cost<mincost:
            mintyp = typ
            mincost= cost
    return mintyp

def analyze_mm(sbu : ase.Atoms) -> (numpy.ndarray,numpy.ndarray):
    """Returns the UFF types and bond matrix for an ASE Atoms."""
    ufflib  = read_uff_library(library="uff4mof")
    bonds   = get_bond_matrix(sbu)
    mmtypes = [None,]*len(sbu)
    for atom in sbu:
        if atom.symbol == "X":
            continue
        # get the starting symbol in uff nomenclature
        symbol = uff_symbol(atom)
        # narrow the choices
        uff_types   = [k for k in ufflib.keys() if k.startswith(symbol)]
        these_bonds = bonds[atom.index].copy()
        # if only one choice, use it
        if len(uff_types)==1:
            mmtypes[atom.index] = uff_types[0]
        # aromatics are easy also
        elif (numpy.abs(these_bonds-1.5)<1e-6).any():
            uff_types = [typ for typ in uff_types if typ.endswith("R")]
            mmtypes[atom.index] = uff_types[0]
        else:
            indices = numpy.where(these_bonds>=0.25)[0]
            # coordination
            dc = len(indices)
            # angle
            da = best_angle(atom.index,sbu,indices)
            # radius
            dx = best_radius(atom.index,sbu,indices,ufflib)
            # complete data
            mmtypes[atom.index] = best_type(dx,da,dc,ufflib,uff_types)
    # now correct the dummies
    for xi in [x.index for x in sbu if x.symbol=="X"]:
        bonded = numpy.argmax(bonds[xi])
        mmtypes[xi] = mmtypes[bonded]
    mmtypes = numpy.array(mmtypes)
    bonds   = numpy.array(bonds)
    return bonds,mmtypes


