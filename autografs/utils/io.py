#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
# Copyright : see accompanying license files for details

__author__  = "Damien Coupry"
__credits__ = ["Prof. Matthew Addicoat"]
__license__ = "MIT"
__maintainer__ = "Damien Coupry"
__version__ = '2.0.3'
__status__  = "beta"

import os
import sys
import numpy
import _pickle as pickle

import ase
from ase              import Atom, Atoms
from ase.spacegroup   import crystal
from ase.data         import chemical_symbols
from ase.neighborlist import NeighborList


from scipy.cluster.hierarchy import fclusterdata as cluster

import warnings

from autografs.utils.pointgroup import PointGroup
from autografs.utils           import __data__

def read_cgd() -> dict:
    """Return a dictionary of topologies as ASE Atoms objects
    
    The format CGD is used mainly by the Systre software
    and by Autografs. All details can be read on the website
    http://rcsr.anu.edu.au/systre
    """
    root = os.path.join(__data__,"topologies")
    topologies = {}
    # we need the names of the groups and their correspondance in ASE spacegroup data
    # this was compiled using Levenshtein distances and regular expressions
    groups_file = os.path.join(root,"HermannMauguin.dat")
    grpf = open(groups_file,"rb")
    groups = {l.split()[0]:l.split()[1] for l in grpf.read().decode("utf8").splitlines()}
    grpf.close()
    # read the rcsr topology data
    topology_file = os.path.join(root,"nets.cgd")
    # the script as such starts here
    with open(topology_file,"rb") as tpf:
        text = tpf.read().decode("utf8")
        # split the file by topology
        topologies_raw = [t.strip().strip("CRYSTAL") for t in text.split("END")]
        topologies_len = len(topologies_raw)
        print("{0:<5} topologies before treatment".format(topologies_len))
        for topology_raw in topologies_raw:
            # read from the template.
            # the edges are easier to comprehend by edge center
            try:
                lines = topology_raw.splitlines()
                lines = [l.split() for l in lines if len(l)>2]
                name = None
                group = None
                cell = []
                symbols = []
                nodes = []
                for l in lines:
                    if l[0].startswith("NAME"):
                        name = l[1].strip()
                    elif l[0].startswith("GROUP"):
                        group = l[1]
                    elif l[0].startswith("CELL"):
                        cell = numpy.array(l[1:], dtype=numpy.float32)
                    elif l[0].startswith("NODE"):
                        this_symbol = chemical_symbols[int(l[2])]
                        this_node = numpy.array(l[3:], dtype=numpy.float32)
                        nodes.append(this_node)
                        symbols.append(this_symbol)
                    elif (l[0].startswith("#") and l[1].startswith("EDGE_CENTER")):
                        # linear connector
                        this_node = numpy.array(l[2:], dtype=numpy.float32)
                        nodes.append(this_node)
                        symbols.append("He")
                    elif l[0].startswith("EDGE"):
                        # now we append some dummies
                        s    = int((len(l)-1)/2)
                        midl = int((len(l)+1)/2)
                        x0  = numpy.array(l[1:midl],dtype=numpy.float32).reshape(-1,1)
                        x1  = numpy.array(l[midl:] ,dtype=numpy.float32).reshape(-1,1)
                        xx  = numpy.concatenate([x0,x1],axis=1).T
                        com = xx.mean(axis=0)
                        xx -= com
                        xx  = xx.dot(numpy.eye(s)*0.5)
                        xx += com
                        nodes   += [xx[0],xx[1]]
                        symbols += ["X","X"]
                nodes = numpy.array(nodes)
                if len(cell)==3:
                    # 2D net, only one angle and two vectors.
                    # need to be completed up to 6 parameters
                    pbc  = [True,True,False] 
                    cell = numpy.array(list(cell[0:2])+[10.0,90.0,90.0]+list(cell[2:]), dtype=numpy.float32)
                    # node coordinates also need to be padded
                    nodes = numpy.pad(nodes, ((0,0),(0,1)), 'constant', constant_values=0.0)
                elif len(cell)<3:
                    continue
                else:
                    pbc = True
                # now some postprocessing for the space groups
                setting = 1
                if ":" in group:
                    # setting might be 2
                    group, setting = group.split(":")
                    try: 
                        setting = int(setting.strip())
                    except ValueError:
                        setting = 1
                # ASE does not have all the spacegroups implemented yet
                if group not in groups.keys():
                    continue
                else:
                    # generate the crystal
                    group     = int(groups[group])
                    topology  = crystal(symbols=symbols,
                                        basis=nodes,
                                        spacegroup=group,
                                        setting=setting,
                                        cellpar=cell,
                                        pbc=pbc,
                                        primitive_cell=False,
                                        onduplicates="keep")
                    topologies[name] = topology
            except Exception as expt:
                continue
    return topologies


def read_sbu(path    : None = None,
             formats : list =["xyz"]) -> dict:
    """Return a dictionary of Atoms objects.

    If the path is not specified, use the default library.
    TODO: Should use a chained iterable of path soon.
    path    -- where to find the sbu
    formats -- what molecular file format to read
    """
    if path is not None:
        path = os.path.abspath(path)
    else:
        path = os.path.join(__data__,"sbu")
    SBUs = {}
    for sbu_file in os.listdir(path):
        ext = sbu_file.split(".")[-1]
        if ext in formats:
            for sbu in ase.io.iread(os.path.join(path,sbu_file)):
                try:
                    name  = sbu.info["name"]
                    SBUs[name] = sbu
                except Exception as e:
                    continue
    return SBUs


def write_gin(path,
              atoms,
              bonds,
              mmtypes):
    """Write an GULP input file to disc"""
    with open(path,"w") as fileobj:
        fileobj.write('opti conp molmec cartesian noautobond\n')
        pbc = atoms.get_pbc()
        if pbc.any():
            cell = atoms.get_cell().tolist()
            if not pbc[2]:
                fileobj.write('{0}\n'.format('svectors'))
                fileobj.write('{0:.3f} {1:.3f} {2:.3f}\n'.format(*cell[0]))
                fileobj.write('{0:.3f} {1:.3f} {2:.3f}\n'.format(*cell[1]))
            else:
                fileobj.write('{0}\n'.format('vectors'))
                fileobj.write('{0:.3f} {1:.3f} {2:.3f}\n'.format(*cell[0]))
                fileobj.write('{0:.3f} {1:.3f} {2:.3f}\n'.format(*cell[1]))
                fileobj.write('{0:.3f} {1:.3f} {2:.3f}\n'.format(*cell[2]))
        fileobj.write('{0}\n'.format('cartesian'))
        symbols = atoms.get_chemical_symbols()
        #We need to map MMtypes to numbers. We'll do it via a dictionary
        symb_types=[]
        mmdic={}
        types_seen=1
        for m,s in zip(mmtypes,symbols):
            if m not in mmdic:
                mmdic[m]="{0}{1}".format(s,types_seen)
                types_seen+=1
                symb_types.append(mmdic[m])
            else:
                symb_types.append(mmdic[m])
        # write it
        for s, (x, y, z), in zip(symb_types, atoms.get_positions()):
            fileobj.write('{0:<4} {1:<7} {2:<15.8f} {3:<15.8f} {4:<15.8f}\n'.format(s,'core',x,y,z))
        fileobj.write('\n')
        bondstring = {4:'quadruple',
                      3:'triple',
                      2:'double',
                      1.5:'resonant',
                      1.0:'',
                      0.5:'half',
                      0.25:'quarter'}
        #write the bonding
        for (i0,i1),b in numpy.ndenumerate(bonds):
            if i0<i1 and b>0.0:
                fileobj.write('{0} {1:<4} {2:<4} {3:<10}\n'.format('connect',i0+1,i1+1,bondstring[b]))
        fileobj.write('\n')
        fileobj.write('{0}\n'.format('species'))
        for  k,v in  mmdic.items():
            fileobj.write('{0:<5} {1:<5}\n'.format(v,k))
        fileobj.write('\n')
        fileobj.write('library uff4mof\n')
        fileobj.write('\n')
        name = ".".join(path.split("/")[-1].split(".")[:-1])
        fileobj.write('output movie xyz {0}.xyz\n'.format(name))
        if pbc.any():
            fileobj.write('output cif {0}.cif\n'.format(name))
        return None