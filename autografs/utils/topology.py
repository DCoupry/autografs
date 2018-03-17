#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
# Copyright : see accompanying license files for details

__author__  = "Damien Coupry"
__credits__ = ["Prof. Matthew Addicoat"]
__license__ = "MIT"
__maintainer__ = "Damien Coupry"
__version__ = '2.0.2'
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
from collections import Counter

from scipy.cluster.hierarchy import fclusterdata as cluster

import warnings

from autografs.utils.pointgroup import PointGroup
from autografs.utils           import __data__

import logging
logger = logging.getLogger(__name__)


warnings.filterwarnings("error")

class Topology(object):
    """Contener class for the topology information"""

    def __init__(self,
                 name  : str,
                 atoms : ase.Atoms) -> None:
        """Constructor for a topology, from an ASE Atoms."""
        logger.debug("Creating Topology {0}".format(name))
        self.name  = name
        self.atoms = atoms 
        # initialize empty fragments
        # shapes and symmops will be used to find
        # corresponding SBUs.
        self.fragments = {}
        self.shapes    = {}
        self.symmops   = {}
        # fill it in
        self._analyze()
        return None

    def get_atoms(self) -> ase.Atoms:
        """Return a copy of the topology as ASE Atoms."""
        logger.debug("Topology {0}: returning atoms.".format(self.name))
        return self.atoms.copy()

    def get_fragments(self) -> ase.Atoms:
        """Return a concatenated version of the fragments."""
        logger.debug("Topology {0}: returning fragments.".format(self.name))
        frags = ase.Atoms(cell=self.atoms.get_cell(),
                          pbc=self.atoms.get_pbc())
        for idx,frag in self.fragments.items():
            tags = numpy.ones(len(frag))*idx
            frag.set_tags(tags)
            frags+=frag
        return frags

    def get_unique_shapes(self) -> set:
        """Return all unique shapes in the topology."""
        logger.debug("Topology {0}: listing unique fragment shapes.".format(self.name))
        return set(self.shapes.values())

    def get_unique_operations(self) -> dict:
        """Return all unique symmetry operations in the topology."""
        logger.debug("Topology {0}: listing symmetry operations.".format(self.name))
        ops = {}
        for idx,shape in self.shapes.items():
            if shape in ops.keys():
                continue
            symmops = self.symmops[idx]
            these_ops = []
            for s,o in symmops.items():
                if o is None:
                    continue
                elif len(o)>0 and s not in ["I","-I"]:
                    sym = ["{0}{1}".format(s,this_o[0]) for this_o in o]
                elif len(o)>0 and o is not None:
                    sym = [s,]
                else:
                    sym = []
                these_ops += sym
            ops[shape] = {o:these_ops.count(o) for o in set(these_ops)}
        return ops

    def has_compatible_slot(self,
                            sbu : object) -> (bool,tuple):
        """Return (True,shape) for a slot compatible with the SBU"""
        compatible = False
        slot      = None
        if sbu.shape in self.get_unique_shapes():
            compatible = True
            slot       = sbu.shape
        else:
            symmops = self.get_unique_operations()
            for shape,ops in symmops.items():
                if shape[1]!=sbu.shape[1]:
                    compatible = False
                    slot      = None
                    break
                elif sbu.is_compatible(ops):
                    compatible = True
                    slot       = shape
                    break
        return compatible,slot

    def _get_cutoffs(self,
                     Xis : numpy.ndarray,
                     Ais : numpy.ndarray) -> numpy.ndarray:
        """Return the cutoffs leading to the desired connectivity"""
        # initialize cutoffs to small non-zero skin partameter
        # TODO check on coordination
        logger.debug("Generating cutoffs")
        skin    = 5e-3
        cutoffs = numpy.zeros(len(self.atoms))+skin
        # we iterate over non-dummies
        for other_index in Ais:
            # cutoff starts impossibly big
            cutoff   = 10000.0
            # we get the distances to all dummies and cluster accordingly
            dists    = self.atoms.get_distances(other_index,Xis,mic=True)
            mindist  = numpy.amin(dists)
            # let's not analyse every single distance...
            dists    = dists[dists<2.0*mindist].reshape(-1,1)
            L   = 0
            eps = mindist*0.5
            coord = self.atoms[other_index].number
            # we nee to coerce the cutoffs to have 
            # the good amount of dummies. 
            # in theory, should loop only once
            while L!=coord:
                clusters = cluster(dists, eps, criterion='distance')
                for cluster_index in set(clusters):
                    # check this cluster distances
                    indices   = numpy.where(clusters==cluster_index)[0]
                    L = len(indices)
                    if L>coord:
                        eps*=0.75
                    elif L<coord:
                        eps*=1.5
                    cutoff_tmp = dists[indices].mean() 
                    if cutoff_tmp<cutoff :
                        # if better score, replace the cutoff
                        cutoff = cutoff_tmp
            cutoffs[other_index] = cutoff
        return cutoffs

    def _analyze(self) -> None:
        """Analyze the topology to cut the fragments out."""
        # separate the dummies from the rest
        logger.debug("Analyzing fragments of topology {0}.".format(self.name))
        numbers = numpy.asarray(self.atoms.get_atomic_numbers())
        Xis  = numpy.where(numbers==0)[0]
        Ais  = numpy.where(numbers >0)[0]
        # setup the tags
        tags = numpy.zeros(len(self.atoms))
        tags[Xis] = Xis + 1
        self.atoms.set_tags(tags)
        tags = self.atoms.get_tags()
        # analyze
        # first build the neighborlist
        cutoffs      = self._get_cutoffs(Xis=Xis,Ais=Ais) 
        neighborlist = NeighborList(cutoffs=cutoffs,
                                    bothways=True,
                                    self_interaction=False,
                                    skin=0.0)
        neighborlist.build(self.atoms)
        # iterate over non-dummies to find dummy neighbors
        for ai in Ais:
            # get indices and offsets of dummies only!
            ni,no = neighborlist.get_neighbors(ai)
            ni,no = zip(*[(idx,off) 
                          for idx,off in list(zip(ni,no)) if idx in Xis])
            ni = numpy.asarray(ni)
            no = numpy.asarray(no)
            # get absolute positions, no offsets
            positions = self.atoms.positions[ni] + no.dot(self.atoms.cell)
            # create the Atoms object
            fragment = Atoms("X"*len(ni),positions,tags=tags[ni]) 
            # calculate the point group properties
            pg       = PointGroup(fragment.copy(),cutoffs.mean())
            # save that info
            self.fragments[ai] = fragment
            self.shapes[ai]    = (pg.schoenflies,self.atoms[ai].number)
            self.symmops[ai]   = pg.symmops
        return None



def download_topologies() -> None:
    """Downloads the topology file from the RCSR website"""
    import requests
    import shutil
    url  ="http://rcsr.anu.edu.au/downloads/RCSRnets.cgd"
    root = os.path.join(__data__,"topologies")
    path = os.path.join(root,"nets.cgd")
    resp = requests.get(url, stream=True)
    if resp.status_code == 200:
        logger.info("Successfully downloaded the nets from RCSR.")
        resp.raw.decode_content = True
        with open(path,"wb") as outpt:
            shutil.copyfileobj(resp.raw, outpt)        
    return

def read_topologies_database(update_db     : bool = False,
                             update_source : bool = False) -> dict:
    """Return a dictionary of topologies as ASE Atoms."""
    from autografs.utils.io import read_cgd
    root     = os.path.join(__data__,"topologies")
    db_file  = os.path.join(root,"topologies.pkl")
    cgd_file = os.path.join(root,"nets.cgd")
    if ((not os.path.isfile(cgd_file)) or (update_source)):
        logger.info("Downloading the topologies from RCSR.")
        download_topologies()
    else:
        logger.info("Using saved nets from RCSR")
    if ((not os.path.isfile(db_file)) or (update_db)):
        logger.info("Reloading the topologies from scratch")
        topologies     = read_cgd()
        topologies_len = len(topologies)
        logger.info("{0:<5} topologies saved".format(topologies_len))
        with open(db_file,"wb") as pkl:
            pickle.dump(obj=topologies,file=pkl)
        return topologies
    else:
        logger.info("Using saved topologies")
        with open(db_file,"rb") as pkl:
            topologies     = pickle.load(file=pkl)
            topologies_len = len(topologies)
            logger.info("{0:<5} topologies loaded".format(topologies_len))
            return topologies



if __name__ == "__main__":

    topologies = read_topologies_database(update_db=True,update_source=True)

