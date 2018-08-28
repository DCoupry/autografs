#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright : see accompanying license files for details

__author__  = "Damien Coupry"
__credits__ = ["Prof. Matthew Addicoat"]
__license__ = "MIT"
__maintainer__ = "Damien Coupry"
__version__ = '2.3.0'
__status__  = "production"


import os
import sys
import numpy
import _pickle as pickle

import ase
from ase              import Atom, Atoms
from ase.spacegroup   import crystal, Spacegroup
from ase.data         import chemical_symbols
from ase.neighborlist import NeighborList
from collections import Counter

from scipy.cluster.hierarchy import fclusterdata as cluster

import warnings

from autografs.utils import symmetry
from autografs.utils import __data__


import logging
logger = logging.getLogger(__name__)


warnings.filterwarnings("error")

class Topology(object):
    """Contener class for the topology information"""

    def __init__(self,
                 name ,
                 atoms,
                 analyze=True):
        """Constructor for a topology, from an ASE Atoms."""
        logger.debug("Creating Topology {0}".format(name))
        self.name  = name
        self.atoms = atoms 
        # initialize empty fragments
        # shapes and symmops will be used to find
        # corresponding SBUs.
        self.fragments = {}
        self.shapes    = {}
        self.pointgroups = {}
        self.equivalent_sites = []
        # fill it in
        if analyze:
            self._analyze()
        return None

    def copy(self):
        """Return a copy of itself as a new instance"""
        new = self.__class__(name  = str(self.name),
                             atoms = self.atoms.copy(),
                             analyze=False)
        new.fragments = self.fragments.copy()
        new.shapes    = self.shapes.copy()
        return new

    def get_atoms(self):
        """Return a copy of the topology as ASE Atoms."""
        logger.debug("Topology {0}: returning atoms.".format(self.name))
        return self.atoms.copy()

    def get_fragments(self):
        """Return a concatenated version of the fragments."""
        logger.debug("Topology {0}: returning fragments.".format(self.name))
        frags = ase.Atoms(cell=self.atoms.get_cell(),
                          pbc=self.atoms.get_pbc())
        for idx,frag in self.fragments.items():
            tags = numpy.ones(len(frag))*idx
            frag.set_tags(tags)
            frags+=frag
        return frags

    def get_unique_shapes(self):
        """Return all unique shapes in the topology."""
        logger.debug("Topology {0}: listing unique fragment shapes.".format(self.name)) 
        return set([tuple(shape) for shape in self.shapes.values()])

    def get_unique_pointgroups(self):
        """Return all unique shapes in the topology."""
        logger.debug("Topology {0}: listing unique fragment point groups.".format(self.name)) 
        return set(self.pointgroups.values())

    def has_compatible_slots(self,
                             sbu,
                             coercion=False):
        """Return [shapes...] for the slots compatible with the SBU"""
        slots = []
        complist = [(ai, self.shapes[ai],self.pointgroups[ai]) 
                    for ai in self.fragments.keys()]
        seen_idx = []
        for idx, shape, pg in complist:
            if idx in seen_idx:
                continue
            eq_sites = [s for s in self.equivalent_sites if idx in s][0]
            seen_idx += eq_sites
            # test for compatible multiplicity  
            mult = (sbu.shape[-1] ==  shape[-1])
            if not mult:
                continue
            # pointgroups are more powerful identifiers
            if pg==sbu.pg:
                slots+=[tuple(c[1]) for c in complist if c[0] in eq_sites]
                continue
            # the sbu has at least as many symmetry axes
            symm = (sbu.shape[:-1]-shape[:-1]>=0).all()
            if symm:
                slots+=[tuple(c[1]) for c in complist if c[0] in eq_sites]
                continue
            if coercion:
                # takes objects of corresponding
                # multiplicity as compatible.
                slots+=[tuple(c[1]) for c in complist if c[0] in eq_sites]
                continue
        return slots

    def _get_cutoffs(self,
                     Xis ,
                     Ais ):
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
            breaker = 0
            while L!=coord:
                breaker+=1
                if breaker > 5:
                    logger.debug("No correct coordination was found. exiting...")
                    raise RuntimeError("Infinite loop in cutoff determination.")
                clusters = cluster(dists, eps, criterion='distance')
                for cluster_index in set(clusters):
                    # check this cluster distances
                    indices   = numpy.where(clusters==cluster_index)[0]
                    cutoff_tmp = dists[indices].mean() 
                    if cutoff_tmp<cutoff :
                        # if better score, replace the cutoff
                        cutoff = cutoff_tmp
                        L = len(indices)
                        if L!=coord:
                            diff = coord/(L-coord)
                            eps*=diff
                            logger.debug("Coordination with cutoff {c}={l}, goal is {r}".format(c=cutoff,l=L,r=coord))
                            logger.debug("\tEpsilon => {eps}".format(eps=eps))
            cutoffs[other_index] = cutoff
        return cutoffs

    def _analyze(self):
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
        neighborlist.update(self.atoms)
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
            max_order = min(8,len(ni))
            shape = symmetry.get_symmetry_elements(mol=fragment.copy(),
                                                   max_order=max_order)
            pg = symmetry.PointGroup(mol=fragment.copy(),
                                       tol=0.1)
            # save that info
            self.fragments[ai] = fragment
            self.shapes[ai] = shape
            self.pointgroups[ai] = pg.schoenflies
        # now getting the equivalent sites using the Spacegroup object
        sg = self.atoms.info["spacegroup"]
        if not isinstance(sg,Spacegroup):
            sg = Spacegroup(sg)
        scaled_positions = self.atoms.get_scaled_positions()
        seen_indices = []
        symbols = numpy.array(self.atoms.get_chemical_symbols())
        for ai in Ais:
            if ai in seen_indices:
                continue
            sites, _ = sg.equivalent_sites(scaled_positions[ai])
            these_indices = []
            for site in sites:
                norms = numpy.linalg.norm(scaled_positions-site,axis=1)
                if norms.min()<1e-6:
                    these_indices.append(norms.argmin())
                # take pbc into account
                norms = numpy.abs(norms-1.0)
                if norms.min()<1e-6:
                    these_indices.append(norms.argmin())
            these_indices = [idx for idx in these_indices if idx in Ais]
            seen_indices += these_indices
            self.equivalent_sites.append(these_indices)
        logger.info("{es} equivalent sites kinds.".format(es=len(self.equivalent_sites)))
        return None

    def view(self):
        """Viewer for the toology"""
        ase.visualize.view(self.atoms)
        return None



def download_topologies():
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

def read_topologies_database(update = False,
                             path = None,
                             use_defaults = True):
    """Return a dictionary of topologies as ASE Atoms."""
    from autografs.utils.io import read_cgd
    root     = os.path.join(__data__,"topologies")
    db_file  = os.path.join(root,"topologies.pkl")
    cgd_file = os.path.join(root,"nets.cgd")
    topologies = {}
    if ((not os.path.isfile(db_file)) or (update)):
        if (not os.path.isfile(cgd_file)) and use_defaults:
            logger.info("Downloading the topologies from RCSR.")
            download_topologies()
        if use_defaults:
            logger.info("Loading the topologies from RCSR default library")
            topologies_tmp = read_cgd(path=None)
            topologies.update(topologies_tmp)
        if path is not None:
            logger.info("Loading the topologies from {0}".format(path))
            topologies_tmp = read_cgd(path=path)
            topologies.update(topologies_tmp)
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
    # update everything
    topologies = read_topologies_database(update = True, 
                                          path = None, 
                                          use_defaults = True)

