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
import ase
import numpy
import scipy
import logging
import itertools
import scipy.optimize

from collections import defaultdict

from autografs.utils.sbu        import read_sbu_database
from autografs.utils.sbu        import SBU
from autografs.utils.topology   import read_topologies_database
from autografs.utils.topology   import Topology
from autografs.framework        import Framework

logger = logging.getLogger(__name__) 

class Autografs(object):
    """Framework maker class to generate ASE Atoms objects from topologies.

    AuToGraFS: Automatic Topological Generator for Framework Structures.
    Addicoat, M., Coupry, D. E., & Heine, T. (2014).
    The Journal of Physical Chemistry. A, 118(40), 9607–14. 
    """

    def  __init__(self,
                  topology_path = None,
                  sbu_path = None,
                  use_defaults = True,
                  update = False):
        """Constructor for the Autografs framework maker."""
        logger.info("{0:*^80}".format("*"))
        logger.info("* {0:^76} *".format("AuToGraFS"))
        logger.info("* {0:^76} *".format("Automatic Topological Generator for Framework Structures"))
        logger.info("* {0:^76} *".format("Addicoat, M., Coupry, D. E., & Heine, T. (2014)"))
        logger.info("* {0:^76} *".format("The Journal of Physical Chemistry. A, 118(40), 9607-14"))
        logger.info("{0:*^80}".format("*"))
        logger.info("Reading the topology database.")
        self.topologies = read_topologies_database(path=topology_path,
                                                   use_defaults=use_defaults)
        logger.info("Reading the building units database.")
        self.sbu        = read_sbu_database(path=sbu_path,
                                            use_defaults=use_defaults,
                                            update=update)
        # container for current topology
        self.topology = None
        #container for current sbu mapping
        self.sbu_dict    = None
        return None

    def set_topology(self,
                     topology_name,
                     supercell = (1,1,1)):
        """Create and store the topology object"""
        logger.info("{0:*^80}".format(" SETTING NEW TOPOLOGY "))
        logger.info("Topology --> {topo}".format(topo=topology_name.upper()))
        # make the supercell prior to alignment
        if isinstance(supercell,int):
            supercell = (supercell,supercell,supercell)
        topology_atoms  = self.topologies[topology_name]
        if supercell!=(1,1,1):
            logger.info("{0}x{1}x{2} supercell of the topology is used.".format(*supercell))
            topology_atoms *= supercell
        # make the Topology object
        logger.info("Analysis of the topology.")
        topology = Topology(name = topology_name,
                            atoms = topology_atoms)
        logger.debug("Unique shapes of topology = ")
        logger.debug("{} ".format(topology.get_unique_shapes()))
        self.topology = topology
        return None

    def make(self,
             topology_name = None,
             sbu_names = None,
             sbu_dict = None,
             supercell = (1,1,1),
             coercion = False) :
        """Create a framework using given topology and sbu.

        Main funtion of Autografs. The sbu names and topology's
        are to be taken from the compiled databases. The sbu_dict
        can also be passed for multiple components frameworks.
        If the sbu_names is a list of tuples in the shape 
        (name,n), the number n will be used as a drawing probability
        when multiple options are available for the same shape.
        topology_name -- name of the topology to use
        sbu_names     -- list of names of the sbu to use
        sbu_dict -- (optional) one to one sbu to slot correspondance
                    in the shape {index of slot : 'name of sbu'}
        supercell -- (optional) creates a supercell pre-treatment
        coercion -- (optional) force the compatibility to only consider
                    the multiplicity of SBU
        """
        logger.info("{0:*^80}".format(" STARTING THE MOF GENERATION "))
        self.sbudict = None
        # only set the topology if not already done
        if topology_name is not None:
            self.set_topology(topology_name=topology_name,
                              supercell=supercell)
        # container for the aligned SBUs
        aligned = Framework()
        aligned.set_topology(self.topology)
        # identify the corresponding SBU
        try:
            if sbu_dict is None and sbu_names is not None:
                logger.info("Scheduling the SBU to slot alignment.")
                self.sbu_dict = self.get_sbu_dict(sbu_names=sbu_names,
                                                  coercion=coercion)
            elif sbu_dict is not None:
                logger.info("SBU to slot alignment is user defined.")
                # the sbu_dict has been passed. if not SBU object, create them
                for k,v in sbu_dict.items():
                    if not isinstance(v,SBU):
                        if not isinstance(v,ase.Atoms):
                            name = str(v)
                            v = self.sbu[name].copy()
                        elif "name" in v.info.keys():
                            name = v.info["name"]
                        else:
                            name = str(k)
                        sbu_dict[k] = SBU(name=name,
                                          atoms=v)
                self.sbu_dict = sbu_dict
            else:
                raise RuntimeError("Either supply sbu_names or sbu_dict.")
        except RuntimeError as exc:
            logger.error("Slot to SBU mappping interrupted.")
            logger.error("{exc}".format(exc=exc))
            logger.info("No valid framework was generated. Please check your input.")
            logger.info("You can coerce sbu assignment by directly passing a slot to sbu dictionary.")
            return
        # some logging
        self.log_sbu_dict(sbu_dict=self.sbu_dict,
                          topology=self.topology)
        # carry on
        alpha = 0.0
        for idx,sbu in self.sbu_dict.items():
            logger.debug("Treating slot number {idx}".format(idx=idx))
            logger.debug("\t|--> Aligning SBU {name}".format(name=sbu.name))
            # now align and get the scaling factor
            sbu,f = self.align(fragment=self.topology.fragments[idx],
                               sbu=sbu)
            alpha += f
            aligned.append(index=idx,
                           sbu=sbu)
        aligned.refine(alpha0=alpha)
        return aligned

    def log_sbu_dict(self,
                     topology,
                     sbu_dict = None):
        """Does some logging on the chosen SBU mapping."""
        for idx,sbu in sbu_dict.items():
            logging.info("Slot {sl}".format(sl=idx))
            logging.info("\t|--> SBU {sbn}".format(sbn=sbu.name))
        return None

    def get_topology(self, 
                     topology_name = None):
        """Generates and return a Topology object"""
        topology_atoms = self.topologies[topology_name]
        return Topology(name=topology_name, atoms=topology_atoms)

    def get_sbu_dict(self,
                     sbu_names,
                     coercion=False):
        """Return a dictionary of SBU by corresponding fragment.

        This stage get a one to one correspondance between
        each topology slot and an available SBU from the list of names.
        topology  -- the Topology object
        sbu_names -- the list of SBU names as strings
        coercion -- wether to force compatibility by coordination alone
        """
        logger.debug("Generating slot to SBU map.")
        assert self.topology is not None
        weights  = defaultdict(list)
        by_shape = defaultdict(list)
        for name in sbu_names:
            # check if probabilities included
            if isinstance(name,tuple):
                name,p = name
                p    = float(p)
                name = str(name)
            else:
                p = 1.0
            # create the SBU object
            sbu = SBU(name=name,atoms=self.sbu[name])
            slots = self.topology.has_compatible_slots(sbu=sbu,coercion=coercion)
            if not slots:
                logger.debug("SBU {s} has no compatible slot in topology {t}".format(s=name,t=self.topology.name))
                continue
            for slot in slots:
                weights[slot].append(p)
                by_shape[slot].append(sbu)
        # now fill the choices
        sbu_dict = {}
        for index,shape in self.topology.shapes.items():       
            # here, should accept weights also
            shape = tuple(shape)
            if shape not in by_shape.keys():
                logger.info("Unfilled slot at index {idx}".format(idx=index))
            p = weights[shape]
            # no weights means same proba
            p /= numpy.sum(p)
            sbu_chosen = numpy.random.choice(by_shape[shape],
                                             p=p).copy()
            logger.debug("Slot {sl}: {sb} chosen with p={p}.".format(sl=index,
                                                                     sb=sbu_chosen.name,
                                                                     p=p))
            sbu_dict[index] = sbu_chosen
        return sbu_dict

    def align(self,
              fragment,
              sbu     ):
        """Return an aligned SBU.

        The SBU is rotated on top of the fragment
        using the procrustes library within scipy.
        a scaling factor is also calculated for all three
        cell vectors.
        fragment -- the slot in the topology, ASE Atoms
        sbu      -- object to align, ASE Atoms
        """
        # first, we work with copies
        fragment       = fragment.copy()
        # normalize and center
        fragment_cop         = fragment.positions.mean(axis=0)
        fragment.positions  -= fragment_cop
        sbu.atoms.positions -= sbu.atoms.positions.mean(axis=0)
        # identify dummies in sbu
        sbu_Xis = [x.index for x in sbu.atoms if x.symbol=="X"]
        # get the scaling factor
        size_sbu      = numpy.linalg.norm(sbu.atoms[sbu_Xis].positions,axis=1)
        size_fragment = numpy.linalg.norm(fragment.positions,axis=1)
        alpha         = size_sbu.mean()/size_fragment.mean()
        # TODO check initial scaling: it goes up too much with unit cell
        ncop = numpy.linalg.norm(fragment_cop)
        if ncop<1e-6:
            direction  = numpy.ones(3,dtype=numpy.float32)
            direction /= numpy.linalg.norm(direction)
        else:
            direction = fragment_cop / ncop
        # scaling for better alignment
        fragment.positions = fragment.positions.dot(numpy.eye(3)*alpha)
        alpha *= direction/2.0
        # getting the rotation matrix
        X0  = sbu.atoms[sbu_Xis].get_positions()
        X1  = fragment.get_positions()
        if X0.shape[0]>5:
            X0 = self.get_vector_space(X0)
            X1 = self.get_vector_space(X1)
        R,s = scipy.linalg.orthogonal_procrustes(X0,X1)
        sbu.atoms.positions = sbu.atoms.positions.dot(R)+fragment_cop
        fragment.positions += fragment_cop
        res_d = ase.geometry.distance(sbu.atoms[sbu_Xis],fragment)
        logger.debug("Residual distance: {d}".format(d=res_d))
        # tag the atoms
        sbu.transfer_tags(fragment)
        return sbu,alpha

    def get_vector_space(self,
                         X   ):
        """Returns a vector space as four points."""
        # initialize
        x0 = X[0]
        # find the point most orthogonal
        dots = [x.dot(x0)for x in X]
        i1 = numpy.argmin(dots)
        x1 = X[i1]
        # the second point maximizes the same with x1
        dots = [x.dot(x1) for x in X[1:]]
        i2 = numpy.argmin(dots)+1
        x2 = X[i2]
        # we find a third point
        dots = [x.dot(x1)+x.dot(x0)+x.dot(x2) for x in X]
        i3 = numpy.argmin(dots)
        x3 = X[i3]
        return numpy.asarray([x0,x1,x2,x3])

    def list_available_frameworks(self,
                                 topology_name = None,
                                 from_list = [],
                                 coercion = False):
        """Return a list of sbu_dict covering all the database"""
        av_sbu = self.list_available_sbu(topology_name=topology_name,
                                         from_list=from_list,
                                         coercion = coercion)
        dicts = []
        for product in itertools.product(*av_sbu.values()):
            tmp_d = {}
            for k,v in zip(av_sbu.keys(),product):
                tmp_d.update({kk:v for kk in k})
            dicts.append(tmp_d)
        return dicts

    def list_available_topologies(self,
                                  sbu_names = [],
                                  full      = True,
                                  max_size  = 100,
                                  from_list = [],
                                  pbc       = "all",
                                  coercion = False):
        """Return a list of topologies compatible with the SBUs

        For each sbu in the list given in input, refines first by coordination
        then by shapes within the topology. Thus, we do not need to analyze
        every topology.
        sbu  -- list of sbu names
        full -- wether the topology is entirely represented by the sbu
        max_size -- maximum size of in SBU numbers of topologies to consider
        from_list -- only consider topologies from this list
        """
        these_topologies_names = self.topologies.keys()
        if max_size is None:
            max_size = 999999   
        if from_list:
            these_topologies_names = from_list
        if pbc == "2D":
            logger.info("only considering 2D periodic topologies.")
            these_topologies_names = [tk for tk,tv in self.topologies.items() if sum(tv.pbc)==2]
        elif pbc == "3D":
            logger.info("only considering 3D periodic topologies.")
            these_topologies_names = [tk for tk,tv in self.topologies.items() if sum(tv.pbc)==3]
        elif pbc != "all":
            logger.info("pbc keyword has to be '2D','3D' or 'all'. Assumed 'all'.")
        if sbu_names:
            logger.info("Checking topology compatibility.")
            topologies = []
            sbu = [SBU(name=n,atoms=self.sbu[n]) for n in sbu_names]
            for tk in these_topologies_names:
                tv = self.topologies[tk]
                if max_size is None or len(tv)>max_size:
                    logger.debug("\tTopology {tk} to big : size = {s}.".format(tk=tk,s=len(tv)))
                    continue
                try:
                    topology = Topology(name=tk,atoms=tv)
                except Exception as exc:
                    logger.debug("Topology {tk} not loaded: {exc}".format(tk=tk,exc=exc))
                    continue
                filled = {shape:False for shape in topology.get_unique_shapes()}
                slots_full  = [topology.has_compatible_slots(s,coercion=coercion) for s in sbu]
                for slots in slots_full:
                    for slot in slots:
                        filled[slot] = True
                if all(filled.values()):
                    logger.info("\tTopology {tk} fully available.".format(tk=tk))
                    topologies.append(tk)
                elif any(filled.values()) and not full:
                    logger.info("\tTopology {tk} partially available.".format(tk=tk))
                    topologies.append(tk)
                else:
                    logger.debug("\tTopology {tk} not available.".format(tk=tk))
        else:
            logger.info("Listing full database of topologies.")
            topologies = list(self.topologies.keys())
        return topologies

    def list_available_sbu(self,
                           topology_name = None,
                           from_list = [],
                           coercion = False):
        """Return the dictionary of compatible SBU.
        
        Filters the existing SBU by shape until only
        those compatible with a slot within the topology are left.
        topology -- name of the topology in the database
        from_list -- only consider SBU from this list
        """
        av_sbu = defaultdict(list)
        if from_list:
            sbu_names = from_list
        else:
            sbu_names = list( self.sbu.keys())
        if topology_name is not None or self.topology is not None:
            if topology_name is not None:
                topology = Topology(name=topology_name,
                                    atoms=self.topologies[topology_name])
            else:
                topology = self.topology
            logger.info("List of compatible SBU with topology {t}:".format(t=topology.name))
            sbu_list = []
            logger.info("\tShape analysis of {} available SBU...".format(len(self.sbu)))
            for sbuk in sbu_names:
                sbuv = self.sbu[sbuk]
                try:
                    sbu = SBU(name=sbuk,
                              atoms=sbuv)
                except Exception as exc:
                    logger.debug("SBU {k} not loaded: {exc}".format(k=sbuk,exc=exc))
                    continue
                sbu_list.append(sbu)
            for sites in topology.equivalent_sites:
                logger.info("\tSites considered : {}".format(", ".join([str(s) for s in sites])))
                shape = topology.shapes[sites[0]]
                for sbu in sbu_list:
                    is_compatible = sbu.is_compatible(shape, coercion=coercion)
                    if is_compatible:
                        logger.info("\t\t|--> {k}".format(k=sbu.name))
                        av_sbu[tuple(sites)].append(sbu.name)
        else:
            logger.info("Listing full database of SBU.")
            av_sbu = list(self.sbu.keys())    
        return dict(av_sbu)



if __name__ == "__main__":

    molgen         = Autografs()
    sbu_names      = ["Benzene_linear","Zn_mof5_octahedral"]
    topology_name  = "pcu"
    mof = molgen.make(topology_name=topology_name,sbu_names=sbu_names)
    mof.view()

