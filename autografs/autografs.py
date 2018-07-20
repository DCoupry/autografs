#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright : see accompanying license files for details

__author__  = "Damien Coupry"
__credits__ = ["Prof. Matthew Addicoat"]
__license__ = "MIT"
__maintainer__ = "Damien Coupry"
__version__ = '2.0.4'
__status__  = "beta"

import os
import sys
import numpy
import ase
import logging
import scipy
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
    Addicoat, M. a, Coupry, D. E., & Heine, T. (2014).
    The Journal of Physical Chemistry. A, 118(40), 9607–14. 
    """

    def  __init__(self):
        """Constructor for the Autografs framework maker.
        """
        logger.info("Reading the topology database.")
        self.topologies = read_topologies_database()
        logger.info("Reading the building units database.")
        self.sbu        = read_sbu_database()


    def make(self,
             topology_name,
             sbu_names = None,
             sbu_dict  = None,
             supercell = (1,1,1)) :
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
        """
        # ase.visualize.view(self.topologies[topology_name])
        logger.info("Starting the MOF generation.")
        # make the supercell prior to alignment
        if isinstance(supercell,int):
            supercell = (supercell,supercell,supercell)
        topology_atoms  = self.topologies[topology_name]
        if supercell!=(1,1,1):
            logger.info("{0}x{1}x{2} supercell of the topology is used.".format(*supercell))
            topology_atoms *= supercell
        # make the Topology object
        logger.info("Analysis of the topology.")
        topology = Topology(name  = topology_name,
                            atoms = topology_atoms)
        # ase.visualize.view(topology.atoms)
        logger.debug("Unique shapes of topology = ")
        logger.debug("{} ".format(topology.get_unique_shapes()))
        # container for the aligned SBUs
        aligned = Framework()
        aligned.set_topology(topology=topology)
        alpha = 0.0
        # identify the corresponding SBU
        logger.info("Scheduling the SBU to slot alignment.")
        try:
            if sbu_dict is None and sbu_names is not None:
                sbu_dict = self.get_sbu_dict(topology=topology,
                                             sbu_names=sbu_names)
            elif sbu_dict is not None:
                # the sbu_dict has been passed. if not SBU object, create them
                for k,v in sbu_dict.items():
                    if not isinstance(v,SBU):
                        assert isinstance(v,ase.Atoms)
                        if "name" in v.info.keys():
                            name = v.info["name"]
                        else:
                            name = str(k)
                        sbu_dict[k] = SBU(name=name,atoms=v)
            else:
                raise RuntimeError("Either supply sbu_names or sbu_dict.")
        except RuntimeError as exc:
            logger.error("Slot to SBU mappping interrupted.")
            logger.error("{exc}".format(exc))
        # some logging
        self.log_sbu_dict(sbu_dict=sbu_dict,topology=topology)
        # carry on
        for idx,sbu in sbu_dict.items():
            logger.debug("Treating slot number {idx}".format(idx=idx))
            logger.debug("\t|-->Aligning SBU {name}".format(name=sbu.name))
            # now align and get the scaling factor
            sbu,f = self.align(fragment=topology.fragments[idx],
                                     sbu=sbu)
            alpha += f
            aligned.append(index=idx,sbu=sbu)
        # refine the cell scaling using a good starting point
        aligned.refine(alpha0=alpha)
        return aligned

    def log_sbu_dict(self,
                     topology,
                     sbu_dict = None):
        """Does some logging on the chosen SBU mapping."""
        for idx,sbu in sbu_dict.items():
            s00 = topology.shapes[idx][0]
            s01 = topology.shapes[idx][1]
            s10 = sbu.shape[0]
            s11 = sbu.shape[1]
            logging.info("Slot {sl}, {s00} {s01}".format(sl=idx,
                                                         s00=s00,
                                                         s01=s01))
            logging.info("\t|-->SBU {sbn} {s10} {s11}.".format(sbn=sbu.name,
                                                                s10=s10,
                                                                s11=s11))
        return None

    def get_topology(self, 
                     topology_name = None):
        """Generates and return a Topology object"""
        topology_atoms = self.topologies[topology_name]
        return Topology(name=topology_name, atoms=topology_atoms)

    def get_sbu_dict(self,
                     topology ,
                     sbu_names):
        """Return a dictionary of SBU by corresponding fragment.

        This stage get a one to one correspondance between
        each topology slot and an available SBU from the list of names.
        TODO: For now, we take the first available if more are given,
        but we should be able to pass this directly to the class,
        or a dictionary of probabilities for the different SBU.
        We also need to implement a check on symmetry operators,
        to catch stuff like 'squares cannot fit in a rectangle slot'.
        topology  -- the Topology object
        sbu_names -- the list of SBU names as strings
        """
        logger.debug("Generating slot to SBU map.")
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
            slots = topology.has_compatible_slots(sbu=sbu)
            if not slots:
                continue
            for slot in slots:
                weights[slot].append(p)
                by_shape[slot].append(sbu)
        # now fill the choices
        sbu_dict = {}
        for index,shape in topology.shapes.items():       
            # here, should accept weights also
            shape = tuple(shape)
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

    def list_available_topologies(self,
                                  sbu_names = [],
                                  full      = True,
                                  max_size  = 100):
        """Return a list of topologies compatible with the SBUs

        For each sbu in the list given in input, refines first by coordination
        then by shapes within the topology. Thus, we do not need to analyze
        every topology.
        sbu  -- list of sbu names
        full -- wether the topology is entirely represented by the sbu
        """
        if sbu_names:
            logger.info("Checking topology compatibility.")
            topologies = []
            sbu = [SBU(name=n,atoms=self.sbu[n]) for n in sbu_names]
            for tk,tv in self.topologies.items():
                if len(tv)>max_size:
                    logger.debug("\tTopology {tk} to big : size = {s}.".format(tk=tk,s=len(tv)))
                    continue
                try:
                    topology = Topology(name=tk,atoms=tv)
                except Exception as exc:
                    logger.debug("Topology {tk} not loaded: {exc}".format(tk=tk,exc=exc))
                    continue
                filled = {shape:False for shape in topology.get_unique_shapes()}
                allsbu = True
                slots_full  = [topology.has_compatible_slots(s) for s in sbu]
                for slots in slots_full:
                    if not slots:
                        allsbu = False
                    else:
                        for slot in slots:
                            filled[slot] = True
                    allsbu = (True&allsbu)
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
                           topology_name = None):
        """Return the dictionary of compatible SBU.
        
        Filters the existing SBU by shape until only
        those compatible with a slot within the topology are left.
        TODO: use the symmetry operators instead of the shape itself.
        topology -- name of the topology in the database
        """
        av_sbu = defaultdict(list)
        if topology_name is not None:
            logger.info("List of compatible SBU with topology {t}:".format(t=topology_name))
            topology = Topology(name=topology_name,
                                atoms=self.topologies[topology_name])
            shapes = topology.get_unique_shapes()
            for sbuk,sbuv in self.sbu.items():
                try:
                    sbu = SBU(name=sbuk,
                              atoms=sbuv)
                except Exception as exc:
                    logger.debug("SBU {k} not loaded: {exc}".format(k=sbuk,exc=exc))
                for shape in shapes:
                    if sbu.is_compatible(shape):
                        logger.info("\t{k}".format(k=sbuk))
                        av_sbu[shape].append(sbuk)
        else:
            logger.info("Listing full database of SBU.")
            av_sbu = list(self.sbu.keys())
        return dict(av_sbu)



if __name__ == "__main__":

    molgen         = Autografs()
    sbu_names      = ["Benzene_linear","Zn_mof5_octahedral"]
    topology_name  = "pcu"
    mof = molgen.make(topology_name=topology_name,sbu_names=sbu_names)
    atoms,_,_ = mof.get_atoms()
    ase.visualize.view(atoms)

