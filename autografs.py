#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
# Copyright : see accompanying license files for details

__author__  = "Damien Coupry"
__credits__ = ["Prof. Matthew Addicoat"]
__license__ = "MIT"
__maintainer__ = "Damien Coupry"
__version__ = 2.0
__status__  = "alpha"

import os
import sys
import numpy
import ase
import scipy
import scipy.optimize
from collections import defaultdict

from autografs.utils.sbu        import read_sbu_database
from autografs.utils.sbu        import SBU
from autografs.utils.topologies import read_topologies_database
from autografs.utils.topologies import Topology
from autografs.utils.mmanalysis import analyze_mm 
from autografs.framework        import Framework



class Autografs(object):
    """Framework maker class to generate ASE Atoms objects from topologies.

    AuToGraFS: Automatic Topological Generator for Framework Structures.
    Addicoat, M. a, Coupry, D. E., & Heine, T. (2014).
    The Journal of Physical Chemistry. A, 118(40), 9607–14. 
    """

    def  __init__(self):
        """Constructor for the Autografs framework maker.
        """
        self.topologies : dict      = read_topologies_database()
        self.sbu        : ase.Atoms = read_sbu_database()

    def make(self,
             topology_name : str,
             sbu_names     : str  = None,
             sbu_dict      : dict = None) -> ase.Atoms :
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
        topology = Topology(name  = topology_name,
                            atoms = self.topologies[topology_name])
        # container for the aligned SBUs
        aligned  = Framework()
        aligned.set_topology(topology=topology.get_atoms())
        alpha    = 0.0
        # identify the corresponding SBU
        if sbu_dict is None:
            sbu_dict = self.get_sbu_dict(topology=topology,
                                         sbu_names=sbu_names)
        else:
            # the sbu_dict has been passed. if not SBU object, create them
            for k,v in sbu_dict.item():
                if not isinstance(v,SBU):
                    assert isinstance(v,ase.Atoms)
                    if "name" in v.info.keys():
                        name = v.info["name"]
                    else:
                        name = str(k)
                    sbu_dict[k] = SBU(name=name,atoms=v)
        # carry on
        for idx,sbu in sbu_dict.items():
            fragment_atoms = topology.fragments[idx]
            sbu_atoms      = sbu.atoms
            # check if has all info
            sbu_info    = list(sbu.atoms.info.keys())
            has_mmtypes = ("mmtypes" in sbu_info)
            has_bonds   = ("bonds"   in sbu_info)
            if has_bonds and has_mmtypes:
                sbu_types = sbu.atoms.info["mmtypes"]
                sbu_bonds = sbu.atoms.info["bonds"]
            else:
                sbu_bonds,sbu_types = analyze_mm(sbu.get_atoms())
            # align and get the scaling factor
            sbu_atoms,f = self.align(fragment=fragment_atoms,
                               sbu=sbu_atoms)
            alpha += f
            sbu.atoms.positions = sbu_atoms.positions
            sbu.atoms.set_tags(sbu_atoms.get_tags())
            aligned.append(index=idx,sbu=sbu,mmtypes=sbu_types,bonds=sbu_bonds)
        # refine the cell scaling using a good starting point
        aligned.refine(alpha0=alpha)
        return aligned

    def get_sbu_dict(self,
                     topology  : dict,
                     sbu_names : list) -> dict:
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
        weights  = defaultdict(list)
        by_shape = defaultdict(list)
        for name in sbu_names:
            # create the SBU object
            sbu = SBU(name=name,atoms=self.sbu[name])
            # check if probability is included
            if isinstance(name,tuple):
                name,p = name
                p    = float(p)
                name = str(name)
                weights[sbu.shape].append(p)
            else:
                weights[sbu.shape].append(1.0)
            by_shape[sbu.shape].append(sbu)
        # now fill the choices
        sbu_dict = {}
        for index,shape in topology.shapes.items():        
            # here, should accept weights also
            sbu_dict[index] = numpy.random.choice(by_shape[shape],
                                                  p=weights[shape]).copy()
        return sbu_dict

    def align(self,
              fragment : ase.Atoms,
              sbu      : ase.Atoms) -> (ase.Atoms, float):
        """Return an aligned SBU.

        The SBU is rotated on top of the fragment
        using the procrustes library within scipy.
        a scaling factor is also calculated for all three
        cell vectors.
        fragment -- the slot in the topology, ASE Atoms
        sbu      -- object to align, ASE Atoms
        """
        # first, we work with copies
        sbu            =      sbu.copy()
        fragment       = fragment.copy()
        # normalize and center
        fragment_cop        = fragment.positions.mean(axis=0)
        fragment.positions -= fragment_cop
        sbu.positions      -= sbu.positions.mean(axis=0)
        # identify dummies in sbu
        sbu_Xis = [x.index for x in sbu if x.symbol=="X"]
        sbu_X   = sbu[sbu_Xis]
        # get the scaling factor
        size_sbu      = numpy.linalg.norm(sbu_X.positions,axis=1)
        size_fragment = numpy.linalg.norm(fragment.positions,axis=1)
        alpha         = 2.0 * numpy.mean(size_sbu/size_fragment)
        ncop          = numpy.linalg.norm(fragment_cop)
        if ncop<1e-6:
            direction  = numpy.ones(3,dtype=numpy.float32)
            direction /= numpy.linalg.norm(direction)
        else:
            direction = fragment_cop / ncop
        alpha *= direction
        # scaling for better alignment
        fragment.positions = fragment.positions.dot(numpy.eye(3)*alpha)
        # getting the rotation matrix
        X0  = sbu_X.get_positions()
        X1  = fragment.get_positions()
        R,s = scipy.linalg.orthogonal_procrustes(X0,X1)
        sbu.positions = sbu.positions.dot(R)
        # tag the atoms
        self.tag(sbu,fragment)
        return sbu,alpha

    def tag(self,
            sbu      : ase.Atoms,
            fragment : ase.Atoms) -> None:
        """Tranfer tags from the fragment to the closest dummies in the sbu"""
        for atom in sbu:
            if atom.symbol!="X":
                continue
            ps = atom.position
            pf = fragment.positions
            d  = numpy.linalg.norm(pf-ps,axis=1)
            fi = numpy.argmin(d)
            atom.tag = fragment[fi].tag
        return None

    def list_available_topologies(self,
                                  sbu_names  : list = [],
                                  full       : bool = True) -> list:
        """Return a list of topologies compatible with the SBUs

        For each sbu in the list given in input, refines first by coordination
        then by shapes within the topology. Thus, we do not need to analyze
        every topology.
        sbu  -- list of sbu names
        full -- wether the topology is entirely represented by the sbu"""
        # if sbu_names:
        #     topologies = []
        #     sbu = [SBU(name=n,atoms=self.topologies[n]) for n in sbu_names]
        #     for tk in self.topologies.keys():

        #         av_sbu = self.list_available_sbu(topology_name=tk)
        #         slot_filled = numpy.zeros(len(av_sbu),dtype=bool)
        #         # for slotk,slotv in av_sbu.items():
        #         #     [s in slotv for s in sbu_names]
        #         # all the SBU are here, and all slots are filled
        #         c0 = all([any([s in av for av in av_sbu.values()]) for s in sbu_names])
        #         # all the sbu are here, and some slots are not filled
        #         c1 = all([any([s in av for av in av_sbu.values()]) for s in sbu_names])

        # else:
        topologies = list(self.topologies.keys())
        return topologies

    def list_available_sbu(self,
                           topology_name : str  = None) -> dict:
        """Return the dictionary of compatible SBU.
        
        Filters the existing SBU by shape until only
        those compatible with a slot within the topology are left.
        TODO: use the symmetry operators instead of the shape itself.
        topology -- name of the topology in the database
        """
        av_sbu = defaultdict(list)
        if topology_name is not None:
            topology = Topology(name=topology_name,
                                atoms=self.topologies[topology_name])
            topops = topology.get_unique_operations()
            shapes = topology.get_unique_shapes()
            # filter according to coordination first
            for sbuk,sbuv in self.sbu.items():
                c = len([x for x in sbuv if x.symbol=="X"])
                for shape in shapes:
                    # first condition is coordination
                    if c==shape[1]:
                        # now check symmops
                        sbu = SBU(name=sbuk,
                                  atoms=sbuv)
                        if sbu.shape == shape:
                            av_sbu[shape].append(sbuk)
                        elif sbu.is_compatible(topops[shape]):
                            av_sbu[shape].append(sbuk)
        else:
            av_sbu = list(self.sbu.keys())
        return dict(av_sbu)



if __name__ == "__main__":

    molgen         = Autografs()
    sbu_names      = ["Benzene_linear","Zn_mof5_octahedral"]
    topology_name  = "pcu"
    mof = molgen.make(topology_name=topology_name,sbu_names=sbu_names)
    ase.visualize.view(mof.get_atoms())

