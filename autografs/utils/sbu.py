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
import logging
import _pickle as pickle

import ase
from ase.spacegroup   import crystal
# from ase.visualize    import view
from ase.data         import chemical_symbols
from ase.neighborlist import NeighborList

from collections import Counter

from scipy.cluster.hierarchy import fclusterdata as cluster
import warnings


from autografs.utils            import symmetry
from autografs.utils.mmanalysis import analyze_mm 

from autografs.utils            import __data__

logger = logging.getLogger(__name__) 


class SBU(object):
    """Container class for a building unit information"""

    def __init__(self,
                 name,
                 atoms=None):
        """Constructor for a building unit, from an ASE Atoms."""
        logger.debug("New instance of SBU {0}".format(name))
        self.name = name
        self.atoms = atoms 
        self.mmtypes = []
        self.bonds = []
        self.shape = []
        self.pg = None
        if self.atoms is not None:
            self._analyze()
        logger.debug("{self}".format(self=self.__str__()))
        return None

    def __repr__(self):
        """Uses repr to print the string.TODO=distinct."""
        return self.name

    def __str__(self):
        """Return representation full printing"""
        strings = []
        strings.append("\nSBU: {name}\n".format(name=self.name))
        # catch for empty object
        if self.atoms is None:
            return "".join(strings)
        strings.append("Coordination: {co}\n".format(co=self.shape[-1]))
        if self.pg is not None:
            strings.append("Detected Point Group: {pg}\n".format(pg=self.pg))
        for atom in self.atoms:
            p0,p1,p2 = atom.position
            sy = atom.symbol
            tp = self.mmtypes[atom.index]
            s = "{sy:<3} {p0:>5.2f} {p1:>5.2f} {p2:>5.2f} {tp:<5}\n".format(sy=sy,
                                                                            p0=p0,
                                                                            p1=p1,
                                                                            p2=p2,
                                                                            tp=tp)
            strings.append(s)
        # bonding matrix
        strings.append("Bonding:\n")
        for (i0,i1),b in numpy.ndenumerate(self.bonds):
            if i0<i1 and b>0:
                s = "{i0:<3} connected to {i1:<3}, bo = {bo:1.2f}\n".format(i0=i0,
                                                                            i1=i1,
                                                                            bo=b)
                strings.append(s)
        return "".join(strings)

    def set_atoms(self,
                  atoms,
                  analyze = False):
        """Set new Atoms object and reanalyze"""
        logger.debug("Resetting Atoms in SBU {0}".format(self.name))
        self.atoms = atoms
        if analyze:
            logger.debug("\tAnalysis required.")
            self._analyze()
        return None

    def copy(self):
        """Return a copy of the object"""
        logger.debug("SBU {0}: creating copy.".format(self.name))
        new = SBU(name=str(self.name),atoms=None)
        new.set_atoms(atoms=self.get_atoms(),analyze=False)
        new.mmtypes = numpy.copy(self.mmtypes)
        new.bonds   = numpy.copy(self.bonds)
        new.shape   = list(self.shape)
        return new

    def is_compatible(self,
                      shape,
                      point_group = None,
                      coercion = False):
        """Return True if symmetry compatible with symmops.
        
        For an SBU to be compatible for alignment with a topology slot
        it has to have at least as many symmetry operators as the slot for
        each type of symmetry. We assume the coordination is checked elsewhere.
        shape -- array of counts for symmetry axes. 
                 last element is the multiplicity
        """
        logger.debug("SBU {0}: checking compatibility with slot.".format(self.name))
        compatible = False
        # test for compatible multiplicity  
        mult = (self.shape[-1] ==  shape[-1])
        if mult:
            # use point group as first test
            if point_group is not None:
                compatible = (point_group==self.pg)
            # the sbu has at least as many symmetry axes
            symm = (self.shape[:-1]-shape[:-1]>=0).all()
            if symm:
                compatible = True
            if coercion:
                compatible = True
        logger.debug("\tCompatibility = {cmp}.".format(cmp=compatible))
        return compatible

    def get_atoms(self):
        """Return a copy of the topology as ASE Atoms."""
        logger.debug("SBU {0}: returning atoms.".format(self.name))
        return self.atoms.copy()

    def _analyze(self):
        """Guesses the mmtypes, bonds and pointgroup"""
        logger.debug("SBU {0}: analyze bonding and symmetry.".format(self.name))
        dummies = ase.Atoms([x for x in self.atoms if x.symbol=="X"])
        if len(dummies)>0:
            pg = symmetry.PointGroup(mol=dummies.copy(),tol=0.1)
            max_order = min(8,len(dummies))
            shape = symmetry.get_symmetry_elements(mol=dummies.copy(),
                                                   max_order=max_order)
            self.shape = shape
            self.pg = pg.schoenflies
        bonds,mmtypes = analyze_mm(self.get_atoms())
        self.bonds    = bonds
        self.mmtypes  = mmtypes
        return None

    def transfer_tags(self,
                      fragment):
        """Transfer tags between an aligned fragment and the SBU"""
        logger.debug("\tTagging dummies in SBU {n}.".format(n=self.name))
        # we keep a record of used tags.
        unused = [x.index for x in self.atoms if x.symbol=="X"]
        for atom in fragment:
            ids = [s.index for s in self.atoms if s.index in unused]
            pf = atom.position
            ps = self.atoms.positions[unused]
            d  = numpy.linalg.norm(ps-pf,axis=1)
            si = ids[numpy.argmin(d)]
            self.atoms[si].tag = atom.tag
            unused.remove(si)
        return None

def read_sbu_database(update = False,
                      path   = None,
                      use_defaults = True):
    """Return a dictionnary of ASE Atoms as SBUs"""
    from autografs.utils.io import read_sbu
    db_file = os.path.join(__data__,"sbu/sbu.pkl")
    user_db = (path is not None)
    no_dflt = (not os.path.isfile(db_file))
    if (user_db or update or no_dflt):
        sbu = {}
        if use_defaults:
            logger.info("Loading the building units from default library")
            sbu_tmp = read_sbu(path=None)
            sbu.update(sbu_tmp)
        if path is not None:
            logger.info("Loading the building units from {0}".format(path))
            sbu_tmp = read_sbu(path=path)
            sbu.update(sbu_tmp)
        sbu_len = len(sbu)
        logger.info("{0:<5} sbu loaded.".format(sbu_len))
        with open(db_file,"wb") as pkl:
            pickle.dump(obj=sbu,file=pkl)
        return sbu
    else:
        logger.info("Using saved sbu")
        with open(db_file,"rb") as pkl:
            sbu = pickle.load(file=pkl)
            sbu_len = len(sbu)
            logger.info("{0:<5} sbu loaded".format(sbu_len))
            return sbu


if __name__ == "__main__":

    sbus = read_sbu_database(update=True)

