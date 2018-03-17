#!/usr/bin/env python3.6
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
import logging
import _pickle as pickle

import ase
from ase.spacegroup   import crystal
from ase.visualize    import view
from ase.data         import chemical_symbols
from ase.neighborlist import NeighborList

from collections import Counter

from scipy.cluster.hierarchy import fclusterdata as cluster
import warnings


from autografs.utils.pointgroup import PointGroup
from autografs.utils.mmanalysis import analyze_mm 
from autografs.utils            import __data__

logger = logging.getLogger(__name__) 


class SBU(object):
    """Container class for a building unit information"""

    def __init__(self,
                 name  : str,
                 atoms : ase.Atoms = None) -> None:
        """Constructor for a building unit, from an ASE Atoms."""
        logger.debug("New instance of SBU {0}".format(name))
        self.name  = name
        self.atoms = atoms 
        self.symmops = {}
        self.mmtypes = []
        self.bonds   = []
        self.shape = ("C1",0)
        if self.atoms is not None:
            self._analyze()
        logger.debug("{self}".format(self=self.__str__()))
        return None

    def __repr__(self) -> str:
        """Return representation full printing"""
        strings = []
        strings.append("\nSBU: {name}\n".format(name=self.name))
        # catch for empty object
        if self.atoms is None:
            return "".join(strings)
        strings.append("Point group : {sc}\n".format(sc=self.shape[0]))
        strings.append("Coordination: {co}\n".format(co=self.shape[1]))
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

    def __str__(self) -> str:
        """Uses repr to print the string.TODO=distinct."""
        return self.__repr__()

    def set_atoms(self,
                  atoms   : ase.Atoms,
                  analyze : bool = False) -> None:
        """Set new Atoms object and reanalyze"""
        logger.debug("Resetting Atoms in SBU {0}".format(self.name))
        self.atoms = atoms
        if analyze:
            logger.debug("\tAnalysis required.")
            self._analyze()
        return None

    def copy(self) -> object:
        """Return a copy of the object"""
        logger.debug("SBU {0}: creating copy.".format(self.name))
        new = SBU(name=str(self.name))
        new.set_atoms(atoms=self.get_atoms(),analyze=False)
        new.symmops = self.symmops.copy()
        new.mmtypes = numpy.copy(self.mmtypes)
        new.bonds   = numpy.copy(self.bonds)
        new.shape   = (self.shape[0],self.shape[1])
        return new

    def get_unique_operations(self) -> dict:
        """Return all unique symmetry operations in the topology."""
        logger.debug("SBU {0}: listing symmetry operations.".format(self.name))
        these_ops = []
        for s,o in self.symmops.items():
            if o is None:
                continue
            elif len(o)>0 and s not in ["I","-I"]:
                sym = ["{0}{1}".format(s,this_o[0]) for this_o in o]
            elif len(o)>0 and o is not None:
                sym = [s,]
            else:
                sym = []
            these_ops += sym
        ops = {o:these_ops.count(o) for o in set(these_ops)}
        return ops

    def is_compatible(self,
                      symmops : list ) -> bool:
        """Return True if symmetry compatible with symmops.
        
        For an SBU to be compatible for alignment with a topology slot
        it has to have at least as many symmetry operators as the slot for
        each type of symmetry. We assume the coordination is checked elsewhere.
        symmops -- (op type, count). e.g: ("C2",5) means 5 C2 in the slot.
        """
        logger.debug("SBU {0}: checking compatibility with slot.".format(self.name))
        compatible = True
        sbuops = self.get_unique_operations()
        for s,o in symmops.items():
            if s not in sbuops.keys():
                compatible = False
            elif o>sbuops[s]:
                 compatible = False
            if not compatible:
                break
        logger.debug("\tCompatibility = {cmp}.".format(cmp=compatible))
        return compatible

    def get_atoms(self) -> ase.Atoms:
        """Return a copy of the topology as ASE Atoms."""
        logger.debug("SBU {0}: returning atoms.".format(self.name))
        return self.atoms.copy()

    def _analyze(self) -> None:
        """Guesses the mmtypes, bonds and pointgroup"""
        logger.debug("SBU {0}: analyze bonding and symmetry.".format(self.name))
        dummies = ase.Atoms([x for x in self.atoms if x.symbol=="X"])
        if len(dummies)>0:
            pg = PointGroup(dummies,0.1)
            self.shape    = (pg.schoenflies,len(dummies))
            self.symmops  = pg.symmops
        bonds,mmtypes = analyze_mm(self.get_atoms())
        self.bonds    = bonds
        self.mmtypes  = mmtypes
        return None

    def transfer_tags(self,
                      fragment : ase.Atoms) -> None:
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

def read_sbu_database(update : bool = False,
                      path   : str  = None):
    """Return a dictionnary of ASE Atoms as SBUs"""
    from autografs.utils.io import read_sbu
    db_file = os.path.join(__data__,"sbu/sbu.pkl")
    user_db = (path is not None)
    no_dflt = (not os.path.isfile(db_file))
    if (user_db or update or no_dflt):
        logger.info("Reloading the building units from scratch")
        sbu = read_sbu(path=path)
        sbu_len = len(sbu)
        logger.info("{0:<5} sbu saved".format(sbu_len))
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

