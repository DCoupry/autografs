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
import _pickle as pickle

import ase
from ase.spacegroup   import crystal
from ase.visualize    import view
from ase.data         import chemical_symbols
from ase.neighborlist import NeighborList

from collections import Counter

from scipy.cluster.hierarchy import fclusterdata as cluster
from progress.bar            import Bar
import warnings


from autografs.utils.pointgroup import PointGroup
from autografs.utils.mmanalysis import analyze_mm 
from autografs.utils            import __data__



class SBU(object):
    """Contener class for a building unit information"""

    def __init__(self,
                 name  : str,
                 atoms : ase.Atoms) -> None:
        """Constructor for a building unit, from an ASE Atoms."""
        self.name  = name
        self.atoms = atoms 
        self._analyze()
        return None

    def is_compatible(self,
                      symmops : list ) -> bool:
        """Return True if symmetry compatible with symmops.
        
        For an SBU to be compatible for alignment with a topology slot
        it has to have at least as many symmetry operators as the slot for
        each type of symmetry. We assme the coordination is checked elsewhere.
        symmops -- (op type, count). e.g: ("C2",5) means 5 C2 in the slot.
        """
        compatible = True
        for optype,count in symmops:
            ops1 = self.symmops[optype]
            if optype=="I":
                i0 = (ops1 is None)
                i1 = (ops0 is None)
                compatible = (i0==i1)
            else:
                # we only care about the order
                o0 = Counter([o[0] for o in ops0])
                o1 = Counter([o[0] for o in ops1])
                compatible = all([o1[o]>=o0[o] for o in o0.keys()])
            if not compatible:
                break
        return compatible

    def get_atoms(self) -> ase.Atoms:
        """Return a copy of the topology as ASE Atoms."""
        return self.atoms.copy()

    def _analyze(self) -> None:
        """Guesses the mmtypes, bonds and pointgroup"""
        dummies = ase.Atoms([x for x in self.atoms if x.symbol=="X"])
        pg = PointGroup(dummies,0.3)
        bonds,mmtypes = analyze_mm(self.get_atoms())
        self.shape    = (pg.schoenflies,len(dummies))
        self.symmops  = pg.symmops
        self.bonds    = bonds
        self.mmtypes  = mmtypes
        return None



def read_sbu_database(update=False,path=None):
    """
    TODO
    """
    from autografs.utils.io import read_sbu
    db_file = os.path.join(__data__,"sbu/sbu.pkl")
    user_db = (path is not None)
    no_dflt = (not os.path.isfile(db_file))
    if (user_db or update or no_dflt):
        print("Reloading the building units from scratch")
        sbu = read_sbu(path=path)
        sbu_len = len(sbu)
        print("{0:<5} sbu saved".format(sbu_len))
        with open(db_file,"wb") as pkl:
            pickle.dump(obj=sbu,file=pkl)
        return sbu
    else:
        print("Using saved sbu")
        with open(db_file,"rb") as pkl:
            sbu = pickle.load(file=pkl)
            sbu_len = len(sbu)
            print("{0:<5} sbu loaded".format(sbu_len))
            return sbu


if __name__ == "__main__":

    sbus = read_sbu_database(update=True)

