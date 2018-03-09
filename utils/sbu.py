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


from scipy.cluster.hierarchy import fclusterdata as cluster
from progress.bar            import Bar
import warnings


from autografs.utils.pointgroup import PointGroup
from autografs.utils            import __data__



class SBU(object):
    """Contener class for a building unit information"""

    def __init__(self,
                 name  : str,
                 atoms : ase.Atoms) -> None:
        """Constructor for a building unit, from an ASE Atoms."""
        self.name  = name
        self.atoms = atoms 
        # initialize empty shape
        # shapes and symmops will be used to find
        # corresponding SBUs.
        dummies = ase.Atoms([x for x in self.atoms if x.symbol=="X"])
        pg = PointGroup(dummies,0.3)
        self.shape    = (pg.schoenflies,len(dummies))
        self.symmops  = pg.symmops
        return None

    def get_atoms(self) -> ase.Atoms:
        """Return a copy of the topology as ASE Atoms."""
        return self.atoms.copy()



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

