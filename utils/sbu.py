"""
TODO
"""

__all__     = ["read_sbu","read_sbu_database"]
__author__  = "Damien Coupry"
__version__ = 2.0


import os
import sys
import numpy
import _pickle as pickle

from ase              import Atom, Atoms
from ase.spacegroup   import crystal
from ase.visualize    import view
from ase.data         import chemical_symbols
from ase.neighborlist import NeighborList


from scipy.cluster.hierarchy import fclusterdata as cluster
from progress.bar            import Bar
import warnings


from autografs.utils.pointgroup import PointGroup
from autografs.utils            import __data__



def read_sbu(path=None,formats=["xyz"]):
    
    from ase.io import iread

    if path is not None:
        path = os.path.abspath(path)
    else:
        path = os.path.join(__data__,"sbu")

    SBUs = {}
    for sbu_file in os.listdir(path):
        ext = sbu_file.split(".")[-1]
        if ext in formats:
            for sbu in iread(os.path.join(path,sbu_file)):
                try:
                    name  = sbu.info["name"]
                    dummies = Atoms([x for x in sbu if x.symbol=="X"])
                    pg    = PointGroup(dummies,0.3)
                    shape = (pg.schoenflies,len(dummies))
                    SBUs[name] = {"Shape"    : shape,
                                  "SBU"      : sbu,
                                  "Symmetry" : pg.symmops}
                except Exception as e:
                    continue

    return SBUs

def read_sbu_database(update=False,path=None):
    """
    TODO
    """
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

