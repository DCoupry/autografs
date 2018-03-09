"""
TODO
"""

__all__     = ["Autografs"]
__author__  = "Damien Coupry"
__version__ = 2.0


import os
import sys
import numpy
import scipy
import pickle

import ase

# todo: refine cell + progress of alignment
import scipy.optimize
# from progress.bar            import Bar

from autografs.utils.sbu        import read_sbu_database
from autografs.utils.topologies import read_topologies_database
from autografs.utils.mmanalysis import analyze_mm 
from autografs.framework        import Framework



class Autografs(object):
    """
    TODO
    """

    def  __init__(self):
        """
        TODO
        """
        self.topologies : dict      = read_topologies_database()
        self.sbu        : ase.Atoms = read_sbu_database()

    def make(self,
             topology_name : str,
             sbu_names     : str  = None,
             sbu_dict      : dict = None) -> ase.Atoms :
        """
        TODO
        """
        # container for the aligned SBUs
        aligned  = Framework()
        alpha    = 0.0
        # get the data from dictionaries
        topology_data = self.topologies[topology_name]
        # identify the corresponding SBU
        sbu_dict = self.get_sbu_dict(topology=topology_data,sbu_names=sbu_names)
        # carry on
        fragments = topology_data["Fragments"]
        topology  = topology_data["Topology"]
        aligned.set_topology(topology=topology)
        for fragment_idx,sbu_data in sbu_dict.items():
            fragment = fragments[fragment_idx][2]
            sbu         = sbu_data["SBU"]
            has_mmtypes = ("mmtypes" in sbu.info.keys())
            has_bonds   = ("bonds" in sbu.info.keys())
            if has_bonds and has_mmtypes:
                sbu_types = sbu.info["mmtypes"]
            else:
                sbu_bonds,sbu_types = analyze_mm(sbu)
            # align and get the scaling factor
            sbu,f = self.align(fragment=fragment,
                               sbu=sbu,
                               box=topology.get_cell())
            alpha += f
            aligned.append(sbu=sbu,mmtypes=sbu_types,bonds=sbu_bonds)
        # refine the cell scaling using a good starting point
        aligned.refine(x0=alpha)
        return aligned

    def get_sbu_dict(self,
                     topology  : dict,
                     sbu_names : list) -> dict:
        """
        TODO
        """
        sbu_dict = {}
        shapes = {shape:[] for shape in topology["Shapes"]}
        for idx,(s0,s1,_,_) in topology["Fragments"].items():
            shapes[(s0,s1)].append(idx)
        for shape,indices in shapes.items():        
            sbu_data = [self.sbu[s] for s in sbu_names if self.sbu[s]["Shape"]==shape]
            if len(sbu_data)==len(indices):
                for idx,sbdat in zip(indices,sbu_data):
                    sbu_dict[idx] = sbdat
            else:
                for idx in indices:
                    sbu_dict[idx] = sbu_data[0]
        return sbu_dict

    def align(self,
              fragment : ase.Atoms,
              sbu      : ase.Atoms,
              box      : numpy.ndarray) -> (ase.Atoms, float):
        """
        TODO
        """
        this_sbu            =      sbu.copy()
        this_fragment       = fragment.copy()
        # first we take advantage that all dummies outside the unit 
        # box will bond over pbc to define a scaling ratio.
        this_fragment.cell = box
        this_fragment.pbc  = [1,1,1]
        p       = this_fragment.get_scaled_positions(wrap=False)
        outside = len(this_fragment) - numpy.sum(numpy.any(((p<0)|(p>1)),axis=1),dtype=int)
        x_ratio = outside/float(len(this_fragment))
        # normalize and center
        this_sbu_cop        = this_sbu.positions.mean(axis=0)
        this_fragment_cop   = this_fragment.positions.mean(axis=0)
        this_sbu.positions -= this_sbu_cop
        fragment.positions -= this_fragment_cop
        # identify dummies in sbu
        sbu_dummies_indices = [x.index for x in this_sbu if x.symbol=="X"]
        sbu_dummies         = this_sbu[sbu_dummies_indices]
        # get the scaling factor
        size_sbu      = numpy.linalg.norm(sbu_dummies.positions,axis=1)
        size_fragment = numpy.linalg.norm(fragment.positions,axis=1)
        alpha         = numpy.mean(size_sbu/size_fragment)
        ncop          = numpy.linalg.norm(this_fragment_cop)
        if ncop<1e-6:
            direction  = numpy.ones(3,dtype=numpy.float32)
            direction /= numpy.linalg.norm(direction)
        else:
            direction = this_fragment_cop / ncop
        alpha *= direction
        # scaling for better alignment
        this_fragment.positions = this_fragment.positions.dot(numpy.eye(3)*alpha)
        # getting the rotation matrix
        X0  = sbu_dummies.get_positions()
        X1  = this_fragment.get_positions()
        R,s = scipy.linalg.orthogonal_procrustes(X0,X1)
        this_sbu.positions = this_sbu.positions.dot(R)
        this_sbu.tags[sbu_dummies_indices] = this_fragment.get_tags()
        alpha =  alpha*x_ratio
        return this_sbu,alpha

    # def get_scaling_ratio(self):
# 
        # return alpha
        
    def list_available_topologies(self,
                                  sbu  : list = [],
                                  full : bool = True) -> list:
        """
        TODO
        """
        if sbu:
            topologies = []
            shapes = set([self.sbu[sbuk]["Shape"] for sbuk in sbu])
            for tk,tv in self.topologies.items():
                tshapes = set(tv["Shapes"])
                c0 = (all([s in tshapes for s in  shapes]))
                c1 = (all([s in  shapes for s in tshapes]) and c0)
                if c1 and full:
                    topologies.append(tk)
                elif c0 and not full:
                    topologies.append(tk)
        else:
            topologies = list(self.topologies.keys())
        return topologies


    def list_available_sbu(self,
                           topology : str) -> dict:
        """
        TODO
        """
        sbu = {shape:list(self.sbu.keys()) for shape in shapes}
        if topology:
            shapes = self.topologies[topology]["Shapes"]
            for shape in shapes:
                sbu[shape] = [sbuk for sbuk in sbu[shape] if self.sbu[sbuk]["Shape"]==shape]
        return sbu



if __name__ == "__main__":

    molgen         = Autografs()
    sbu_names      = ["Benzene_linear","Zn_mof5_octahedral"]
    topology_name  = "pcu"
    mof = molgen.make(topology_name=topology_name,sbu_names=sbu_names)
    ase.visualize.view(mof.get_atoms())

