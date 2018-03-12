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
import scipy
import typing
import ase

from autografs.utils.sbu        import read_sbu_database
from autografs.utils.topologies import read_topologies_database
from autografs.utils.mmanalysis import analyze_mm 



class Framework(object):
    """
    The Framework object contain the results of an Autografs run.

    Designed to store and post-process the aligned fragments and the
    corresponding topology of an Autografs-made structure. 
    Methods available are setters and getters for the topology, individual SBU,
    rescaling of the total structure, and finally the connecting and returning
    of a clean ASE Atoms object.
    The SBUs are assumed to already be aligned and tagged by the 
    Framework generator, and both bonds and mmtypes are assumed to be ordered.
    The bond matrices are also assumed to be block symmetrical.
    """

    def __init__(self,
                 topology : ase.Atoms = None,
                 SBU      : dict      = {},
                 mmtypes  : numpy.ndarray = [],
                 bonds    : numpy.ndarray = []) -> None:
        """Constructor for the Framework class.

        Can be initialized empty and filled consecutively.
        topology -- ASE Atoms object.
        SBU      -- list of ASE Atoms objects.
        mmtypes  -- array of type names. e.g: 'C_R','O_3'...
        bonds    -- block-symmetric matrix of bond orders.
        """
        mmtypes = numpy.asarray(mmtypes)
        bonds   = numpy.asarray(bonds)
        self.topology : ase.Atoms     = topology
        self.SBU      : dict          = SBU
        self.mmtypes  : numpy.ndarray = mmtypes
        self.bonds    : numpy.ndarray = bonds
        return None

    def __contains__(self, 
                     obj : object) -> None:
        """Iterable intrinsic"""
        r = False
        if hasattr(obj, 'atoms'):
            r = any([obj.atoms==sbu.atoms for sbu in self.SBU.values()])
        return r

    def __delitem__(self,
                    key : str) -> None:
        """Indexable intrinsic"""
        del self.SBU[key]
        return None

    def __setitem__(self,
                    key : str,
                    obj : object) -> None:
        """Indexable intrinsic"""
        if hasattr(obj, 'atoms'):
            self.SBU[key] = object
        return None

    def __getitem__(self,
                    key : str) -> None:
        """Indexable intrinsic"""
        return self.SBU[key]

    def __len__(self):
        """Sizeable intrinsic"""
        return len(self.SBU)

    def __iter__(self):
        """Iterable intrinsic"""
        return iter(self.SBU.values())

    def set_topology(self,
                     topology : ase.Atoms) -> None:
        """Set the topology attribute with an ASE Atoms object."""
        self.topology = topology.copy()
        return None

    def get_topology(self) -> ase.Atoms:
        """Return a copy of the topology attribute as an ASE Atoms object."""
        return self.topology.copy()

    def append(self,
               index   : int,
               sbu     : ase.Atoms,
               bonds   : numpy.ndarray,
               mmtypes : numpy.ndarray) -> None:
        """Append all data releted to a building unit in the framework.

        This includes the ASE Atoms object, the bonding matrix, and the 
        molecular mechanics atomic types as numpy arrays. These three objects 
        are related through indexing: sbu[i] has a MM type mmtypes[i] and 
        a bonding array of bonds[i,:] or bonds[:,i]
        sbu     -- the Atoms object
        bonds   -- the bonds numpy array, of size len(sbu) by len(sbu).
        mmtypes -- the MM atomic types.
        """
        # first append the atoms object to the list of sbu
        self.SBU[index] = sbu
        # make the bonds matrix with a new block
        self.bonds   = scipy.linalg.block_diag(self.bonds,bonds)
        # append the atom types
        self.mmtypes = numpy.hstack([self.mmtypes,mmtypes])
        return None

    def scale(self,
              alpha  : float = 1.0) -> None:
        """Scale the building units positions by a factor alpha.

        This uses the correspondance between the atoms in the topology
        and the building units in the SBU list. Indeed, SBU[i] is centered on 
        topology[i]. By scaling the topology, we obtain a new center for the 
        sbu.
        alpha -- scaling factor
        """
        # get the scaled cell, normalized
        I    = numpy.eye(3)*alpha
        cell = self.topology.get_cell()
        cell = cell.dot(I/numpy.linalg.norm(cell,axis=0))
        self.topology.set_cell(cell,scale_atoms=True)
        # then center the SBUs on this position
        for i,sbu in self.SBU.items():
            center = self.topology[i]
            cop    = sbu.positions.mean(axis=0)
            sbu.positions += center.position - cop
        return None

    def refine(self,
               alpha0 : numpy.ndarray = [1.0,1.0,1.0]) -> None:
        """Refine cell scaling to minimize distances between dummies.

        We already have tagged the corresponding dummies during alignment,
        so we just need to calculate the MSE of the distances between 
        identical tags in the complete structure
        alpha0 -- starting point of the scaling search algorithm
        """
        import copy
        def MSE(x : numpy.ndarray) -> float:
            """Return cost of scaling as MSE of distances."""
            # scale with this parameter
            x = x*alpha0
            old_sbu  = copy.deepcopy(self.SBU)
            old_topo = self.topology.copy() 
            self.scale(alpha=x)
            atoms        = self.get_atoms(dummies=True)
            tags         = atoms.get_tags()
            # reinitialize stuff
            self.SBU      = old_sbu
            self.topology = old_topo
            # find the pairs...
            pairs = [numpy.argwhere(tags==tag) for tag in set(tags) if tag>0]
            pairs =  numpy.asarray(pairs).reshape(-1,2)
            # ...and the distances
            d = [atoms.get_distance(i0,i1,mic=True) for i0,i1 in pairs]
            d = numpy.asarray(d)
            mse = numpy.mean(d**2)
            return mse
        # first get an idea of the bounds.
        # minimum cell of a mof should be over 2.0 Ang.
        low  = 0.5
        high = 2.0
        if numpy.amin(low*alpha0)<2.0:
            low   = 2.0/numpy.amin(alpha0)
            high *= 0.5/low
        # optimize
        result = scipy.optimize.minimize_scalar(fun    = MSE,
                                                bounds = (low,high),
                                                method = "Bounded",
                                                options= {"disp"  : True,
                                                          "xatol" : 1e-02})
        # scale with result
        alpha = result.x*alpha0
        self.scale(alpha=alpha)
        return None

    def rotate(self,
               index : int,
               angle : float) -> None:
        """Rotate the SBU at index around a Cinf symmetry axis"""
        sbu = self.SBU[index]
        if sbu.shape[1]==2:
            axis = [x.position for x in sbu.atoms if x.symbol=="X"]
            axis = numpy.asarray(axis)
            self.SBU[index].atoms.rotate(v=axis,a=angle)
        return None

    def flip(self,
             index : int) -> None:
        """Flip the SBU at index around a C* symmetry axis or Sigmav plane"""
        sbu    = self.SBU[index]
        if sbu.shape[1]==2:
            axis = [x.position for x in sbu.atoms if x.symbol=="X"]
            axis = numpy.asarray(axis)
            self.SBU[index].atoms.rotate(v=axis,a=180.0)
        else:
            sigmav = [op[2] for op in sbu.symmops["sigma"] if op[0]=="v"]
            if sigmav:
                pos = sbu.atoms.positions.dot(sigmav[0])
                self.SBU[index].atoms.set_positions(pos)
        return None

    def functionalize(self):
        return None

    def get_atoms(self,
                  dummies : bool = False) -> ase.Atoms:
        """Return the concatenated Atoms objects.

        The concatenation can either remove the dummies and
        connect the corresponding atoms or leave hem in place
        clean -- remove the dummies if True
        """
        # concatenate every sbu into one Atoms object
        cell = self.topology.get_cell()
        pbc  = self.topology.get_pbc()
        structure = ase.Atoms(cell=cell,pbc=pbc)
        bonds     = self.bonds.copy()
        mmtypes   = self.mmtypes.copy()  
        for sbu in self.SBU.values():
            structure += sbu
        symbols = numpy.asarray(structure.get_chemical_symbols())
        if not dummies:
            # keep track of dummies
            xis   = [x.index for x in structure if x.symbol=="X"]
            tags  = structure.get_tags()
            pairs = [numpy.argwhere(tags==tag) for tag in set(tags) if tag>0]
            for pair in pairs:
                # if lone dummy, cap with hydrogen
                if len(pair)==1:
                    xi0 = pair[0]
                    xis.remove(xi0)
                    symbols[xi0] = "H"
                    mmtypes[xi0] = "H_" 
                else:
                    xi0,xi1 = pair
                    bonds0  = numpy.where(bonds[xi0,:]>0.0)[0]
                    bonds1  = numpy.where(bonds[xi1,:]>0.0)[0]
                    if len(bonds0)==0:
                        # dangling bit, mayhaps from defect
                        xis.remove(xi0)
                        xis.remove(xi1)
                        if len(bonds1)==0:
                            symbols[xi1] = "H"
                            mmtypes[xi0] = "H_" 
                    else:
                        # the bond order will be the maximum one
                        bo      = max(numpy.amax(bonds[xi0,:]),
                                      numpy.amax(bonds[xi1,:]))
                        # change the bonds
                        ix        = numpy.ix_(bonds0,bonds1)
                        bonds[ix] = bo
                        ix        = numpy.ix_(bonds1,bonds0)
                        bonds[ix] = bo
            # book keeping on what has disappeared
            structure.set_chemical_symbols(symbols)
            bonds   = numpy.delete(bonds,xis,axis=0)
            bonds   = numpy.delete(bonds,xis,axis=1)
            mmtypes = numpy.delete(mmtypes,xis)
            del structure[xis]
        return structure
