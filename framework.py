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

from collections import defaultdict

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
        self.topology = topology
        self.SBU      = SBU
        self.mmtypes  = numpy.asarray(mmtypes)
        self.bonds    = numpy.asarray(bonds)
        # keep a dict of elements to delete in 
        # each SBU at connection time. Necessary for example
        # during iterative functionalization.
        self._todel = defaultdict(list)
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
        return iter(self.SBU.items())

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

    def get_bonds(self) -> numpy.ndarray:
        """Return and update the current bond matrix"""
        bonds = []
        for _,sbu in self:
            bonds = scipy.linalg.block_diag(bonds,sbu.bonds)
        self.bonds = bonds
        return numpy.copy(self.bonds)

    def get_mmtypes(self) -> numpy.ndarray:
        """Return and update the current bond matrix"""
        mmtypes = []
        for _,sbu in self:
            mmtypes = numpy.hstack([mmtypes,sbu.mmtypes])
        self.mmtypes = mmtypes
        return numpy.copy(self.mmtypes)

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
        for i,sbu in self:
            center = self.topology[i]
            cop    = sbu.atoms.positions.mean(axis=0)
            sbu.atoms.positions += center.position - cop
        return None

    def refine(self,
               alpha0 : numpy.ndarray = [1.0,1.0,1.0]) -> None:
        """Refine cell scaling to minimize distances between dummies.

        We already have tagged the corresponding dummies during alignment,
        so we just need to calculate the MSE of the distances between 
        identical tags in the complete structure
        alpha0 -- starting point of the scaling search algorithm
        """
        def MSE(x : numpy.ndarray) -> float:
            """Return cost of scaling as MSE of distances."""
            # scale with this parameter
            x = x*alpha0
            self.scale(alpha=x)
            atoms,_,_    = self.get_atoms(dummies=True)
            tags         = atoms.get_tags()
            # reinitialize stuff
            self.scale(alpha=1.0/x)
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
        if self[index].shape[1]==2:
            axis = [x.position for x in self[index].atoms 
                               if x.symbol=="X"]
            axis = numpy.asarray(axis)
            axis = axis[0]-axis[1]
            self[index].atoms.rotate(v=axis,a=angle)
        return None

    def flip(self,
             index : int) -> None:
        """Flip the SBU at index around a C* symmetry axis or Sigmav plane"""
        if self[index].shape[1]==2:
            axis = [x.position for x in self[index].atoms 
                               if x.symbol=="X"]
            axis = numpy.asarray(axis)
            axis = axis[0]-axis[1]
            self[index].atoms.rotate(v=axis,a=180.0)
        else:
            sigmav = [op[2] for op in self[index].symmops["sigma"]
                            if op[0]=="v"]
            if sigmav:
                cop = sbu.atoms.positions.mean(axis=0)
                pos = sbu.atoms.positions - cop
                pos = pos.dot(sigmav[0])  + cop
                self[index].atoms.set_positions(pos)
        return None

    def list_functionalizable_sites(self,symbol=None) -> list:
        """Return a list of tuple for functionalizable sites"""
        sites = []
        for idx,sbu in self:
            bonds = sbu.bonds
            for atom in sbu.atoms:
                if symbol is not None and atom.symbol!=symbol:
                    continue
                bidx = numpy.where(bonds[atom.index,:]>0.0)[0]
                if len(bidx)!=1:
                    continue
                elif bonds[atom.index,bidx[0]]!=1.0:
                    continue
                else:
                    sites.append((idx,atom.index))
        return sites

    def functionalize(self,
                      where : tuple,
                      fg    : ase.Atoms) -> None:
        """Modify a valid slot atom to become a functional group.
        
        The specified slot index and atom index within the slot
        are examined: if the corresponding atom is a single
        atom connected by a bond order of 1 to exactly one other
        atom, it can be functionalized. The given functional group
        has to have exactly one dummy atom. For best performance,
        set the x-func distance at the C-H bond distance.
        where -- (slot index, atom index)
        fg    -- ASE Atoms to replace the atom in where by.
        TODO : move the SBU import to toplevel
        """
        from autografs.utils.sbu import SBU
        sidx, aidx = where
        # create the SBU object for the functional group
        fg_name = "func:{0}".format(fg.get_chemical_formula(mode="hill"))
        fg = SBU(name=fg_name,atoms=fg)
        # center the positions
        fg_cop = fg.atoms.positions.mean(axis=0)
        fg.atoms.positions -= fg_cop
        # check that only one dummy exists
        xidx = [x.index for x in fg.atoms if x.symbol=="X"]
        assert len(xidx)==1
        xidx  = xidx[0]
        # center the sbu
        sbu   = self.SBU[sidx].atoms
        sbu_name = self.SBU[sidx].name
        sbu_cop = sbu.positions.mean(axis=0)
        sbu.positions -= sbu_cop
        # find the bonds
        bonds = self.SBU[sidx].bonds
        bidx  = numpy.where(bonds[aidx]>0.0)[0]
        # check that only one bond exists
        assert len(bidx)==1 and bonds[aidx,bidx[0]]==1.0
        # now get vectors to align
        fgbidx = numpy.where(fg.bonds[xidx]>0.0)[0]
        assert len(fgbidx)==1 and fg.bonds[xidx,fgbidx]==1.0
        # func-x
        v0 = fg.atoms.positions[fgbidx]-fg.atoms.positions[xidx]
        # where[1]-bonded atom
        v1 = sbu.positions[aidx]-sbu.positions[bidx]
        R,s = scipy.linalg.orthogonal_procrustes(v0,v1)
        fg.atoms.positions = fg.atoms.positions.dot(R)
        fg.atoms.positions -= fg.atoms.positions[xidx]
        fg.atoms.positions += sbu.positions[bidx]
        # create the new object
        # keep note of what to delete.
        self._todel[sidx].append(aidx)
        del fg.atoms[xidx]
        sbu += fg.atoms
        sbu.positions += sbu_cop
        self.SBU[sidx].set_atoms(sbu,analyze=False)
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
        for idx,sbu in self:
            atoms = sbu.atoms
            del atoms[self._todel[idx]]
            self[idx].set_atoms(atoms,analyze=True)
            structure += atoms
        bonds   = self.get_bonds()
        mmtypes = self.get_mmtypes()
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
        return structure, bonds, mmtypes

    def write(self,
              f   : str = "./mof",
              ext : str = "gin") -> None:
        """Write a chemical information file to disk in selected format"""
        atoms,bonds,mmtypes = self.get_atoms(dummies=False)
        print(atoms)
        path = os.path.abspath("{path}.{ext}".format(path=f,ext=ext))
        if ext=="gin":
            from autografs.utils.io import write_gin
            write_gin(path,atoms,bonds,mmtypes)
        else:
            ase.io.write(path,atoms)
