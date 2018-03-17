#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
# Copyright : see accompanying license files for details

__author__  = "Damien Coupry"
__credits__ = ["Prof. Matthew Addicoat"]
__license__ = "MIT"
__maintainer__ = "Damien Coupry"
__version__ = '2.0.3'
__status__  = "beta"

import os
import sys
import numpy
import scipy
import typing
import ase
import itertools
import logging
from collections import defaultdict

from autografs.utils.sbu        import read_sbu_database
from autografs.utils.topology   import read_topologies_database
from autografs.utils.mmanalysis import analyze_mm 
from autografs.utils.topology   import Topology

logger = logging.getLogger(__name__) 


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
        logger.debug("Creating Framework instance.")
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

    def copy(self) -> object:
        """Return a copy of itself as a new instance"""
        new = self.__class__(topology=self.get_topology(),
                             SBU = self.SBU.copy(),
                             mmtypes= self.get_mmtypes(),
                             bonds=self.get_bonds())
        new._todel = self._todel.copy()
        return new

    def set_topology(self,
                     topology : ase.Atoms) -> None:
        """Set the topology attribute with an ASE Atoms object."""
        logger.debug("Setting topology.")
        self.topology = topology.copy()
        return None

    def get_topology(self) -> ase.Atoms:
        """Return a copy of the topology attribute as an ASE Atoms object."""
        return self.topology.copy()

    def get_supercell(self,
                      m : tuple = (2,2,2)) -> object:
        """Return a framework supercell usin m as multiplier"""
        # TODO !!! [CRITICAL] correct tagging needed!!!.
        if isinstance(m,int):
            m = (m,m,m)
        logger.info("Creating supercell {0}x{1}x{2}.".format(*m))
        # get the offset direction ranges
        x = list(range(0,m[0],1))
        y = list(range(0,m[1],1))
        z = list(range(0,m[2],1)) 
        # new framework object
        supercell = self.copy()
        ocell = supercell.topology.get_cell()
        otopo = supercell.topology.copy()
        cellfactor = numpy.asarray([x[-1]+1,y[-1]+1,z[-1]+1],dtype=float)
        newcell = ocell.dot(numpy.eye(3)*cellfactor)
        supercell.topology.set_cell(newcell,scale_atoms=False)
        # we need a topology object to correctly tag.
        supertopo = Topology(name="supercell",atoms=otopo*m)
        noff  = 0
        L     = len(otopo)
        # iterate over offsets and add the corresponding objects
        for offset in itertools.product(x,y,z):
            # central cell, ignore
            if offset==(0,0,0):
                for atom in otopo.copy():
                    if atom.symbol=="X":
                        continue
                    if atom.index not in supercell.SBU.keys():
                        continue
                    # directly tranfer new tags
                    sbu = supercell[atom.index]
                    sbu.transfer_tags(supertopo.fragments[atom.index])           
            else:
                noff += 1 
                coffset = ocell.dot(offset)
                for atom in otopo.copy():
                    atom.position += coffset
                    supercell.topology.append(atom)
                    if atom.symbol=="X":
                        continue
                    newidx = len(supercell.topology)-1
                    # check that the SBU was not deleted before
                    if atom.index not in supercell.SBU.keys():
                        continue
                    sbu = supercell[atom.index].copy()
                    sbu.atoms.positions += coffset
                    sbu.transfer_tags(supertopo.fragments[newidx])
                    supercell.append(index = newidx,
                                     sbu   = sbu,
                                     update= False)
                    supercell._todel[newidx] = list(supercell._todel[atom.index])
        # redo the tagging
        return supercell

    def append(self,
               index   : int,
               sbu     : ase.Atoms,
               update  : bool = False) -> None:
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
        logger.debug("Appending SBU {n} to framework.".format(n=sbu.name))
        self.SBU[index] = sbu
        if update:
            # make the bonds matrix with a new block
            self.bonds   = self.get_bonds()
            # append the atom types
            self.mmtypes = self.get_mmtypes()
        return None

    def get_bonds(self) -> numpy.ndarray:
        """Return and update the current bond matrix"""
        logger.debug("Updating framework bond matrix.")
        bonds = []
        for _,sbu in self:
            bonds.append(sbu.bonds)
        bonds = scipy.linalg.block_diag(*bonds)
        self.bonds = bonds
        return bonds

    def get_mmtypes(self) -> numpy.ndarray:
        """Return and update the current bond matrix"""
        logger.debug("Updating framework MM types.")
        mmtypes = []
        for _,sbu in self:
            mmtypes.append(sbu.mmtypes)
        mmtypes = numpy.hstack(mmtypes)
        self.mmtypes = mmtypes
        return numpy.copy(self.mmtypes)

    def scale(self,
              alpha  : numpy.ndarray) -> None:
        """Scale the building units positions by a factor alpha.

        This uses the correspondance between the atoms in the topology
        and the building units in the SBU list. Indeed, SBU[i] is centered on 
        topology[i]. By scaling the topology, we obtain a new center for the 
        sbu.
        alpha -- scaling factor
        """
        logger.debug("Scaling framework by {0:.3f}x{1:.3f}x{2:.3f}.".format(*alpha))
        # get the scaled cell, normalized
        I     = numpy.eye(3)*alpha
        cell  = self.topology.get_cell()
        ncell = numpy.linalg.norm(cell,axis=0)
        cell  = cell.dot(I*1.73/ncell)
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
        logger.info("Refining unit cell.")
        def MSE(x : float) -> float:
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
            logger.info("\tScaling error = {e:>5.3f}".format(e=mse))
            return mse
        # first get an idea of the bounds.
        # minimum cell of a mof should be over 2.0 Ang.
        # and directions with no pbc should be 1.0
        alpha0[alpha0<1e-6] = 20.0
        low  = 0.1
        high = 1.5
        if numpy.amin(low*alpha0)<2.0:
            low   = 1.5/numpy.amin(alpha0)
            high *= 0.1/low
        # optimize
        result = scipy.optimize.minimize_scalar(fun    = MSE,
                                                bounds = (low,high),
                                                method = "Bounded",
                                                options= {"disp"  : True,
                                                          "xatol" : 1e-02})
        # scale with result
        alpha = result.x*alpha0
        logger.info("Best scaling achieved by {0:.3f}x{1:.3f}x{2:.3f}.".format(*alpha))
        self.scale(alpha=alpha)
        return None

    def rotate(self,
               index : int,
               angle : float) -> None:
        """Rotate the SBU at index around a Cinf symmetry axis"""
        if self[index].shape[1]==2:
            logger.info("Rotating {idx} by {a}.".format(idx=index,a=angle))
            axis = [x.position for x in self[index].atoms 
                               if x.symbol=="X"]
            axis = numpy.asarray(axis)
            axis = axis[0]-axis[1]
            self[index].atoms.rotate(v=axis,a=angle)
        return None

    def flip(self,
             index : int) -> None:
        """Flip the SBU at index around a C* symmetry axis or Sigmav plane"""
        logger.info("Flipping {idx}".format(idx=index))
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
        if symbol is not None:
            logger.info("Listing available functionalizable {s}".format(symbol))
        else:
            logger.info("Listing all available functionalization sites")
        sites = []
        for idx,sbu in self:
            bonds = sbu.bonds
            for atom in sbu.atoms:
                if symbol is not None and atom.symbol!=symbol:
                    continue
                if atom.symbol=="X":
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
        logger.info("Functionalization of atom {1} in slot {0}.".format(*where))
        try:
            logger.info("\t|--> replace by {f}.".format(fg_name))
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
            self._todel[sidx] += [aidx,xidx+len(sbu)]
            sbu += fg.atoms
            sbu.positions += sbu_cop
            self.SBU[sidx].set_atoms(sbu,analyze=False)
            self.SBU[sidx].bonds = scipy.linalg.block_diag(bonds,
                                                           fg.bonds)
            self.SBU[sidx].mmtypes = numpy.hstack([self.SBU[sidx].mmtypes,
                                                  fg.mmtypes])
        except Exception as exc:
            logger.error("\t|--> ERROR WHILE FUNCTIONALIZING.")
            logger.error("\t|--> {exc}".format(exc))
        return None

    def get_atoms(self,
                  dummies : bool = False) -> ase.Atoms:
        """Return the concatenated Atoms objects.

        The concatenation can either remove the dummies and
        connect the corresponding atoms or leave hem in place
        clean -- remove the dummies if True
        """
        logger.debug("Creating ASE Atoms from framework.")
        if dummies:
            logger.debug("\tDummies will be kept.")
            logger.debug("\tNo connection between SBU will occur.")
        else:
            logger.debug("\tDummies will be removed during connection.")
        # concatenate every sbu into one Atoms object
        framework = self.copy()
        cell = framework.topology.get_cell()
        pbc  = framework.topology.get_pbc()
        structure = ase.Atoms(cell=cell,pbc=pbc)
        for idx,sbu in framework:
            atoms = sbu.atoms
            todel = framework._todel[idx]
            if len(todel)>0:
                del atoms[todel]
            framework[idx].set_atoms(atoms,analyze=True)
            structure += atoms
        bonds   = framework.get_bonds()
        mmtypes = framework.get_mmtypes()
        symbols = numpy.asarray(structure.get_chemical_symbols())
        if not dummies:
            # keep track of dummies
            xis   = [x.index for x in structure if x.symbol=="X"]
            tags  = structure.get_tags()
            pairs = [numpy.argwhere(tags==tag) for tag in set(tags[xis])]
            # raise
            for pair in pairs:
                # if lone dummy, cap with hydrogen
                if len(pair)==1:
                    xi0 = pair[0]
                    xis.remove(xi0)
                    symbols[xi0] = "H"
                    mmtypes[xi0] = "H_" 
                else:
                    xi0,xi1 = pair
                    bonds0  = numpy.where(bonds[:,xi0]>0.0)[0]
                    bonds1  = numpy.where(bonds[:,xi1]>0.0)[0]
                    # dangling bit, mayhaps from defect
                    if len(bonds0)==0 and len(bonds1)!=0:
                        xis.remove(xi1)
                        symbols[xi1] = "H"
                        mmtypes[xi1] = "H_" 
                    elif len(bonds1)==0 and len(bonds0)!=0:
                        xis.remove(xi0)
                        symbols[xi0] = "H"
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
        path = os.path.abspath("{path}.{ext}".format(path=f,ext=ext))
        logger.info("Framework saved to disk at {p}.".format(p=path))
        if ext=="gin":
            from autografs.utils.io import write_gin
            write_gin(path,atoms,bonds,mmtypes)
        else:
            ase.io.write(path,atoms)
