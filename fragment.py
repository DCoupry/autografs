# -*- coding: utf-8 -*-

"""
This module defines the fragment object, which is a wrapper around the :func:`~ase.Aroms` object
"""


from ase import Atoms
import numpy



class Fragment(Atoms):

    """ 
    Class for the Secondary Building Units in the Autografs framework generator.
    Is a modified version of the ASE Atoms object, plus attributes of form, name, size and type
    Attributes awaited by the constructor: 
        -- unit : "center", "linker" or "functional_group"
        -- name : string describing the chemical nature of the SBU. e.g: "Benzene linear linker".
        -- shape : geometry of the SBU after symmetrization. e.g : linear, triangle, tetrahedral ...
                   see the topology module for more information.
        -- idx : index of fragment in the corresponding model, if necessary
        -- bonds : array of bonding information. passed as a matrix of N*N dimensions; 
                   N being the number of atoms in the fragment. the element i, j of the bonds array
                   is the bond order between atom i and atom j (default is 0)
        -- mmtypes : list of all UFF atomtypes names in the molecule, indexed like their corresponding atoms
    """

    def __init__(self, symbols=None,
                 positions=None, numbers=None,
                 tags=None, momenta=None, masses=None,
                 magmoms=None, charges=None,
                 scaled_positions=None,
                 cell=None, pbc=None, celldisp=None,
                 constraint=None,
                 calculator=None, info=None, 
                 unit=None, name=None, shape=None, idx=None, bonds=None, mmtypes=None):

        Atoms.__init__(self, symbols, positions, numbers,
                       tags, momenta, masses, magmoms, charges,
                       scaled_positions, cell, pbc, celldisp,
                       constraint, calculator, info)

        #defines if fragment is linker, center, or functional group
        self.unit = unit
        #simple string for naming and output purposes
        self.name = name
        #shape of the fragment.
        self.shape = shape
        #index of the fragment in the model
        self.idx = idx 
        #bond matrix for book keeping
        self.bonds = numpy.zeros((len(self), len(self)))
        if bonds is not None:
            self.bonds = numpy.array(bonds)
        self.mmtypes = [None,]*len(self)
        if mmtypes is not None:
            self.mmtypes=mmtypes


    def extend(self, other):
      
        """ 
        Modification of the Atoms method. 
        Takes bonding information into account.
        """

        #create new bond matrix of appropriate size and shape
        newbonds = numpy.zeros((len(self)+len(other), len(self)+len(other)))
        #fill in the existing bonds
        newbonds[:len(self), :len(self)] = self.bonds
        #if bonding info exists for the other molecule, fill it in. else, stays zero
        if isinstance(other, Fragment):
            newbonds[len(self):, len(self):] = other.bonds
        self.bonds = newbonds
        #new mmtypes
        self.mmtypes.extend(other.mmtypes) 
        #calls the Atoms extender
        Atoms.extend(self, other)


    def copy(self):

        """
        Modification of the Atoms method. 
        Takes additional information into account.
        """

        fragment = Atoms.copy(self)
        fragment.name = self.name 
        fragment.shape = self.shape 
        fragment.unit = self.unit
        fragment.idx = self.idx
        fragment.bonds = self.get_bonds() 
        fragment.mmtypes = self.mmtypes

        return fragment


    def __imul__(self, m):
      
        """
        Modification of the Atoms method. 
        Takes bonding information into account.
        """

        #transforms simple number in three components
        if isinstance(m, int):
            m = (m, m, m)
        #get the repetition number for the bonding info
        repetitions = numpy.prod(m)
        #update the bond matrix
        newbonds = numpy.zeros((len(self)*repetitions,len(self)*repetitions))
        for repetition in range(repetitions):
            newbonds[len(self)*repetition:len(self)*(repetition+1), len(self)*repetition:len(self)*(repetition+1)] = self.bonds
        self.bonds = newbonds
        self.mmtypes *= repetitions
        #calls the Atoms multiplier
        Atoms.__imul__(self, m)

        return self


    def __delitem__(self, i):

        """
        Modification of the Atoms method. 
        Takes bonding information into account.
        """

        self.bonds = numpy.delete(self.bonds, i, axis=0)
        self.bonds = numpy.delete(self.bonds, i, axis=1)
        if isinstance(i, int):
            del self.mmtypes[i]
        else:
            self.mmtypes = [mmtype for j, mmtype in enumerate(self.mmtypes) if j not in i]
        Atoms.__delitem__(self, i)


    def set_bond(self, index1, index2, bondOrder):

        """
        Simple method to update bonding information
        """

        self.bonds[index1, index2] = bondOrder
        self.bonds[index2, index1] = bondOrder


    def get_mmtypes(self):

        """
        Returns a copy of the UFF atom types in order
        """

        new_mmtypes = []
        for mmtype in self.mmtypes:
            new_mmtypes.append(mmtype)
        return new_mmtypes


    def get_bonds(self):

        """
        Returns a copy of the bonding matrix
        """

        return self.bonds.copy()


    def write(self, name="framework", verbose=True):

        """
        Write the fragment to a file of correct extension. by default, the file is an input file for the
        General Utility Lattice Program (GULP) in the current directory.
        GULP relevan litterature:
            -- GULP - a computer program for the symmetry adapted simulation of solids, J.D. Gale, JCS Faraday Trans., 93, 629 (1997)
            -- Empirical potential derivation for ionic materials, J.D. Gale, Phil. Mag. B, 73, 3, (1996)
            -- The General Utility Lattice Program, J.D. Gale and A.L. Rohl, Mol. Simul., 29, 291-341 (2003)
            -- GULP: Capabilities and prospects, J.D. Gale, Z. Krist., 220, 552-554 (2005)
        """

        from ase.io import write as w
        from utils  import write_gin
        import platform
        import os

        if "windows" in platform.system().lower():
            name = name.replace('\\','/',)
        extension = os.path.splitext(name)[1]
        print extension
        print len(extension)
        print extension==""
        if (len(extension)==0) or (extension ==".gin"):    
            write_gin(name+".gin", self)
        else:
            w(name, self)
        return