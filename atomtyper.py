# -*- coding: utf-8 -*-
#!/usr/bin/python2.7

"""
Title       : AtomTyper v0.2 
Author      : D.E.Coupry, Jacobs University Bremen
Description : Assigns UFF atom types to any ASE-readable cristal, and returns the corresponding array.
              Also guesses bond orders, including for paddlewheel-like compounds, H-bonds, etc.
              Designed to work closely with UFF4MOF extension of UFF
              See the file 'uff4mof.csv' for designing your own custom libraries.
              Caution: the angles specified are not always the UFF equilibrium bond angles,
              they are the angles best suited for detection.
"""

from ase.data import chemical_symbols,covalent_radii
from ase.calculators.neighborlist import *
from ase import Atom, Atoms
from ase.io   import read
from fragment import *
import numpy 
import pandas
import itertools
import platform
import argparse
import math
import time
import ase
import sys
import os



def py_ang(v1, v2):
    """ 
    Returns the angle in radians between vectors 'v1' and 'v2'    
    """
    
    cosang = numpy.dot(v1, v2)
    sinang = numpy.linalg.norm(numpy.cross(v1, v2))
    return numpy.arctan2(sinang, cosang)


def check_planarity(structure, atom, atoms, epsilon):
    
    """ 
    Checks the coplanarity of a list of ASE Atom objects with an ASE Atom object using dihedrals
    """
    
    dihedrals = []
    for plan in itertools.combinations(atoms,3):
        list_of_vectors = []
        for a in itertools.permutations(plan,3):
            distance_1 = structure.get_distance(a[0], a[1],mic= True)
            distance_2 = structure.get_distance(a[1], a[2],mic= True)
            list_of_vectors.append((a, distance_1+distance_2))
        list_of_vectors.sort(key = lambda x : x[1])
        best_vectors = list(zip(*list_of_vectors)[0][0])
        best_vectors.append(atom.index)
        if abs(math.degrees(structure.get_dihedral(best_vectors))) < epsilon or abs(math.degrees(structure.get_dihedral(best_vectors))) > (360-epsilon):
            dihedrals.append(True)
        else:
            dihedrals.append(False)
    if all(dihedrals):
        return True
    else:
        return False


def check_ring_planarity(structure, ring, neighbourList):

    """ 
    Checks the planarity of a ring. some of this is quite involved...
    Sometimes, should be modified for something smarter...
        -- based on 5 consecutive atoms
        -- enough, since max ring size is 6
    """

    newMol = ase.Atoms()
    #point of view is taken arbitrarily
    pov = ring[0]
    newMol.append(structure[pov])
    queue_ring = [r for r in ring]
    queue_ring.remove(pov)
    connectivityInfo = ConnectivityList(structure, structure[pov], neighbourList)
    connectivity = connectivityInfo.connectivity
    offsets = connectivityInfo.offsets
    for bondedAtomIndex in connectivity:
        if bondedAtomIndex in queue_ring and list(offsets[numpy.where(connectivity == bondedAtomIndex)][0])!= [0,0,0]:
            offed = numpy.dot(offsets[list(connectivity).index(bondedAtomIndex)],structure.get_cell())
            newMol.append(ase.Atom('X',(structure.positions[bondedAtomIndex] + offed)))
            queue_ring.remove(bondedAtomIndex)
            connectivityInfo_sec = ConnectivityList(structure, structure[bondedAtomIndex], neighbourList)
            connectivity_sec = connectivityInfo_sec.connectivity
            offsets_sec = connectivityInfo_sec.offsets
            for bondedAtomIndex_sec in connectivity_sec:
                carried_offset = offsets[list(connectivity).index(bondedAtomIndex)]
                if bondedAtomIndex_sec in queue_ring and list(offsets_sec[numpy.where(connectivity_sec == bondedAtomIndex_sec)][0])!= [0,0,0]:
                    offed = numpy.dot(offsets_sec[list(connectivity_sec).index(bondedAtomIndex_sec)]+carried_offset, structure.get_cell())
                    newMol.append(ase.Atom('X',(structure.positions[bondedAtomIndex_sec]+ offed)))
                    queue_ring.remove(bondedAtomIndex_sec)
                elif bondedAtomIndex_sec in queue_ring:
                    offed =  numpy.dot(carried_offset, structure.get_cell())
                    newMol.append(ase.Atom('X',(structure.positions[bondedAtomIndex_sec]+offed)))
                    queue_ring.remove(bondedAtomIndex_sec)
        elif bondedAtomIndex in queue_ring:
            newMol.append(structure[bondedAtomIndex])
            queue_ring.remove(bondedAtomIndex)
            connectivityInfo_sec = ConnectivityList(structure, structure[bondedAtomIndex], neighbourList)
            connectivity_sec = connectivityInfo_sec.connectivity
            offsets_sec = connectivityInfo_sec.offsets
            for bondedAtomIndex_sec in connectivity_sec:
                if bondedAtomIndex_sec in queue_ring and list(offsets_sec[numpy.where(connectivity_sec == bondedAtomIndex_sec)][0])!= [0,0,0]:
                    offed = numpy.dot(offsets_sec[list(connectivity_sec).index(bondedAtomIndex_sec)], structure.get_cell())
                    newMol.append(ase.Atom('X',(structure.positions[bondedAtomIndex_sec]+ offed)))
                    queue_ring.remove(bondedAtomIndex_sec)
                elif bondedAtomIndex_sec in queue_ring:
                    newMol.append(structure[bondedAtomIndex_sec])
                    queue_ring.remove(bondedAtomIndex_sec)
    dihedrals = []
    ring_positions = numpy.array([r.position for r in newMol])
    median_point = numpy.sum(ring_positions, axis=0)/len(newMol)
    newMol.append(ase.Atom('X', median_point))
    ring_planarity = check_planarity(newMol, newMol[len(newMol)-1], [nm.index for nm in newMol[:-1]], 30)
    return ring_planarity



class BondMatrix:

    """ 
    Class bond matrix: provides an easily manipulable matrix storing detailed connectivity information
    attributes awaited by constructor:
        -- ASE Atoms object
    """

    def __init__(self, structure):
        
        #the matrix has an empty column and row, to coincide with atomic indexation in ASE Atoms objects
        self.bond_matrix = numpy.zeros((len(structure)+1,len(structure)+1))


    def __repr__(self):
        
        return numpy.array_repr(self.bond_matrix)


    def __getitem__(self, index1, index2):
        
        return self.bond_matrix[index1+1,index2+1]


    def set_bond(self, index1, index2, newOrder):
        
        """ 
        modifies the bond order between two atoms in the bond matrix """

        self.bond_matrix[index1+1,index2+1] = newOrder
        self.bond_matrix[index2+1,index1+1] = newOrder


    def first_guess(self, structure, bondRadii, neighbourList):

        """ 
        Guesses the bond order in neighbourlist based on covalent radii 
        the radii for BO > 1 are extrapolated by removing 0.15 Angstroms by order 
        see Beatriz Cordero, Veronica Gomez, Ana E. Platero-Prats, Marc Reves, Jorge Echeverria,
        Eduard Cremades, Flavia Barragan and Santiago Alvarez (2008). 
        "Covalent radii revisited". 
        Dalton Trans. (21): 2832-2838 
        http://dx.doi.org/10.1039/b801115j
        """

        BO1=[]
        BO2=[]
        BO3=[]
        for atom in structure:
            BO1.append(bondRadii[atom.symbol][0])
            BO2.append(bondRadii[atom.symbol][1])
            BO3.append(bondRadii[atom.symbol][2])
        for atom in structure:
            connectivity =  ConnectivityList(structure, atom, neighbourList).connectivity
            for bondedAtomIndex in connectivity:
                #Hydrogen has BO of 1
                if atom.symbol == 'H' or structure[bondedAtomIndex].symbol == 'H':
                    self.set_bond(atom.index, bondedAtomIndex, 1)
                #Metal-Metal bonds: if no special case, nominal bond
                elif atom.symbol in transitionMetals and structure[bondedAtomIndex].symbol in transitionMetals:
                    self.set_bond(atom.index, bondedAtomIndex, 0.25)
                #Metal-Organic bonds: most often coordination bonds
                elif (atom.symbol in transitionMetals) != (structure[bondedAtomIndex].symbol in transitionMetals) :
                    self.set_bond(atom.index, bondedAtomIndex, 0.5)
                else:
                    distance = structure.get_distance(atom.index, bondedAtomIndex, mic= True)
                    cutoffs = [BO1[atom.index] + BO1[bondedAtomIndex],
                               BO2[atom.index] + BO2[bondedAtomIndex],
                               BO3[atom.index] + BO3[bondedAtomIndex]]
                    errors = [abs(distance-c) for c in cutoffs]
                    bestFit = errors.index(min(errors))+1
                    self.set_bond(atom.index, bondedAtomIndex, bestFit)



class ConnectivityList:

    """ 
    Class connectivityList : provides a correctable connectivity information.
    attributes awaited by constructor:
        -- ASE Atom object
        -- corresponding ASE Atoms object
        -- Atoms object corresponding ASE neighborlist 
    """

    def __init__(self, structure, atom, neighbourList):
    
        connectivity, offsets = neighbourList.get_neighbors(atom.index)
        self.connectivity = connectivity
        self.offsets = offsets


    def remove_bond(self, atom):

        """ 
        Removes the bond between two atoms in neighbourlist 
        """

        work_connectivity = list(self.connectivity)
        work_offset = list(self.offsets)
        offset_index = work_connectivity.index(atom.index)
        work_connectivity.remove(atom.index)
        del work_offset[offset_index]
        self.connectivity = numpy.array(work_connectivity)
        self.offsets = numpy.array(work_offset)



class NeighborListMaker(NeighborList):

    """
    Class NeighborListNeue: facilitates generation of neighbour lists for molecules
    attributes awaited by constructor:
        -- ASE Atoms object
        -- List of bond radii 
    """
    
    def __init__(self, structure, bondRadii, bondOrder, skin):
    
        covalentRadii = []
        for atom in structure:
            covalentRadii.append(bondRadii[atom.symbol][bondOrder-1])
        neighbourList = NeighborList(covalentRadii,skin = skin,self_interaction=False,bothways=True)
        neighbourList.build(structure)
        self.neighbourList = neighbourList



class AtomTyper:

    """ 
    ClassAtomTyper : assigns assign UFF type to an atom.
    attributes awaited by constructor:
        -- ASE Atom object
        -- corresponding ASE Atoms object 
    """
    
    def __init__(self, atom, structure, neighbourList, mmtypes,  uff_atypes, uff_reference):
   
        self.atom = atom
        self.structure = structure
        self.mmtypes = mmtypes
        self.typesymb = self.typesymb()
        self.connectivityInfo = ConnectivityList(self.structure, self.atom, neighbourList)
        self.neighbourList = neighbourList
        self.uff_atypes = uff_atypes
        self.uff_atypes_pureRappe = uff_reference


    def correct_connectivity(self, bondMatrix):
   
        """ 
        Modifies the connectivityInfo, so that further work is done on correct infromation
            -- the case of ferrocene-likes is treated
            -- no alkali-alkali, or metal-alkali bond
            -- metal-metal bonds are nominal by default
            -- carbon-metal bonds are forbidden is the carbon is sp3 hybridized
            -- hydrogen to metal direct bonds are forbidden
            -- pincer ligands are also treated
        """

        connectivity = self.connectivityInfo.connectivity
        bondsToRemove = []
        if self.atom.symbol in transitionMetals:
            #ferrocene-likes
            if self.is_ferrocene():
                return
            secondaryConnectivity = []
            for bondedAtomIndex in connectivity:
                newConnectivity = ConnectivityList(self.structure,self.structure[bondedAtomIndex], self.neighbourList).connectivity
                #metal to metal
                if self.structure[bondedAtomIndex].symbol in transitionMetals:
                    if self.structure.get_distance(self.atom.index, bondedAtomIndex, mic=True) < 2.0:
                        bondMatrix.set_bond(self.atom.index, bondedAtomIndex, 0.25)
                    else:
                        bondMatrix.set_bond(self.atom.index, bondedAtomIndex, 0)
                        bondsToRemove.append(bondedAtomIndex)
                #metal linked to hydrogen
                if self.structure[bondedAtomIndex].symbol == 'H':
                    bondMatrix.set_bond(self.atom.index, bondedAtomIndex, 0)
                    bondsToRemove.append(bondedAtomIndex)
                #metal linked to carbon
                if self.structure[bondedAtomIndex].symbol == 'C':
                    if len(newConnectivity)>3:
                        bondMatrix.set_bond(self.atom.index, bondedAtomIndex, 0)
                        bondsToRemove.append(bondedAtomIndex)
                #metal linked to alkali
                if self.structure[bondedAtomIndex].symbol in alkali:
                    bondMatrix.set_bond(self.atom.index, bondedAtomIndex, 0)
                    bondsToRemove.append(bondedAtomIndex)
                #pincer ligands
                for secondaryBondedAtomIndex in newConnectivity:
                    secondaryConnectivity.append(secondaryBondedAtomIndex)
            for secondaryBondedAtomIndex_key in secondaryConnectivity:
                if secondaryConnectivity.count(secondaryBondedAtomIndex_key) > 1 and secondaryBondedAtomIndex_key != self.atom.index:
                    if self.structure[secondaryBondedAtomIndex_key] in ['C', 'P']:
                        bondMatrix.set_bond(self.atom.index, secondaryBondedAtomIndex_key, 0)
                        bondsToRemove.append(secondaryBondedAtomIndex_key)            
        #carbon linked to metal
        if self.atom.symbol == 'C':
            for bondedAtomIndex in connectivity:
                if self.structure[bondedAtomIndex].symbol in transitionMetals and len(connectivity)>3:
                    bondMatrix.set_bond(self.atom.index, bondedAtomIndex, 0)
                    bondsToRemove.append(bondedAtomIndex)
        if self.atom.symbol in alkali:
            for bondedAtomIndex in connectivity:
                #alkali linked to metal
                if self.structure[bondedAtomIndex].symbol in transitionMetals:
                    bondMatrix.set_bond(self.atom.index, bondedAtomIndex, 0)
                    self.connectivityInfo.remove_bond(self.structure[bondedAtomIndex])
                #alkali linked to alkali
                if self.structure[bondedAtomIndex].symbol in alkali:
                    bondMatrix.set_bond(self.atom.index, bondedAtomIndex, 0)
                    bondsToRemove.append(bondedAtomIndex)
        for atomToRemove in list(set(bondsToRemove)):
            if atomToRemove in self.connectivityInfo.connectivity:
                self.connectivityInfo.remove_bond(self.structure[atomToRemove])
    

    def is_ferrocene(self):
    
        """ 
        Returns True if the atom is in the center of a ferrocene-like mofif 
        """

        connectivity = self.connectivityInfo.connectivity
        if len(connectivity) > 10:
            carbons = 0
            for bondedAtomIndex in connectivity:
                if self.structure[bondedAtomIndex].symbol == 'C':
                    carbons += 1
            if carbons > 10:
                return True
            else:
                return False


    def typesymb(self):
        
        """ 
        Returns the first part of a UFF atom type according to UFF symbolism 
        """

        asymb = self.atom.symbol
        if len(asymb) == 1 :
            asymb =''.join([asymb,'_'])
        return asymb


    def is_paddlewheel(self):

        """ 
        Returns True if the atom is part of a paddlewheel motif 
        """

        connectivity = self.connectivityInfo.connectivity
        offsets = self.connectivityInfo.offsets
        metalNeighbours = 0 
        oxygenNeighbours = 0
        for bondedAtomIndex in connectivity:
            if self.structure[bondedAtomIndex].symbol in transitionMetals:
                metalNeighbours += 1
            if self.structure[bondedAtomIndex].symbol == 'O':
                oxygenNeighbours += 1
        if metalNeighbours == 1 and oxygenNeighbours == 4 and len(connectivity) >= 5 and len(connectivity) <=6 :
            print 'Paddlewheel detected'
            return True
        else:
            return False


    def is_carboxylate(self):
        """ 
        Returns the atoms of the carboxylate if the atom is a carboxylate carbon, else returns None 
        """
        
        connectivity = self.connectivityInfo.connectivity
        offsets = self.connectivityInfo.offsets
        oxygens = [o for o in connectivity if self.structure[o].symbol == 'O']
        if len(oxygens)==2:
            onlyMetals = []
            for oxygenIndex in oxygens:
                oxygenConnectivity = list(ConnectivityList(self.structure, self.structure[oxygenIndex], self.neighbourList).connectivity)
                oxygenConnectivity.remove(self.atom.index)
                if all([self.structure[bondedToOxygenIndex].symbol in transitionMetals for bondedToOxygenIndex in oxygenConnectivity]):
                    onlyMetals.append(True)
                else:
                    onlyMetals.append(False)
            if all(onlyMetals):
                return oxygens
            else:
                return None
    

    def work_moldecule(self):
        
        """ 
        Returns a pseudo molecule representing the immediale local environment of an atom 
        """

        newMol = ase.Atoms()
        newMol.append(self.atom)
        centralAtom = newMol[0]
        connectivity = self.connectivityInfo.connectivity
        offsets = self.connectivityInfo.offsets
        for bondedAtomIndex in connectivity:
            offset = offsets[numpy.where(connectivity==bondedAtomIndex)][0]
            if any(offset != numpy.array([0,0,0])):
                newMol.append(ase.Atom(self.structure[bondedAtomIndex].symbol,
                             (self.structure[bondedAtomIndex].position + numpy.dot(offset, self.structure.get_cell())),
                              index = bondedAtomIndex))
            else:
                newMol.append(self.structure[bondedAtomIndex])
        return newMol, centralAtom


    def is_bipyramid(self):
        
        """ 
        Returns True if the atom is the center of a bipyramidal cluster 
        """

        connectivity = self.connectivityInfo.connectivity
        offsets = self.connectivityInfo.offsets
        bipyramids = []
        if len(connectivity) >= 3:
            planOrderOptions = range(3, len(connectivity)+1)
            for planOrder in planOrderOptions:
                newMol, centralAtom = self.work_moldecule()
                newNeighbourList=NeighborListMaker(newMol,bondRadii, bondOrder=1, skin=0.2).neighbourList
                newConnectivity = list(ConnectivityList(newMol, centralAtom, newNeighbourList).connectivity)
                possiblePlans = list(itertools.combinations(newConnectivity, planOrder))
                plans = []
                for plan in possiblePlans:
                    if check_planarity(newMol, centralAtom, plan, epsilon=30):
                        plans.append(plan)
                if len(plans) != 0:
                    bipyramids.append(plans)
                else:
                    break
            try:
                maxPlan = max(bipyramids, key=lambda x : len(x[0]))[0]
            except Exception as e:
                maxPlan = [[]]
            if len(maxPlan) >= 3:
                notInPlan = [i for i in newMol  if i.index not in maxPlan and i.index != centralAtom.index]
                if len(notInPlan) ==1 or len(notInPlan) == 2:
                    print  'Possible bipyramid'
                    if all([all([(abs(math.degrees(newMol.get_angle([nip.index,
                                                                    centralAtom.index,
                                                                    mp]))-90))<=15 for mp in maxPlan]) for nip in notInPlan]):
                        print '\t Pyramid confirmed : base', len(maxPlan)
                        return True
                    else:
                        return False
                elif len(notInPlan) == 0 and len(maxPlan) != 3:
                    print '\t Pure planar detected'
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
    

    def get_angles(self):
        
        """ 
        Generate a list of all bond angles in the local environment  
        """
        
        newMol, centralAtom = self.work_moldecule()
        newNeighbourList=NeighborListMaker(newMol,bondRadii, bondOrder=1, skin=0.2).neighbourList
        newConnectivity = list(ConnectivityList(newMol, centralAtom, newNeighbourList).connectivity)
        if len(newConnectivity) <= 1:
            angles = [180]
        else:
            validAngles = [permutation for permutation in list(itertools.permutations(newConnectivity,2))]
            angles = [math.degrees(newMol.get_angle([a[0], centralAtom.index, a[1]])) for a in validAngles]
        return angles
    

    def fit_angles(self, angles, angleToFit, maxEpsilon=27.5):

        """ 
        Returns the respective fits of existing parameters to the experimental angles 
        """
        
        referenceAngle = angleToFit
        print 'Angle to fit: ', referenceAngle 
        expAngles = numpy.array(angles)
        epsilon = 2.5
        matchFound = False
        while not matchFound and epsilon <= maxEpsilon :
            minAngle = referenceAngle - epsilon
            maxAngle = referenceAngle + epsilon
            minDoubleAngle = (2 * referenceAngle) - epsilon
            maxDoubleAngle = (2 * referenceAngle) + epsilon
            minHalfAngle = (0.5 * referenceAngle) - epsilon
            maxHalfAngle = (0.5 * referenceAngle) + epsilon
            anglePercent = 100 * ((minAngle < expAngles) & (expAngles < maxAngle)).sum()/float(len(expAngles))
            halfAnglePercent = 100 * ((minHalfAngle < expAngles) & (expAngles < maxHalfAngle)).sum()/float(len(expAngles))
            doubleAnglePercent = 100 * ((minDoubleAngle < expAngles) & (expAngles < maxDoubleAngle)).sum()/float(len(expAngles))
            fullPercent = anglePercent + halfAnglePercent + doubleAnglePercent
            if fullPercent > 100:
                fullPercent = 100
            if fullPercent >= 75:
                matchFound = True
                break
            else:
                epsilon += 2.5
        if matchFound:
            print '\t',fullPercent, 'Percent of angles were matched to uff type with ',epsilon, 'degrees tolerance'
        else:
            print '\t Max tolerance reached. no match found'
        return matchFound, fullPercent, epsilon


    def uff_option_byangle(self, angle):
        
        """ 
        Generates the best uff option for a parameter with a known angle. e.g: returns the best octahedral 
        """
        
        uffOptions = [atype for atype in self.uff_atypes.index if atype.startswith(self.typesymb) 
                                                               and self.uff_atypes.loc[atype, "angle"] == angle  
                                                               and not atype.endswith('_R')]
        uffOptions.sort(key=lambda x : abs(self.uff_atypes.loc[x,"radius"]-len(self.connectivityInfo.connectivity)))
        if len(uffOptions) > 0:
            if all([option in self.uff_atypes_pureRappe.index for option in uffOptions]):
                return uffOptions[0]
            else:
                uff4mofOptions = [option for option in uffOptions if option not in self.uff_atypes_pureRappe.index]
                return uff4mofOptions[0]
        else:
            return None
    

    def uff_options_byelement(self):

        """
        Generates a list of all available atom types for this element
        """

        uffOptions = [atype for atype in self.uff_atypes.index if atype.startswith(self.typesymb) and not atype.endswith('_R')]
        return uffOptions


    def type_atom(self, bondMatrix, rings):

        """ 
        Typing script proper. returns the best fitting UFF atom type 
        """
        
        #preliminary fix of bondmatrix and connectivity
        print 'Bond fix applied'
        self.correct_connectivity(bondMatrix)
        print 'Neighbouring atoms : '
        printinfo=zip([self.structure[s].symbol for s in self.connectivityInfo.connectivity], self.connectivityInfo.connectivity)
        for printsymb, printindex in printinfo:
            print "\t", printsymb, "number", printindex

        #already typed
        if self.mmtypes[self.atom.index] != None:
            print '\tType already assigned'
            return
        
        if self.atom.symbol=="X":
            self.mmtypes[self.atom.index] = "H_"

        # Only one type exists and suffices
        monotypes = ['H', 'F', 'Cl', 'Br', 'I']
        if self.atom.symbol in monotypes:
            self.mmtypes[self.atom.index] = self.typesymb
            return
        
        #special paddlewheel check. priority is given to UFF4MOF parameters
        if self.atom.symbol in transitionMetals:
            if self.is_paddlewheel():
                print 'Paddlewheel detected : types and bond orders applied'
                self.mmtypes[self.atom.index] = self.uff_option_byangle(90.0)
                for bondedAtomIndex in self.connectivityInfo.connectivity:
                    if self.structure[bondedAtomIndex].symbol == 'O':
                        self.mmtypes[bondedAtomIndex] = 'O_2'
                    if self.structure[bondedAtomIndex].symbol in transitionMetals:
                        if self.structure[bondedAtomIndex].symbol == self.atom.symbol and self.atom.symbol in mmbond.iterkeys():
                            bondMatrix.set_bond(self.atom.index, bondedAtomIndex, mmbond[self.atom.symbol])
                        else:
                            bondMatrix.set_bond(self.atom.index, bondedAtomIndex, 0.25)

        #special aromaticity check
        for ring in rings:
            if self.atom.index in ring:
                planar =  check_ring_planarity(self.structure, ring, self.neighbourList) 
                if planar and len(ring) > 4 and all([self.structure[r].symbol in ['S', 'N', 'C', 'O'] for r in ring]) :
                    print 'Aromatic ring detected : ', ring 
                    print '\tAromatic atype applied to full ring'
                    print '\tBond order fix'
                    for ringAtom in ring:
                        self.mmtypes[ringAtom] = self.structure[ringAtom].symbol + '_R' 
                        #for combination in itertools.combinations(ring,2):
                    for pair in itertools.combinations(ring,2): 
                        if bondMatrix.__getitem__(pair[0],pair[1]) != 0:
                            print  '\t\tAtom {0} and atom {1} are resonant'.format(pair[0],pair[1])
                            bondMatrix.set_bond(pair[0],pair[1],1.5)
                    return

        #special carboxylate check
        if self.atom.symbol == 'C':
            oxygens = self.is_carboxylate()
            if oxygens != None:
                self.mmtypes[self.atom.index] = 'C_R'
                for oxygenIndex in oxygens:
                    self.mmtypes[oxygenIndex] = 'O_2'
                    bondMatrix.set_bond(self.atom.index, oxygenIndex, 1.5)
                print 'Carboxylate detected : types and resonant bond order applied'
                return

        #special bipyramid check
        if self.atom.symbol in transitionMetals:
            if self.is_bipyramid():
                self.mmtypes[self.atom.index] = self.uff_option_byangle(90.0)
                return
        
        #special ferrocenes check
        if self.atom.symbol in transitionMetals:
            if self.is_ferrocene():
                self.mmtypes[self.atom.index] = self.uff_option_byangle(180.0)
                for bondedAtomIndex in self.connectivityInfo.connectivity:
                    if self.structure[bondedAtomIndex].symbol == 'C':
                        bondMatrix.set_bond(self.atom.index,bondedAtomIndex,0.5)
                print 'Ferrocene-like detected. Linear type is applied'
                return

        #no special case was detected: fitting the parameters one by one
        uffOptions = self.uff_options_byelement()
        anglesToFit = list(set([self.uff_atypes.loc[p, "angle"] for p in uffOptions]))
        fittingAngles = []
        angles = self.get_angles()
        print 'Angles : '
        for angle in angles:
            print '\t', angle
        for angleOption in anglesToFit:
            fit, angleFit, angleEpsilon = self.fit_angles(angles, angleOption)
            if fit:
                fittingAngles.append((angleOption, angleEpsilon, angleFit))
        if len(fittingAngles) == 0:
            print 'No mmtype available with matching angle'
            self.mmtypes[self.atom.index] = None
            return
        else:
            fittingAngles.sort(key=lambda x : x[1])
            bestEpsilons = [fA for fA in fittingAngles if fA[1] == fittingAngles[0][1]]
            bestEpsilons.sort(key=lambda x : x[2])
            self.mmtypes[self.atom.index] = self.uff_option_byangle(bestEpsilons[0][0])
            return



class MolTyper:

    """
    Class MolTyper : assigns UFF types to an ASE readable molecule.
    attributes awaited by constructor:
        -- inputfile containing structure information (minimum: cartesian coordinates) 
    """

    def __init__(self, inputfile=None, structure=None):

        if structure is not None:
            self.structure = Fragment(structure)
        elif inputfile is not None:
            self.structure = Fragment(read(inputfile))
            print self.structure
        else:
            raise AttributeError("No valid structure given to constructor.")
        self.neighbourList = NeighborListMaker(self.structure, bondRadii, bondOrder=1, skin=0.2).neighbourList
        self.rings = self.detect_rings() 


    def create_graph(self):
        
        """ 
        Creates the graph of connections in te structure for ring detection 
        """

        graph = {}
        links = {}
        for atom in self.structure:
            if atom.symbol in transitionMetals:
                graph.update({atom.index : set([])})
                links.update({atom.index :[]})
            else:
                connectivity = ConnectivityList(self.structure, atom, self.neighbourList).connectivity
                graph.update({atom.index : set([atomIndex for atomIndex in connectivity if self.structure[atomIndex].symbol not in transitionMetals])})
                links.update({atom.index : [atomIndex for atomIndex in connectivity if self.structure[atomIndex].symbol not in transitionMetals]})
        return graph,links


    def bfs_paths(self, graph, start, goal):
        
        """ 
        Breadth First Algorithm to check paths between atoms 
        """

        queue = [(start, [start])]
        while queue:
            (vertex, path) = queue.pop(0)
            for next in graph[vertex] - set(path):
                if next == goal:
                    yield path + [next]
                elif len(path) > 6:
                    break
                else:
                    queue.append((next, path + [next]))


    def detect_rings(self):
        
        """ 
        Returns a list of rings in the structure 
        """

        rings = []
        graph, links = self.create_graph()
        for atom in self.structure:
            rings_tmp = []
            for link in links[atom.index]:
                paths = list(self.bfs_paths(graph, atom.index, link))
                for path in paths:
                    #check if the path is a dead end or a false start
                    if len(path) > 2 :
                        rings_tmp.append(sorted(path))
            rings.extend(rings_tmp)
        rings.sort()
        listOfRings = [ring for ring,_ in itertools.groupby(rings) if len(ring) != 0]
        return listOfRings


    def detect_hbonds(self, bondMatrix):
        
        """ 
        Detects Hydrogen bonds, and asigns a custom value for the bond order 
        """

        firstConnectivity = []
        newRadii = [None]*len(self.structure)
        for atom in self.structure:
            if atom.symbol=='H':
                connectivity = ConnectivityList(self.structure, atom, self.neighbourList).connectivity
                if any([self.structure[a].symbol in ['O','N','S'] for a in connectivity]):
                    newRadii[atom.index] = 2.0
                    firstConnectivity.append((atom.index, connectivity))
                else:
                    newRadii[atom.index] = bondRadii[atom.symbol][0]
            else:
                newRadii[atom.index] = bondRadii[atom.symbol][0]
        newNeighbourList = NeighborList(newRadii, skin=0.2, self_interaction=False, bothways=True) 
        newNeighbourList.build(self.structure)
        for atom in self.structure:
            if atom.symbol=='H':
                connectivity = ConnectivityList(self.structure, atom, self.neighbourList).connectivity
                oldOffsets = ConnectivityList(self.structure, atom, self.neighbourList).offsets
                newConnectivity, newOffsets = newNeighbourList.get_neighbors(atom.index)
                for newAtomIndex in newConnectivity:
                    for oldAtomData in firstConnectivity:
                        if oldAtomData[0] == atom.index:
                            if self.structure[newAtomIndex].symbol in ['O', 'N', 'S', 'F', 'Cl', 'Br', 'I']:
                                if newAtomIndex not in oldAtomData[1]:
                                    newOffset = newOffsets[numpy.where(newConnectivity==newAtomIndex)][0]
                                    posNew = self.structure[newAtomIndex].position + numpy.dot(newOffset, self.structure.get_cell())
                                    posH = atom.position
                                    oldOffset = oldOffsets[numpy.where(connectivity==oldAtomData[1][0])]
                                    posOld = self.structure[oldAtomData[1][0]].position + numpy.dot(oldOffset, self.structure.get_cell())[0]
                                    vHNew = posNew - posH
                                    vHOld = posOld - posH
                                    angle = math.degrees(py_ang(vHNew, vHOld))
                                    if angle > 140.0 and angle < 220.0:
                                        global args
                                        print 'Hydrogen bond detected between ',atom.symbol,atom.index,'AND',self.structure[newAtomIndex].symbol,newAtomIndex
                                        bondMatrix.set_bond(atom.index, newAtomIndex, 0.001)
        return
    

    def type_mol(self, library, reference_library):
        
        """ 
        Calls an Atom typer to assign UFF types for each atom of the structure 
        """

        # Load the libraries
        uff_atypes    = pandas.read_csv(library,           index_col="symbol")
        uff_reference = pandas.read_csv(reference_library, index_col="symbol")
        #bondMatrix_numpy = numpy.zeros((len(self.structure)+1,len(self.structure)+1))
        bondMatrix = BondMatrix(self.structure)
        bondMatrix.first_guess(self.structure, bondRadii, self.neighbourList)
        fullTypes = [None]*len(self.structure)
        #typing proper
        for atom in self.structure:
            print "\n-------", atom.symbol, 'Atom number', atom.index 
            typer = AtomTyper(atom, self.structure, self.neighbourList, mmtypes=fullTypes, uff_atypes=uff_atypes, uff_reference=uff_reference)
            typer.type_atom(bondMatrix, self.rings)
            if  typer.mmtypes[atom.index] != None:
                print 'UFF type applied : ', typer.mmtypes[atom.index]
            else:
                print 'No UFF type applied'
        # define H bonds
        # Called last, changes connectivity Info
        self.detect_hbonds(bondMatrix)
        self.structure.mmtypes = typer.mmtypes
        self.structure.bonds = bondMatrix.bond_matrix[1:,1:]
        return


    def get_mmtypes(self):
        
        """
        Returns the list of UFF types
        """

        return self.structure.mmtypes


    def get_guibonds(self):

        """
        returns the bonding information : 
        [(i, j, k), ...]
        for bond order k between atoms i and j (indexed from 1)
        """

        nzbonds = numpy.argwhere(numpy.tril(self.structure.bonds)!=0)
        return [(i+1, j+1, self.structure.bonds[i,j]) for i, j in nzbonds]



# some public objects
# uses the ase.data module fro covalent radii and chemical_symbols
bondRadii = {symbol:[r, r-0.1, r-0.2] for symbol,r in zip(chemical_symbols,covalent_radii)}
# assumed as positively charged through the bonding analysis
transitionMetals = [symbol for symbol in chemical_symbols if symbol not in [chemical_symbols[main_index] for main_index in [1,2,5,6,7,8,9,10,14,15,16,17,18,33,34,35,36,52,53,54,85,86]]]
# assumed as positively charged through the typing
alkali = ['Li','Be','Na','Mg','K','Ca','Rb','Sr','Cs','Ba']
# Used for special paddlewheel cases. 
# See Cotton et al, "Multiple bonds between metal atoms"
mmbond = {'Cr':4,'Mo':4,'V':3,'Rh':1,'Ru':2,'W':4,'Os':3,'Re':4,'Pt':1,'Tc':4,'Ir':1,'Pd':1}



if __name__ == '__main__':

    # parser for arguments
    parser = argparse.ArgumentParser(prog='AtomTyper' , description='Assigns connectivity and UFF atom types to ASE-readable files')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', help="typing mode: by file. write name of the file to be typed, with extension")
    parser.add_argument('-l', type=str, default='libraries/uff4mof.csv', help="""optional: specifies the library of parameters to fit to the structures. uff4mof : uff4mof extended library. See the file for your own custom libraries.""")
    parser.add_argument('-r', type=str, default='libraries/rappe.csv',   help="""optional: specifies the reference library of UFF parameters.""")
    args = parser.parse_args()
    #timer
    start_time = time.time()
    fileToType = args.f
    #typing proper
    rootName = '.'.join(fileToType.split('.')[:-1]) 
    structure = MolTyper(fileToType)
    structure.type_mol(library=args.l, reference_library=args.r)
    structure.structure.write(name=rootName)
    print 'Done --- time = ', time.time() - start_time, 'seconds'
