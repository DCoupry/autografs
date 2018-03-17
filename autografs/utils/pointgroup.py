#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
# Heavily adapted from the pymatgen library

__author__  = "Damien Coupry"
__credits__ = ["Prof. Matthew Addicoat","Shyue Ping Ong"]
__license__ = "MIT"
__maintainer__ = "Damien Coupry"
__version__ = '2.0.4'
__status__  = "beta"

import numpy as np
from   scipy import cluster
import itertools
from ase.visualize import view
from ase import Atom, Atoms

from autografs.utils.operations import rotation, reflection, inertia


class PointGroup(object):
    """A class to analyze the point group of a molecule. The general outline of
    the algorithm is as follows:

    1. Center the molecule around its center of mass.
    2. Compute the inertia tensor and the eigenvalues and eigenvectors.
    3. Handle the symmetry detection based on eigenvalues.

        a. Linear molecules have one zero eigenvalue. Possible symmetry
           operations are C*v or D*v
        b. Asymetric top molecules have all different eigenvalues. The
           maximum rotational symmetry in such molecules is 2
        c. Symmetric top molecules have 1 unique eigenvalue, which gives a
           unique rotation axis.  All axial point groups are possible
           except the cubic groups (T & O) and I.
        d. Spherical top molecules have all three eigenvalues equal. They
           have the rare T, O or I point groups.

    .. attribute:: schoenflies

        Schoenflies symbol of the detected point group.
    """

    def __init__(self, mol, tol, out=None):
        """
        The default settings are usually sufficient.

        Args:
            mol (Molecule): Molecule to determine point group for.
        """

        self.tol  = tol
        self.etol = self.tol /3.0
        self.mol  = mol
        # center molecule
        self.mol.positions -= self.mol.positions.mean(axis=0)
                # normalize
        norm_factor = np.amax(np.linalg.norm(self.mol.positions,axis=1,keepdims=True))
        if norm_factor<1e-6:
            norm_factor = 1.0
        self.mol.positions /= norm_factor

        self.symmops      = {"C"    :[],
                             "S"    :[],
                             "sigma":[]}
        # identity
        self.symmops["I"] = np.eye(3)
        #inversion
        inversion = - np.eye(3)
        if self.is_valid_op(inversion):
            self.symmops["-I"] = inversion
            # print("Inversion center present")
        else:
            self.symmops["-I"] = None

        self.nrot  = 0
        self.nref  = 0

        self.out = out

        self.analyze()
        
        if self.schoenflies in ["C1v", "C1h"]:
            self.schoenflies = "Cs"

    def analyze(self):
        """TODO"""
        if len(self.mol) == 1:
            self.schoenflies = "Kh"
        else:
            # Get the inertia tensor
            xyz = self.mol.get_positions()
            W   = self.mol.get_masses()
            I = inertia(xyz=xyz,W=W)
            I /= np.sum(np.linalg.norm(xyz,axis=1)*W)
            # calculate the eigenvalues
            eigvals, eigvecs = np.linalg.eigh(I)
            self.eigvecs = eigvecs.T
            self.eigvals = eigvals
            e0,e1,e2 = self.eigvals
            eig_zero = np.abs(np.product(self.eigvals)) < self.etol**3
            eig_same = (abs(e0-e1)<self.etol)&(abs(e0-e2)<self.etol)
            eig_diff = (abs(e0-e1)>self.etol)&(abs(e0-e2)>self.etol)&(abs(e1-e2)>self.etol)
            # print("Inertia tensor eigenvalues = {0:3.2f} {1:3.2f} {2:3.2f}".format(*self.eigvals),file=self.out)
            # determine which process to go through
            if eig_zero:
                # print("Linear molecule detected",file=self.out)
                self.analyze_linear()
            elif eig_same:
                # print("Spherical top molecule detected",file=self.out)
                self.analyze_spherical_top()
            elif eig_diff:
                # print("Asymmetric top molecule detected",file=self.out)
                self.analyze_asymmetric_top()
            else:
                # print("Symmetric top molecule detected",file=self.out)
                self.analyze_symmetric_top()


    def analyze_linear(self):
        """TODO"""
        inversion = - np.eye(3)
        if self.symmops["-I"] is not None:
            self.schoenflies = "D*h"
        else:
            self.schoenflies = "C*v"

    def analyze_asymmetric_top(self):
        """Handles assymetric top molecules, which cannot contain rotational
        symmetry larger than 2.
        """

        for axis in self.eigvecs:
            op = rotation(axis=axis,order=2)
            if self.is_valid_op(op):
                self.symmops["C"] += [(2,axis,op),]
                self.nrot += 1

        if   self.nrot == 0:
            # print("No rotation symmetries detected.",file=self.out)
            self.analyze_nonrotational_groups()
        elif self.nrot == 3:
            # print("Dihedral group detected.",file=self.out)
            self.analyze_dihedral_groups()
        else:
            # print("Cyclic group detected.",file=self.out)
            self.analyze_cyclic_groups()

    def analyze_symmetric_top(self):
        """
        Handles symetric top molecules which has one unique eigenvalue whose
        corresponding principal axis is a unique rotational axis.  More complex
        handling required to look for R2 axes perpendicular to this unique
        axis.
        """

        # identify unique axis
        if   abs(self.eigvals[0] - self.eigvals[1]) < self.etol:
            unique_axis = self.eigvecs[2]
        elif abs(self.eigvals[1] - self.eigvals[2]) < self.etol:
            unique_axis = self.eigvecs[0]
        else:
            unique_axis = self.eigvecs[1]
        order = self.detect_rotational_symmetry(unique_axis)
        # print("Rotation symmetries = {0}".format(order),file=self.out)
        if self.nrot > 0:
            self.has_perpendicular_C2(unique_axis)
        if self.nrot > 1:
            self.analyze_dihedral_groups()
        elif self.nrot == 1:
            self.analyze_cyclic_groups()
        else:
            self.analyze_nonrotational_groups()

    def analyze_nonrotational_groups(self):
        """
        Handles molecules with no rotational symmetry. Only possible point
        groups are C1, Cs and Ci.
        """
        self.schoenflies = "C1"
        if self.symmops["-I"] is not None:
            self.schoenflies = "Ci"
        else:
            for v in self.eigvecs:
                mirror_type = self.find_reflection_plane(v)
                if mirror_type is not None:
                    self.schoenflies = "Cs"
                    break

    def analyze_cyclic_groups(self):
        """
        Handles cyclic group molecules.
        """
        order,axis,op = max(self.symmops["C"], key=lambda v: v[0])
        self.schoenflies = "C{0}".format(order)
        mirror_type = self.find_reflection_plane(axis)
        if mirror_type == "h":
            self.schoenflies += "h"
        elif mirror_type == "v":
            self.schoenflies += "v"
        elif mirror_type is None:
            rotoref = reflection(axis).dot(rotation(axis=axis,order=2*order))
            if self.is_valid_op(rotoref):
                self.schoenflies = "S{0}".format(2*order)
                self.symmops["S"] +=  [(order,axis,rotoref),]

    def analyze_dihedral_groups(self):
        """
        Handles dihedral group molecules, i.e those with intersecting R2 axes
        and a main axis.
        """
        order,axis,op = max(self.symmops["C"], key=lambda v: v[0])
        self.schoenflies = "D{0}".format(order)
        mirror_type = self.find_reflection_plane(axis)
        if mirror_type == "h":
            self.schoenflies += "h"
        elif not mirror_type == "":
            self.schoenflies += "d"

    def find_reflection_plane(self, axis):
        """
        Looks for mirror symmetry of specified type about axis.  Possible
        types are "h" or "vd".  Horizontal (h) mirrors are perpendicular to
        the axis while vertical (v) or diagonal (d) mirrors are parallel.  v
        mirrors has atoms lying on the mirror plane while d mirrors do
        not.
        """
        symmop = None

        # First test whether the axis itself is the normal to a mirror plane.
        op = reflection(axis)
        if self.is_valid_op(op):
            symmop = ("h",axis,op)
        else:
            # Iterate through all pairs of atoms to find mirror
            for s1, s2 in itertools.combinations(self.mol, 2):
                if s1.symbol == s2.symbol:
                    normal = s1.position - s2.position
                    if normal.dot(axis) < self.tol:
                        op = reflection(normal)
                        if self.is_valid_op(op):
                            if self.nrot > 1:
                                symmop = ("d",normal,op)
                                for prev_order,prev_axis,prev_op in self.symmops["C"]:
                                    if not np.linalg.norm(prev_axis - axis) < self.tol:
                                        if np.dot(prev_axis, normal) < self.tol:
                                            symmop = ("v",normal,op)
                                            break
                            else:
                                symmop = ("v",normal,op)
                            break
        if symmop is not None:
            self.symmops["sigma"] += [symmop,]
            return symmop[0]
        else:
            return symmop

    def find_possible_equivalent_positions(self,axis=None):
        """
        Returns the smallest list of atoms with the same species and
        distance from origin AND does not lie on the specified axis.  This
        maximal set limits the possible rotational symmetry operations,
        since atoms lying on a test axis is irrelevant in testing rotational
        symmetryOperations.
        """

        from scipy.cluster.hierarchy import fclusterdata as cluster

        def not_on_axis(index):
            v = np.cross(self.mol.positions[index], axis)
            return np.linalg.norm(v) > self.tol

        valid_sets = []
        numbers  = self.mol.get_atomic_numbers()
        dists    = np.linalg.norm(self.mol.get_positions(),axis=1,keepdims=True)
        clusters = cluster(dists, self.tol, criterion='distance')
        for cval in set(clusters):
            indices = np.where(clusters==cval)[0]
            #most common specie of set only
            counts = np.bincount(numbers[indices])
            indices = indices[np.where(numbers[indices]==np.argmax(counts))[0]]
            if axis is not None:
                indices = list(filter(not_on_axis, indices))
            if len(indices)>1:
                # no symmetry for a set of 1...
                valid_sets.append(self.mol.positions[indices])

        return min(valid_sets, key=lambda s: len(s))

    def detect_rotational_symmetry(self, axis):
        """
        Determines the rotational symmetry about supplied axis.  Used only for
        symmetric top molecules which has possible rotational symmetry
        operations > 2.
        """
        min_set = self.find_possible_equivalent_positions(axis=axis)
        max_sym = len(min_set)
        for order in range(max_sym, 0, -1):
            if max_sym % order != 0:
                continue
            op = rotation(axis=axis,order=order)
            if self.is_valid_op(op,verbose=True):
                # print("Found axis with order {0}".format(order))
                self.symmops["C"] += [(order,axis,op),]
                self.nrot += 1
                return order
        return 1

    def has_perpendicular_C2(self, axis):
        """
        Checks for R2 axes perpendicular to unique axis.  For handling
        symmetric top molecules.
        """
        min_set = self.find_possible_equivalent_positions(axis=axis)
        found=False
        for s1, s2 in itertools.combinations(min_set, 2):
            test_axis = np.cross(s1 - s2, axis)
            if np.linalg.norm(test_axis) > self.tol:
                op = rotation(axis=test_axis,order=2)
                if self.is_valid_op(op):
                    self.symmops["C"] += [(2,test_axis,op),]
                    self.nrot += 1
                    found= True
        return found
        
    def analyze_spherical_top(self):
        """
        Handles Sperhical Top Molecules, which belongs to the T, O or I point
        groups.
        """
        self.find_spherical_axes()
        if self.nrot == 0:
            # print("Accidental spherical top!",file=self.out)
            self.analyze_symmetric_top()
        # get the main axis and order
        order,axis,op = max(self.symmops["C"], key=lambda v: v[0])
        if order < 3:
            # print("Accidental speherical top!",file=self.out)
            self.analyze_symmetric_top()
        elif order == 3:
            mirror_type = self.find_reflection_plane(axis)
            if mirror_type != "":
                if self.symmops["-I"] is not None:
                    self.schoenflies = "Th"
                else:
                    self.schoenflies = "Td"
            else:
                self.schoenflies = "T"
        elif order == 4:
            if self.symmops["-I"] is not None:
                self.schoenflies = "Oh"
            else:
                self.schoenflies = "O"
        elif order == 5:
            if self.symmops["-I"] is not None:
                self.schoenflies = "Ih"
            else:
                self.schoenflies = "I"

    def find_spherical_axes(self):
        """
        Looks for R5, R4, R3 and R2 axes in spherical top molecules.  Point
        group T molecules have only one unique 3-fold and one unique 2-fold
        axis. O molecules have one unique 4, 3 and 2-fold axes. I molecules
        have a unique 5-fold axis.
        """

        rot_present = {2:False,3:False,4:False,5:False}
        test_set = self.find_possible_equivalent_positions()
        for c1, c2, c3 in itertools.combinations(test_set, 3):
            for cc1, cc2 in itertools.combinations([c1, c2, c3], 2):
                if not rot_present[2]:
                    test_axis = cc1 + cc2
                    if np.linalg.norm(test_axis) > self.tol:
                        op = rotation(axis=test_axis,order=2)
                        if self.is_valid_op(op):
                            # print("Found axis with order {0}".format(2))
                            rot_present[2] = True
                            self.symmops["C"] += [(2,test_axis,op),]
                            self.nrot += 1

            test_axis = np.cross(c2 - c1, c3 - c1)
            if np.linalg.norm(test_axis) > self.tol:
                for order in (3, 4, 5):
                    if not rot_present[order]:
                        op = rotation(axis=test_axis,order=order)
                        if self.is_valid_op(op):
                            # print("Found axis with order {0}".format(order))
                            rot_present[order] = True
                            self.symmops["C"] += [(order,test_axis,op),]
                            self.nrot += 1
                            break

            if (rot_present[2]&rot_present[3]&(rot_present[4]|rot_present[5])):
                break
        return

    def is_valid_op(self, symmop,verbose=True):
        """
        Check if a particular symmetry operation is a valid symmetry operation
        for a molecule, i.e., the operation maps all atoms to another
        equivalent atom.

        Args:
            symmop (SymmOp): Symmetry operation to test.

        Returns:
            (bool): Whether SymmOp is valid for Molecule.
        """
        
        distances = []
        mol0 = self.mol.copy()
        mol1 = self.mol.copy()
        mol1.positions = mol1.positions.dot(symmop)
        workmol = mol0+mol1
        other_indices = list(range(len(mol0),len(workmol),1))
        for atom_index in range(0,len(mol0),1):
            dist = workmol.get_distances(atom_index,other_indices,mic=False)
            distances.append(np.amin(dist))
        distances = np.array(distances)
        return (distances<self.tol).all()


