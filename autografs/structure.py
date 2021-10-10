"""
Module docstrings are similar to class docstrings. Instead of classes and class methods being documented,
it’s now the module and any functions found within. Module docstrings are placed at the top of the file
even before any imports. Module docstrings should include the following:

A brief description of the module and its purpose
A list of any classes, exception, functions, and any other objects exported by the module
"""
from __future__ import annotations

import copy
import functools
import logging
import warnings
from typing import Dict, List, Tuple, Union

import numpy
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import FunctionalGroups, Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")



class Fragment(object):
    """
    [description]

    Attributes
    ----------

    Methods
    -------
    has_compatible_symmetry(other)
        checks symmetry compatibilities with another fragment
    """
    def __init__(self,
                 atoms: Molecule,
                 symmetry: PointGroupAnalyzer,
                 name: str = ""
                 ) -> None:
        """
        Parameters
        ----------
        atoms : Molecule
            A pymatgen Molecule object, complete with dummies
        symmetry : PointGroupAnalyzer
            A pymatgen PointGroupAnalyzer containing the symmetry information
            for the dummy connectivity only
        name : str, optional
            The name of this fragment, by default ''
        """
        self.symmetry = symmetry
        self.atoms = atoms
        self.name = name
        return None

    def __str__(self) -> str:
        return	f"{self.symmetry.sch_symbol} {len(self.atoms.indices_from_symbol('X'))}"

    def __repr__(self) -> str:
        return f"{self.name} : {self.__str__()} valent"

    def __eq__(self,
               other: Fragment
               ) -> bool:
        # for now only considers symmetries
        symm = (self.symmetry.sch_symbol == other.symmetry.sch_symbol)
        size = (len(self.atoms) == len(other.atoms))
        return (symm and size)

    def __ne__(self,
               other: Fragment
               ) -> bool:
        return not self.__eq__(other)

    def __hash__(self):
        return hash(f"SYMM={self.symmetry.sch_symbol}/NAT={len(self.atoms)}")

    def copy(self) -> Fragment:
        """
        Provides a deep copy of the starting object

        Returns
        -------
        Fragment
            the copy of the starting object
        """
        return copy.deepcopy(self)

    def extract_dummies(self) -> Molecule:
        """
        Creates and returns a pymatgen Molecule containing only
        the dummies from the original Fragment.atoms object

        Returns
        -------
        Molecule
            The extracted dummy atoms
        """
        dummies_idx = self.atoms.indices_from_symbol("X")
        dummies = Molecule.from_sites([self.atoms[i] for i in dummies_idx])
        return dummies

    @functools.cached_property
    def max_dummy_distance(self) -> float:
        """
        Returns the maximum distance between any two dummies in the
        Fragment.atoms object, providing a sense of the size of the fragment

        Returns
        -------
        float
            The maximum distance between dummies
        """
        dummies = self.extract_dummies().cart_coords
        dist = 0.0
        for i in range(len(dummies)):
            c_i = dummies[i]
            for j in range(i, len(dummies)):
                c_j = dummies[j]
                dist = max(dist, numpy.linalg.norm(c_i - c_j))
        return dist

    def has_compatible_symmetry(self,
                                other: Fragment
                                ) -> bool:
        """
        Verifies that the symmetry elements of another fragment are part of
        the symmetry elements of the present one. This is the only condition
        for compatibility in Autografs, and is sufficient to ensure:
            * the equality of connectivity
            * that square fragments are compatible with rectangular slots...

        Parameters
        ----------
        other : Fragment
            The fragment being compared

        Returns
        -------
        bool
            True, if all symmmetries of self are present in other
        """
        try:
            this_size = len(self.extract_dummies())
            that_size = len(other.extract_dummies())
            if not this_size == that_size:
                return False
            elif this_size <= 3:
                return True
            if self.symmetry.sch_symbol == other.symmetry.sch_symbol:
                return True
            return self.symmetry.is_subgroup(other.symmetry)
        except Exception:
            return False

    def rotate(self,
               theta: float
               ) -> None:
        """
        Rotates in place the atoms in the Fragment.atoms object around the
        axis provided by dummies by an angle of theta radians. This method will
        fail if the number of dummies is not 2.

        Parameters
        ----------
        theta : float
            angle of rotation in radians
        """
        dummies = self.extract_dummies()
        if len(dummies) == 2:
            sites = list(range(len(self.atoms)))
            axis = dummies.cart_coords[0] - dummies.cart_coords[1]
            anchor = dummies.cart_coords.mean(axis=0)
            self.atoms.rotate_sites(indices=sites, theta=theta, axis=axis, anchor=anchor)
        return None

    def flip(self) -> None:
        """
        Flips in-place all the atoms in the Fragment.atoms object using a reflection
        plane symmetry operation from the Fragment.symmetry object

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("Flipping is not yet implemented")

    def functionalize(self,
                      index: int,
                      functional_group: str
                      ) -> None:
        """
        replaces the atom at index with the functional group provided.
        the available functional groups are listed in the dictionary object
        pymatge.core.structure.FunctionalGroups.

        Parameters
        ----------
        index : int
            The index of the atom to substitute
        functional_group : str
            The name of the substitution
        """
        dummies_idx = self.atoms.indices_from_symbol("X")
        self.atoms.replace_species({"X": "H"})
        fg = FunctionalGroups[functional_group]
        # the substitution deletes the atom at index, then appends
        # then appends the functional group. dummies need to be reindexed.
        self.atoms.substitute(index=index, func_group=fg, bond_order=1)
        for idx in dummies_idx:
            if index < idx:
                idx -= 1
            self.atoms[idx].species = "X"
        return None



class Topology(object):
    """
    """
    def __init__(self,
                 name: str,
                 slots: List[Fragment],
                 cell: Union[numpy.ndarray, Lattice]
                 ) -> None:
        """
        Parameters
        ----------
        name : str
            the name given to the topology (RCSR symbol in defaults)
        slots : List[Fragment]
            the list of Fragment objects describing the orientation and
            connectivity of slots in the topology.
        cell : numpy.ndarray
            The information on periodicity in matrix form (3x3)
        """
        self.name = name
        if isinstance(cell, Lattice):
            self.cell = cell
        else:
            self.cell = Lattice(cell)
        self.slots = numpy.array(slots, dtype=object)
        sizes = [len(fragment.atoms) for fragment in self.slots]
        self.sizes = numpy.array(sizes, dtype=numpy.int8)
        mappings = {}
        for slot_type in set(slots):
            mappings[slot_type] = [i for i, s in enumerate(slots) if s == slot_type]
        self.mappings = mappings
        return None

    def __len__(self):
        return len(self.slots)

    def __repr__(self):
        return self.name

    def copy(self) -> Topology:
        """
        Provides a deep copy of the starting object

        Returns
        -------
        Topology
            the copy of the starting object
        """
        return copy.deepcopy(self)

    def get_compatible_slots(self,
                             candidate: Fragment
                             ) -> Dict[Fragment, List[int]]:
        """
        Returns a dictionary of the slot indices available for a candidate
        Fragment object, taking into account the symmetry elements common to
        it and the slots.

        Parameters
        ----------
        candidate : Fragment
            the query Fragment with which to test compatibility

        Returns
        -------
        Dict[Fragment, List[int]]
            A ictionary of available slot indices
        """
        available_slots = {}
        for slot in self.mappings:
            available_slots[slot] = []
            if slot.has_compatible_symmetry(candidate):
                available_slots[slot] += self.mappings[slot]
        return available_slots

    def scale_slots(self,
                    scales: Tuple[float, float, float] = (1.0, 1.0, 1.0)
                    ) -> None:
        """
        Applies in-place a scaling along cell vectors of the slots contines in
        the topology.
        TODO: rename scales to three a, b, c parameters for clarity

        Parameters
        ----------
        scales : Tuple[float, float, float], optional
            the cell vector lengths to apply, by default (1.0, 1.0, 1.0)
        """
        alpha, beta, gamma = self.cell.angles
        a, b, c = scales
        scaled_cell = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        scaled_slots = []
        for slot in self.slots:
            scaled_slot = copy.deepcopy(slot)
            fract_coords = self.cell.get_fractional_coords(slot.atoms.cart_coords)
            scaled_coords = scaled_cell.get_cartesian_coords(fract_coords)
            scaled_slot.atoms = Molecule(slot.atoms.species, scaled_coords, site_properties=slot.atoms.site_properties)
            scaled_slots.append(scaled_slot)
        self.slots = scaled_slots
        self.cell = scaled_cell
        return None

