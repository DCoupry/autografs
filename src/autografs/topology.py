"""
Topology module for AuToGraFS periodic topology blueprints.

This module defines the Topology class used to represent periodic
topology blueprints in AuToGraFS.

Classes
-------
Topology
    Represents a periodic topology blueprint with slots for SBUs.

Examples
--------
>>> topology = mofgen.topologies["pcu"]
>>> print(f"{topology.name}: {len(topology)} slots")
>>> print(f"Cell: {topology.cell.abc}")
"""

from __future__ import annotations

import copy
import logging

import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Molecule

from autografs.fragment import Fragment

__all__ = [
    "Topology",
]

logger = logging.getLogger(__name__)


class Topology:
    """Periodic topology blueprint for framework structure generation.

    A Topology represents the periodic arrangement of slots where Secondary
    Building Units (SBUs) can be placed. Each slot defines the local
    geometry and connectivity requirements.

    Attributes
    ----------
    name : str
        Topology identifier (typically RCSR three-letter symbol).
    cell : Lattice
        Periodic cell parameters as a pymatgen Lattice object.
    slots : np.ndarray[Fragment]
        Array of Fragment objects representing topology slots.
    sizes : np.ndarray[int]
        Array of slot sizes (number of atoms per slot).
    mappings : dict[Fragment, list[int]]
        Groups equivalent slots by their Fragment type.

    Examples
    --------
    >>> topology = mofgen.topologies["pcu"]
    >>> print(f"{topology.name}: {len(topology)} slots")
    >>> print(f"Cell: {topology.cell.abc}")
    """

    def __init__(
        self,
        name: str,
        slots: list[Fragment],
        cell: np.ndarray | Lattice,
        equivalence_classes: list[int] | None = None,
        spacegroup_number: int | None = None,
        is_2d: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        name : str
            the name given to the topology (RCSR symbol in defaults)
        slots : list[Fragment]
            the list of Fragment objects describing the orientation and
            connectivity of slots in the topology. The fragments are
            stored as-is (not copied); when equivalence_classes is
            given, their ``equivalence_class`` attribute is set in
            place.
        cell : np.ndarray
            The information on periodicity in matrix form (3x3)
        equivalence_classes : list[int] or None, optional
            Crystallographic orbit id for each slot. Slots sharing an id
            are symmetry-equivalent and grouped under one slot type, and
            crystallographically distinct orbits stay distinct even when
            their local point group and size coincide. When omitted,
            slots are grouped by point group and size alone (legacy
            behavior).
        spacegroup_number : int or None, optional
            International spacegroup number of the source net, kept for
            provenance and symmetry-constrained cell optimization. For
            2D nets (is_2d True) this is the ITA plane-group number
            (1-17) instead.
        is_2d : bool, optional
            True for layer nets: the blueprint is periodic in the a-b
            plane only, and c is a padding value (a 10 A slab) that the
            cell optimizer must keep frozen.
        """
        self.name = name
        self.is_2d = bool(is_2d)
        if isinstance(cell, Lattice):
            self.cell = cell
        else:
            self.cell = Lattice(cell)
        if equivalence_classes is not None:
            if len(equivalence_classes) != len(slots):
                raise ValueError(
                    f"{len(equivalence_classes)} equivalence classes for "
                    f"{len(slots)} slots in topology {name}."
                )
            for slot, eq_class in zip(slots, equivalence_classes, strict=True):
                slot.equivalence_class = int(eq_class)
        self.spacegroup_number = spacegroup_number
        self.slots = np.array(slots, dtype=object)
        sizes = [len(fragment.atoms) for fragment in self.slots]
        self.sizes = np.array(sizes, dtype=np.int32)
        # group equivalent slots, keyed by first occurrence: iterating a
        # set here would make the key order depend on hash randomization
        # and vary between processes
        mappings: dict[Fragment, list[int]] = {}
        for i, slot_type in enumerate(slots):
            mappings.setdefault(slot_type, []).append(i)
        self.mappings = mappings

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

    def get_compatible_slots(self, candidate: Fragment) -> dict[Fragment, list[int]]:
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
        dict[Fragment, list[int]]
            A dictionary of available slot indices
        """
        available_slots: dict[Fragment, list[int]] = {}
        for slot in self.mappings:
            available_slots[slot] = []
            if slot.has_compatible_symmetry(candidate):
                available_slots[slot] += self.mappings[slot]
        return available_slots

    def scale_slots(self, scales: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
        """
        Applies in-place a scaling along cell vectors of the slots contained in
        the topology.
        TODO: rename scales to three a, b, c parameters for clarity

        Parameters
        ----------
        scales : tuple[float, float, float], optional
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
            scaled_slot.atoms = Molecule(
                slot.atoms.species,
                scaled_coords,
                site_properties=slot.atoms.site_properties,
            )
            scaled_slots.append(scaled_slot)
        self.slots = np.array(scaled_slots, dtype=object)
        self.cell = scaled_cell
