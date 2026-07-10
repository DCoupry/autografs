"""
Fragment module for AuToGraFS molecular data structures.

This module defines the Fragment class used to represent molecular
fragments (Secondary Building Units) in AuToGraFS.

Classes
-------
Fragment
    Represents a molecular fragment with symmetry information and dummy atoms.

Examples
--------
>>> from autografs.fragment import Fragment
>>> from pymatgen.core.structure import Molecule
>>> from pymatgen.symmetry.analyzer import PointGroupAnalyzer
>>> mol = Molecule(["C", "X", "X"], [[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
>>> pg = PointGroupAnalyzer(mol)
>>> frag = Fragment(atoms=mol, symmetry=pg, name="linear_carbon")
"""

from __future__ import annotations

import copy
import functools
import logging
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.core.structure import FunctionalGroups, Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from scipy.spatial.distance import pdist

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Tolerance for point group symmetry detection on dummy arrangements
SYMMETRY_TOLERANCE = 0.1

# Default directional-RMSD threshold for slot/SBU compatibility.
# Deliberately permissive: the sieve lists what is worth attempting,
# and build(max_rmsd=...) / build(min_distance=...) are the strict
# acceptance gates. 0.35 accepts distorted stars (profiling the
# library showed nearly all real vertex figures fall within 0.35 of
# an existing SBU, while genuinely different figures - square vs
# tetrahedral - score ~0.6); raising it from 0.25 moved topology
# coverage from 90.2% to 96.5%. Alignment places SBUs by their own
# arm directions, so a permissive sieve costs distortion only where
# the SBU truly differs from the slot - screen with the build gates
# and clean up with Framework.relax().
COMPATIBILITY_MAX_RMSD = 0.35


@functools.cache
def _match_rmsd_cached(this_bytes: bytes, that_bytes: bytes, size: int) -> float:
    """Memoized directional match RMSD between two arm-unit sets.

    Keyed on rounded coordinate bytes: the same star shapes recur
    across the topology library (every Oh 6-c vertex, every linear
    edge...), so the full matcher runs once per distinct shape pair.
    """
    # local import: alignment imports Fragment for construction, so
    # fragment cannot import it at module level
    from autografs.alignment import match_directions

    this_units = np.frombuffer(this_bytes, dtype=float).reshape(size, 3)
    that_units = np.frombuffer(that_bytes, dtype=float).reshape(size, 3)
    _, _, rmsd = match_directions(this_units, that_units)
    return rmsd


def analyze_dummy_pointgroup(
    atoms: Molecule, tolerance: float = SYMMETRY_TOLERANCE
) -> PointGroupAnalyzer:
    """Analyze the point group of a fragment's dummy-atom arrangement.

    Builds a helper molecule containing only the dummy positions
    (as hydrogens, since dummies have no mass and break symmetrization)
    and runs pymatgen's point group analysis on it.

    Parameters
    ----------
    atoms : Molecule
        A pymatgen Molecule containing dummy atoms ("X").
    tolerance : float, optional
        Distance tolerance for symmetry detection.

    Returns
    -------
    PointGroupAnalyzer
        Analyzer for the dummy arrangement only.
    """
    dummies_idx = atoms.indices_from_symbol("X")
    symm_mol = Molecule(
        ["H"] * len(dummies_idx),
        [atoms[idx].coords for idx in dummies_idx],
        charge=len(dummies_idx),
    )
    return PointGroupAnalyzer(symm_mol, tolerance=tolerance)


class Fragment:
    """Molecular fragment with symmetry information for topology mapping.

    A Fragment represents a Secondary Building Unit (SBU) or topology slot,
    containing atomic coordinates and dummy atoms ("X") that define
    connection points.

    Attributes
    ----------
    atoms : Molecule
        A pymatgen Molecule object containing the atomic structure.
        Dummy atoms are represented by the symbol "X".
    pointgroup : str
        Schoenflies symbol of the dummy arrangement's point group.
    name : str
        Human-readable identifier for this fragment.
    equivalence_class : int or None
        Crystallographic orbit id for topology slots: slots in the same
        orbit share one id. None for SBUs and legacy topologies.

    Examples
    --------
    >>> from pymatgen.core.structure import Molecule
    >>> mol = Molecule(["C", "X", "X"], [[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    >>> frag = Fragment(atoms=mol, name="linear_carbon")
    >>> print(frag)  # D*h 2
    """

    def __init__(
        self,
        atoms: Molecule,
        symmetry: PointGroupAnalyzer | None = None,
        name: str = "",
        pointgroup: str | None = None,
        equivalence_class: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        atoms : Molecule
            A pymatgen Molecule object, complete with dummies
        symmetry : PointGroupAnalyzer or None, optional
            A pymatgen PointGroupAnalyzer containing the symmetry information
            for the dummy connectivity only. When omitted, it is computed
            lazily from the dummies on first access.
        name : str, optional
            The name of this fragment, by default ''
        pointgroup : str or None, optional
            Schoenflies symbol of the dummy arrangement. Providing it
            (e.g. from a serialized topology) avoids running the point
            group analysis. Derived from `symmetry` when that is given.
        equivalence_class : int or None, optional
            Crystallographic orbit id for topology slots.
        """
        self._symmetry = symmetry
        self._pointgroup: str | None = (
            symmetry.sch_symbol if symmetry is not None else pointgroup
        )
        self.atoms = atoms
        self.name = name
        self.equivalence_class = equivalence_class

    @property
    def symmetry(self) -> PointGroupAnalyzer:
        """Point group analyzer of the dummy arrangement, computed lazily."""
        if self._symmetry is None:
            self._symmetry = analyze_dummy_pointgroup(self.atoms)
        return self._symmetry

    @symmetry.setter
    def symmetry(self, value: PointGroupAnalyzer) -> None:
        self._symmetry = value
        self._pointgroup = value.sch_symbol

    @property
    def pointgroup(self) -> str:
        """Schoenflies symbol of the dummy arrangement's point group."""
        if self._pointgroup is None:
            self._pointgroup = self.symmetry.sch_symbol
        return self._pointgroup

    def __str__(self) -> str:
        return f"{self.pointgroup} {len(self.atoms.indices_from_symbol('X'))}"

    def __repr__(self) -> str:
        return f"{self.name} : {self.__str__()} valent"

    def __eq__(self, other: object) -> bool:
        # equality means interchangeable as a slot type: same point
        # group, same size, and same crystallographic orbit (when known)
        if not isinstance(other, Fragment):
            return NotImplemented
        return (
            self.pointgroup == other.pointgroup
            and len(self.atoms) == len(other.atoms)
            and self.equivalence_class == other.equivalence_class
        )

    def __hash__(self):
        return hash((self.pointgroup, len(self.atoms), self.equivalence_class))

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
        dummy_sites = [site for site in self.atoms if site.specie.symbol == "X"]
        return Molecule.from_sites(dummy_sites)

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
        if len(dummies) < 2:
            return 0.0
        return float(pdist(dummies).max())

    @functools.cached_property
    def arm_units(self) -> np.ndarray:
        """Unit vectors from the dummy centroid to each dummy.

        These directions are the geometric identity of the fragment's
        connectivity; compatibility and alignment both work on them.
        """
        dummy_idx = list(self.atoms.indices_from_symbol("X"))
        dummies = np.asarray(self.atoms.cart_coords)[dummy_idx]
        arms = dummies - dummies.mean(axis=0)
        norms = np.linalg.norm(arms, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return np.asarray(arms / norms)

    @functools.cached_property
    def _shape_signature(self) -> np.ndarray:
        """Sorted pairwise dots of arm units: rotation and permutation
        invariant, used to reject mismatched shapes cheaply."""
        units = self.arm_units
        i, j = np.triu_indices(len(units), k=1)
        return np.sort(np.einsum("ij,ij->i", units[i], units[j]))

    def has_compatible_symmetry(
        self, other: Fragment, max_rmsd: float = COMPATIBILITY_MAX_RMSD
    ) -> bool:
        """
        Checks whether another fragment can occupy this fragment's
        position, by geometrically matching the two sets of unit arm
        vectors (see autografs.alignment.match_directions).

        Compatibility requires an equal number of dummies; fragments
        with one or two dummies are then always compatible (two arms
        seen from their own centroid are always antiparallel). Larger
        fragments are compatible when their arm directions match under
        some proper rotation to within max_rmsd. Point group symbols
        are no longer consulted: they over-rejected near-symmetric
        SBUs, under-rejected same-symbol shape mismatches, and made
        low-symmetry (C1) vertices unusable.

        Parameters
        ----------
        other : Fragment
            The fragment being compared
        max_rmsd : float, optional
            Directional RMSD threshold (dimensionless; 0 = identical
            shapes). The default is permissive - the sieve lists what
            is worth trying; build(max_rmsd=...) is the strict gate.

        Returns
        -------
        bool
            True if other is considered compatible with self
        """
        this_units = self.arm_units
        that_units = other.arm_units
        if len(this_units) != len(that_units):
            return False
        if len(this_units) <= 2:
            return True
        # cheap rotation/permutation-invariant prefilter: grossly
        # different pairwise-angle multisets cannot match
        gap = float(np.abs(self._shape_signature - other._shape_signature).max())
        if gap > 4.0 * max_rmsd:
            return False
        return (
            _match_rmsd_cached(
                np.round(this_units, 6).tobytes(),
                np.round(that_units, 6).tobytes(),
                len(this_units),
            )
            <= max_rmsd
        )

    def _clear_geometry_caches(self) -> None:
        """Clear cached geometry after coordinates change."""
        for name in ("max_dummy_distance", "arm_units", "_shape_signature"):
            if name in self.__dict__:
                del self.__dict__[name]

    def rotate(self, theta: float) -> None:
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
            self.atoms.rotate_sites(
                indices=sites, theta=theta, axis=axis, anchor=anchor
            )
            self._clear_geometry_caches()

    def flip(self) -> None:
        """
        Flips in-place all the atoms in the Fragment.atoms object using a reflection
        plane symmetry operation from the Fragment.symmetry object.

        The method finds a reflection plane (mirror symmetry) from the point group
        symmetry operations and applies it to all atomic coordinates.

        Raises
        ------
        ValueError
            If no reflection plane is found in the symmetry operations.
        """
        # Get symmetry operations from the PointGroupAnalyzer
        for op in self.symmetry.symmops:
            rot_matrix = op.rotation_matrix
            # A reflection has det = -1 and trace can vary
            # For a pure reflection: det(R) = -1
            if np.isclose(np.linalg.det(rot_matrix), -1.0):
                # This is a reflection or improper rotation
                # Apply the symmetry operation to all coordinates
                new_coords = op.operate_multi(self.atoms.cart_coords)
                self.atoms = Molecule(
                    self.atoms.species,
                    new_coords,
                    site_properties=self.atoms.site_properties,
                )
                self._clear_geometry_caches()
                return
        raise ValueError("No reflection plane found in symmetry operations")

    def functionalize(self, index: int, functional_group: str) -> None:
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
        self._clear_geometry_caches()
