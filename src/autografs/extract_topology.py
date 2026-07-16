"""Topology extraction from tetrahedral framework structures.

The fast path from an idealized zeolite crystal (T atoms bridged by
oxygen, e.g. the IZA idealized SiO2 CIFs) to a buildable library
Topology: T atoms become 4-connected node slots, bridging oxygens
become 2-connected edge-center slots, and the connection dummies sit
at the T-O midpoints - the quarter points of the T...T edge, exactly
the convention the CGD parser produces for RCSR nets, so an extracted
zeolite and a CGD-imported net are structurally identical down to the
exact identification tier. Slot extraction, point groups, and
crystallographic orbits are cgd.analyze / cgd.orbit_equivalence_classes,
shared with the CGD path rather than re-implemented.

>>> from autografs.extract_topology import topology_from_tetrahedral
>>> topology = topology_from_tetrahedral(Structure.from_file("FAU.cif"), "FAU")
>>> identify_net(topology_quotient_edges(topology), mofgen.topologies)
['fau']

Interrupted frameworks (terminal OH/F on a T site, IZA dash codes) are
rejected: a T atom with fewer than four bridges has no 4-c vertex, and
the extracted net would not be the framework type. General MOF
topology extraction (from arbitrary deconstructions) is a separate,
harder problem and deliberately out of scope here.
"""

from __future__ import annotations

import logging

import numpy as np
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from autografs.cgd import analyze, orbit_equivalence_classes
from autografs.exceptions import TopologyExtractionError
from autografs.topology import Topology

__all__ = ["topology_from_tetrahedral"]

logger = logging.getLogger(__name__)

# a T-O bond in tetrahedral frameworks is 1.5-1.8 A (Si/Al/P/Ge...);
# the next-nearest contact (O-O at ~2.6 A) is comfortably beyond
T_O_CUTOFF = 2.3


def topology_from_tetrahedral(
    structure: Structure, name: str, cutoff: float = T_O_CUTOFF
) -> Topology:
    """Extract a buildable Topology from a tetrahedral framework.

    Parameters
    ----------
    structure : Structure
        The idealized crystal: T atoms (any non-oxygen species) and
        bridging oxygens, nothing else. Cell and coordinates are used
        as given - IZA idealized CIFs are already maximum-symmetry
        embeddings.
    name : str
        Name of the resulting topology (e.g. the framework type code).
    cutoff : float, optional
        T-O bond cutoff in Angstrom.

    Returns
    -------
    Topology
        Node slots on the T atoms, edge-center slots on the oxygens,
        shared dummies at the T-O midpoints, orbits from spglib.

    Raises
    ------
    TopologyExtractionError
        For non-tetrahedral input: an oxygen not bridging exactly two
        T atoms, or a T atom without exactly four bridges (interrupted
        frameworks, dash-coded in the IZA nomenclature).
    """
    structure = structure.copy()
    structure.remove_oxidation_states()
    t_indices = [i for i, site in enumerate(structure) if site.specie.symbol != "O"]
    o_indices = [i for i, site in enumerate(structure) if site.specie.symbol == "O"]
    if not t_indices or not o_indices:
        raise TopologyExtractionError(
            f"{name}: a tetrahedral framework needs T atoms and bridging "
            f"oxygens; got {len(t_indices)} T and {len(o_indices)} O sites."
        )
    t_set = set(t_indices)

    # every oxygen must bridge exactly two T atoms; collect the bridge
    # geometry (the neighbor coords carry the correct periodic image).
    # Species follow the CGD convention: centers encode their
    # coordination as the atomic number (Z=4 nodes, Z=2 edge centers),
    # dummies are X at the T-O midpoints - the quarter points of the
    # T...T edge, exactly where the CGD parser puts them
    node_species = get_el_sp(4)
    edge_species = get_el_sp(2)
    dummy_species = get_el_sp("X")
    species: list = [node_species] * len(t_indices)
    coords: list[np.ndarray] = [structure[i].coords for i in t_indices]
    t_bridges = dict.fromkeys(t_indices, 0)
    for o_index in o_indices:
        site = structure[o_index]
        neighbors = [
            neighbor
            for neighbor in structure.get_neighbors(site, r=cutoff)
            if neighbor.index in t_set
        ]
        if len(neighbors) != 2:
            raise TopologyExtractionError(
                f"{name}: an oxygen bridges {len(neighbors)} T atom(s) "
                "instead of two - a terminal or over-coordinated oxygen "
                "(interrupted or non-tetrahedral framework)."
            )
        species.append(edge_species)
        coords.append(site.coords)
        for neighbor in neighbors:
            t_bridges[neighbor.index] += 1
            species.append(dummy_species)
            coords.append((site.coords + neighbor.coords) / 2.0)
    under = {i: n for i, n in t_bridges.items() if n != 4}
    if under:
        example = structure[min(under)].specie.symbol
        raise TopologyExtractionError(
            f"{name}: {len(under)} T atom(s) (e.g. {example}) do not carry "
            "exactly four oxygen bridges - an interrupted or "
            "non-tetrahedral framework, which has no 4-coordinated net."
        )

    net = Structure(
        structure.lattice,
        species,
        [structure.lattice.get_fractional_coords(c) for c in coords],
    )
    fragments = analyze(net)
    # center indices inside net: all T first, then one He per oxygen
    # every third site (each O appended as He, X, X)
    n_t = len(t_indices)
    centers = list(range(n_t)) + [n_t + 3 * k for k in range(len(o_indices))]
    equivalence_classes = orbit_equivalence_classes(net, centers)
    try:
        spacegroup = SpacegroupAnalyzer(
            structure, symprec=1e-3
        ).get_space_group_number()
    except Exception:  # noqa: BLE001 - spglib failure is metadata loss only
        spacegroup = None
    logger.info(
        f"\t[x] extracted {name}: {len(t_indices)} T slots, "
        f"{len(o_indices)} edge centers, spacegroup {spacegroup}."
    )
    return Topology(
        name=name,
        slots=fragments,
        cell=net.lattice,
        equivalence_classes=equivalence_classes or None,
        spacegroup_number=spacegroup,
        is_2d=False,
    )
