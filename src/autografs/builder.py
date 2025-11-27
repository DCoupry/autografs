"""
Builder module for AuToGraFS framework generation.

This module contains the main Autografs class responsible for generating
Metal-Organic Frameworks (MOFs) and other periodic crystalline structures
from topological blueprints and Secondary Building Units (SBUs).

Classes
-------
Autografs
    Main framework maker class for generating structures.

Examples
--------
>>> from autografs.builder import Autografs
>>> mofgen = Autografs()
>>> topology = mofgen.topologies["pcu"]
>>> mappings = mofgen.list_building_units(sieve="pcu")
>>> graph = mofgen.build(topology, mappings={k: v[0] for k, v in mappings.items()})
"""

from __future__ import annotations

import copy
import itertools
import logging
import os
import random
import time

import dill
import networkx
import numpy as np
from pymatgen.analysis.molecule_matcher import HungarianOrderMatcher
from pymatgen.core.structure import Molecule
from scipy.optimize import brute
from tqdm.auto import tqdm

import autografs.data
import autografs.utils
from autografs.structure import Fragment, Topology

logger = logging.getLogger(__name__)


class Autografs(object):
    """Framework maker class to generate periodic structures from topologies.

    AuToGraFS: Automatic Topological Generator for Framework Structures.

    This class manages libraries of Secondary Building Units (SBUs) and
    topological blueprints, providing methods to combine them into
    periodic framework structures.

    Attributes
    ----------
    topologies : dict[str, Topology]
        Dictionary of available topology blueprints, keyed by name.
    sbu : dict[str, Fragment]
        Dictionary of available Secondary Building Units, keyed by name.

    References
    ----------
    .. [1] Addicoat, M., Coupry, D. E., & Heine, T. (2014).
           The Journal of Physical Chemistry A, 118(40), 9607-14.

    Examples
    --------
    >>> mofgen = Autografs()
    >>> print(len(mofgen.topologies))  # Available topologies
    >>> print(len(mofgen.sbu))  # Available building units
    """

    def __init__(
        self,
        xyzfile: str | None = None,
        topofile: str | None = None,
    ) -> None:
        """Initialize the Autografs framework generator.

        Parameters
        ----------
        xyzfile : str or None, optional
            Path to an XYZ file containing custom Secondary Building Units.
            If provided, these SBUs will be added to the default library.
        topofile : str or None, optional
            Path to a pickle file containing custom topologies.
            If None, uses the default RCSR topology library.

        Examples
        --------
        >>> mofgen = Autografs()  # Use defaults
        >>> mofgen = Autografs(xyzfile="my_sbus.xyz")  # Add custom SBUs
        """
        super().__init__()
        # Initialize instance attributes to avoid sharing between instances
        self.topologies: dict[str, Topology] = {}
        self.sbu: dict[str, Fragment] = {}
        logger.info(f"{'*':*^78}")
        logger.info(f"*{'AuToGraFS':^76}*")
        logger.info(
            f"*{'Automatic Topological Generator for Framework Structures':^76}*"
        )
        logger.info(f"*{'Addicoat, M., Coupry, D. E., & Heine, T. (2014)':^76}*")
        logger.info(f"*{'The Journal of Physical Chemistry A, 118(40), 9607-14':^76}*")
        logger.info(f"{'*':*^78}")
        logger.info("")
        logger.info("Setting up libraries...")
        self._setup_sbu(xyzfile=xyzfile)
        self._setup_topologies(topofile=topofile)
        logger.info("")

    def build_all(
        self,
        sbu_subset: list[str] | None = None,
        topology_subset: list[str] | None = None,
        refine_cell: bool = True,
    ) -> list[networkx.Graph]:
        """
        Builds all available structures based on the SBU and Topologies libraries

        Parameters
        ----------
        sbu_subset : list[str] | None, optional
            The subset of SBU to build on, by default None
        topology_subset : list[str] | None, optional
            the subset of topologies to build on, by default None

        Returns
        -------
        list[networkx.Graph]
            the list of connected graphs produced by the building method
        """
        graphs = []
        logger.info("Building All Available Structures! This will take some time.")
        logger.info("============================================================")
        # logger.info(f"Number of cores for parrallel building = {cpu_count}")
        total_len = 0
        topologies = self.list_topologies(subset=topology_subset)
        topology_pbar = tqdm(topologies)
        for topology_name in topology_pbar:
            topology_pbar.set_description(f"Processing topology {topology_name:<5}")
            maps = []
            topology = self.topologies[topology_name].copy()
            sbu_dict = self.list_building_units(
                sieve=topology_name, verbose=False, subset=sbu_subset
            )
            if not all(sbu_dict.values()):
                continue
            for sbus in list(itertools.product(*sbu_dict.values())):
                maps.append((topology, dict(zip(sbu_dict.keys(), sbus))))
            total_len += len(maps)
            results_graphs = []
            for build_args in tqdm(maps):
                try:
                    g = self.build(*build_args, refine_cell=refine_cell, verbose=False)
                    results_graphs.append(g)
                except AssertionError:
                    # probably an unfilled slot in a topology
                    # need to be a graceful failure here
                    continue
            graphs += [g for g in results_graphs if g is not None]
        logger.info(f"\t[x] Generated a total of {len(graphs):^6} periodic graphs.")
        logger.info(f"\t[x] Rate of error: {1.0 - len(graphs)/total_len:2.2%}.")
        return graphs

    def build(
        self,
        topology: Topology,
        mappings: dict[Fragment | int, Fragment | str],
        refine_cell: bool = True,
        verbose: bool = False,
    ) -> networkx.Graph:
        """
        Generates a graph from a mapping of SBU to topology slots.

        Parameters
        ----------
        topology : Topology
            the topology to consider
        mappings : dict[Fragment | int, Fragment | str]
            the mappings to go from a slot to a compatible SBU
        verbose : bool, optional
            If True, will log additional information, by default False

        Returns
        -------
        networkx.Graph
            The connected molecular graph of the constructed structure
        """
        mappings = self._validate_mappings(topology=topology, mappings=mappings)
        if verbose:
            logger.info("Starting building process:")
            logger.info(f"\tTopology =  {topology}")
            formatted_mappings = autografs.utils.format_mappings(mappings)
            logger.info(f"\tMappings =  {formatted_mappings}")
            logger.info(f"Aligning with cell scaling...")

        def opt_fun(scales: tuple[float, float, float]) -> float:
            """Objective function for cell parameter optimization.

            Parameters
            ----------
            scales : tuple[float, float, float]
                Cell vector lengths (a, b, c) to evaluate.

            Returns
            -------
            float
                Root mean square error of fragment alignment.
            """
            this_topology = topology.copy()
            this_mapping = copy.deepcopy(mappings)
            _, rmse = self._align_all_mappings(this_topology, this_mapping, scales)
            return rmse

        # try:
        t0 = time.time()
        #  We need better initialization of search space for faster convergence
        abc_norm = sum([f.max_dummy_distance for f in mappings.values()]) / 3.0
        abc_norm = np.ones(3) * abc_norm
        if refine_cell:
            x_min = abc_norm * 0.1
            x_max = abc_norm * 2.0
            best_scales, _, _, _ = brute(
                opt_fun, ranges=list(zip(x_min, x_max)), Ns=3, full_output=True
            )
        else:
            if verbose:
                logger.info("\t[x] No cell refinement performed.")
            best_scales = abc_norm
        best_alignment, _ = self._align_all_mappings(topology, mappings, best_scales)
        if verbose:
            logger.info("\t[x] Best cell parameters:")
            logger.info(f"\t\ta = {best_scales[0]:<.1f}")
            logger.info(f"\t\tb = {best_scales[1]:<.1f}")
            logger.info(f"\t\tc = {best_scales[2]:<.1f}")
            logger.info(
                f"\t[x] Aligned {len(topology.slots)} fragments in {time.time() - t0:.1f} seconds"
            )
        graph = autografs.utils.fragments_to_networkx(
            best_alignment, cell=topology.cell.matrix
        )
        # except Exception as e:
        # logger.debug(e)
        # graph = None
        return graph

    def _validate_mappings(
        self, topology: Topology, mappings: dict[Fragment | int, Fragment | str]
    ) -> dict[int, Fragment]:
        """Validate and normalize the slot-to-fragment mappings.

        Converts string SBU names to Fragment objects and ensures all
        topology slots have corresponding mappings.

        Parameters
        ----------
        topology : Topology
            The topology blueprint being filled.
        mappings : dict[Fragment | int, Fragment | str]
            User-provided mappings from slot identifiers to SBU names or objects.

        Returns
        -------
        dict[int, Fragment]
            Normalized mappings from slot indices to Fragment objects.

        Raises
        ------
        AssertionError
            If any required slot is not present in the mappings.
        """
        for k in topology.mappings.keys():
            assert k in mappings, f"Unfilled {k} slot."
        #  make sure all strings are converted to the correct fragment
        for k, v in mappings.items():
            if isinstance(v, str):
                mappings[k] = self.sbu[v]
        #  now check and convert all keys to the slot indices
        true_mappings = {}
        for k, v in mappings.items():
            if isinstance(k, int):
                true_mappings[k] = v
            else:
                for i in topology.mappings[k]:
                    true_mappings[i] = copy.deepcopy(v)
        return true_mappings

    def _setup_sbu(self, xyzfile: str | None = None) -> None:
        """Load Secondary Building Units from XYZ files.

        Loads default SBUs from the package data directory and optionally
        adds custom SBUs from a user-provided file.

        Parameters
        ----------
        xyzfile : str or None, optional
            Path to an XYZ file containing additional SBUs to load.
            Custom SBUs with the same name as defaults will override them.
        """
        t0 = time.time()
        default_path = os.path.join(autografs.data.__path__[0], "defaults.xyz")
        sbu = autografs.utils.xyz_to_sbu(default_path)
        logger.info(
            f"\t[x] loaded {len(sbu)} default building units in {time.time() - t0:.0f} seconds."
        )
        if xyzfile is not None:
            t0 = time.time()
            added_sbu = autografs.utils.xyz_to_sbu(xyzfile)
            sbu.update(added_sbu)
            logger.info(
                f"\t[x] loaded {len(added_sbu)} custom building units in {time.time() - t0:.0f} seconds."
            )
        self.sbu = sbu
        return None

    def _setup_topologies(self, topofile: str | None = None) -> None:
        """Load topologies from a pickle file.

        Loads topology blueprints from a serialized pickle file containing
        pre-processed RCSR or custom topologies.

        Parameters
        ----------
        topofile : str or None, optional
            Path to a pickle file containing topologies.
            If None, loads from the default package data location.
        """
        t0 = time.time()
        if topofile is None:
            topofile = os.path.join(autografs.data.__path__[0], "topologies.pkl")
        with open(topofile, "rb") as topo:
            topologies = dill.load(topo)
        logger.info(
            f"\t[x] loaded {len(topologies)} topologies in {time.time() - t0:.0f} seconds."
        )
        self.topologies.update(topologies)
        return None

    def list_topologies(
        self,
        sieve: str | None = None,
        verbose: bool = False,
        subset: list[str] | None = None,
    ) -> list[str]:
        """List available topologies, optionally filtered by SBU compatibility.

        Parameters
        ----------
        sieve : str or None, optional
            Name of an SBU to filter topologies by compatibility.
            Only topologies with slots matching this SBU will be returned.
        verbose : bool, optional
            If True, logs detailed information about the filtering process.
        subset : list[str] or None, optional
            If provided, only consider topologies in this list.

        Returns
        -------
        list[str]
            Sorted list of available topology names.

        Examples
        --------
        >>> mofgen = Autografs()
        >>> all_topos = mofgen.list_topologies()
        >>> compatible = mofgen.list_topologies(sieve="Benzene_linear")
        """
        full_list = sorted(self.topologies.keys())
        if subset is not None:
            full_list = subset
        if sieve is not None:
            sieve = self.sbu[sieve]
            if verbose:
                logger.info(
                    f"filtering topologies for compatibility with {sieve} building unit..."
                )
            for topology in self.topologies.values():
                if topology.get_compatible_slots(candidate=sieve):
                    continue
                else:
                    full_list.remove(topology.name)
        if verbose:
            logger.info(f"\t[x] {len(full_list):>5} topologies available.")
        return full_list

    def list_building_units(
        self,
        sieve: str | None = None,
        verbose: bool = False,
        subset: list[str] | None = None,
    ) -> dict[str, Fragment]:
        """List available SBUs, optionally filtered by topology compatibility.

        Parameters
        ----------
        sieve : str or None, optional
            Name of a topology to filter SBUs by compatibility.
            Returns SBUs grouped by compatible slot types.
        verbose : bool, optional
            If True, logs detailed information about available SBUs per slot.
        subset : list[str] or None, optional
            If provided, only consider SBUs in this list.

        Returns
        -------
        dict[str, list[str]]
            Dictionary mapping slot identifiers to lists of compatible SBU names.
            If no sieve is provided, returns an empty dict.

        Examples
        --------
        >>> mofgen = Autografs()
        >>> sbu_dict = mofgen.list_building_units(sieve="pcu")
        >>> for slot, sbus in sbu_dict.items():
        ...     print(f"Slot {slot}: {len(sbus)} compatible SBUs")
        """
        full_list = []
        if subset is not None:
            sbus = [self.sbu[k] for k in subset]
        else:
            sbus = list(self.sbu.values())
        if sieve is not None:
            sieve = self.topologies[sieve]
            if verbose:
                logger.info(
                    f"filtering building units for compatibility with {sieve} topology..."
                )
            for sbu in sbus:
                mappings = sieve.get_compatible_slots(candidate=sbu)
                full_list += [(k, sbu.name) for k, v in mappings.items() if v]
        slot_groups = itertools.groupby(full_list, lambda x: x[0])
        building_units = {}
        for k, v in slot_groups:
            v = list(list(zip(*v))[1])
            if k not in building_units:
                building_units[k] = v
            else:
                building_units[k] += v
        out_dict = {}
        for k, v in building_units.items():
            out_dict[k] = list(set(v))
            if verbose:
                logger.info(f"\t[x] {len(out_dict[k]):>5} SBU available for slot {k}")
        # now check that all the slots from the sieves are filled
        if sieve is not None and verbose:
            for k in sieve.mappings.keys():
                if k not in out_dict:
                    logger.info(f"\t[!] {0:>5} SBU available for slot {k}")
        return out_dict

    def _align_all_mappings(
        self,
        topology: Topology,
        mappings: dict[int, Fragment],
        scales: tuple[float, float, float],
    ) -> tuple[list[Fragment], float]:
        """Align all SBU fragments to their corresponding topology slots.

        Scales the topology cell and aligns each fragment to its designated
        slot using the Hungarian algorithm for optimal atom matching.

        Parameters
        ----------
        topology : Topology
            The topology blueprint (will be modified in-place by scaling).
        mappings : dict[int, Fragment]
            Mapping from slot indices to Fragment objects.
        scales : tuple[float, float, float]
            Cell vector lengths (a, b, c) to apply.

        Returns
        -------
        tuple[list[Fragment], float]
            A tuple containing:
            - List of aligned Fragment objects.
            - Total weighted RMSE of the alignment.
        """
        all_aligned_fragments = []
        topology.scale_slots(scales)
        sum_rmse = 0.0
        for slot_index, this_fragment in mappings.items():
            slot = topology.slots[slot_index]
            slot_weight = this_fragment.max_dummy_distance / slot.max_dummy_distance
            aligned_fragment, rmse = self._align_slot(slot, this_fragment)
            all_aligned_fragments.append(aligned_fragment)
            denominator = len(slot.atoms) * len(topology.mappings[slot])
            if denominator == 0:
                logger.warning(f"Empty slot encountered at index {slot_index}, skipping score calculation")
                continue
            score = slot_weight * rmse / denominator
            sum_rmse += score
        logger.debug(f"Cell scaled with RMSE = {sum_rmse}")
        return all_aligned_fragments, sum_rmse

    def _align_slot(self, slot: Fragment, fragment: Fragment) -> tuple[Fragment, float]:
        """Align a single fragment to a topology slot.

        Uses the Hungarian algorithm to find the optimal rotation and
        translation that minimizes the RMSD between dummy atoms.

        Parameters
        ----------
        slot : Fragment
            The topology slot defining the target orientation and position.
        fragment : Fragment
            The SBU fragment to align (modified in-place).

        Returns
        -------
        tuple[Fragment, float]
            A tuple containing:
            - The aligned fragment with updated coordinates and tags.
            - RMSD of the alignment.
        """
        # m0 = slot.atoms.copy()
        m0 = slot.extract_dummies()
        m0.perturb(distance=0.01)
        m0.replace_species({"X": "H"})
        m1 = fragment.extract_dummies()
        m1.perturb(distance=0.01)
        m1.replace_species({"X": "H"})
        _, U, V, rmsd = HungarianOrderMatcher(m0).match(m1)
        new_coords = fragment.atoms.cart_coords.dot(U) + V
        fragment.atoms = Molecule(fragment.atoms.species, coords=new_coords)
        fragment_tags = np.array(
            [
                -1,
            ]
            * len(fragment.atoms)
        )
        for j, frag_tag in enumerate(fragment_tags):
            fragment.atoms[j].properties["tags"] = frag_tag
        slot_tags = slot.atoms.site_properties["tags"]
        for i, slot_tag in enumerate(slot_tags):
            tag_by_dist = []
            for j in fragment.atoms.indices_from_symbol("X"):
                d = slot.atoms[i].coords - fragment.atoms[j].coords
                tag_by_dist.append((np.linalg.norm(d), j))
            _, best_match = sorted(tag_by_dist)[0]
            fragment.atoms[best_match].properties["tags"] = slot_tag
        return fragment, rmsd


if __name__ == "__main__":
    from pprint import pprint

    molgen = Autografs()
    # pprint(molgen.list_topologies())
    # pprint(sorted(molgen.sbu.keys()))
    # sbu = 'Persulfurated_benzene_hexagonal', 'Zn_imidazolate_carboxylated_tetrahedral', 'Al_trimeric_prism', 'Benzene_pentagonal',
    #  'Benzene_pentagram', 'CuCN_trigonal_bipyramid' 'Ni_pyrazole_cubic_cluster' 'UIO66_Zr_icosahedral'
    # print(molgen.sbu['Persulfurated_benzene_hexagonal'])
    # for name, topology in molgen.topologies.items():
    #     if len(topology) > 15:
    #         continue
    #     try:
    #         maps = molgen.list_building_units(sieve=name, verbose=True)
    #         g = molgen.build(topology=topology, mappings={k: random.choice(v) for k, v in maps.items()}, verbose=True, refine_cell=True)
    #         autografs.utils.view_graph(g)
    #     except AssertionError:
    #         continue
    # pprint(topos)
    # raise
    # topos = molgen.list_topologies()
    # i = 0
    # while i < 5:
    #     name = random.choice(topos)
    #     topology = molgen.topologies[name]
    #     if len(topology) > 50:
    #         continue
    #     else:
    #         i += 1
    #         maps = molgen.list_building_units(sieve=name, verbose=True)
    #         g = molgen.build(topology=topology, mappings={k: random.choice(v) for k, v in maps.items()}, verbose=True, refine_cell=False)
    # autografs.utils.networkx_to_gulp(g, name=name, write_to_file=True)
    # autografs.utils.view_graph(g)
    # pprint(sbu_names)
    # print(list(molgen.topologies.keys()))
    # benz = molgen.sbu["Benzene_linear"].copy()
    # benz.rotate(theta=0.79)
    # hidx = benz.atoms.indices_from_symbol("H")
    # benz.functionalize(hidx[2], "nitro")
    # maps = {0: "Zn_mof5_octahedral", 1:"Bipyridine_linear", 2:"Acetylene_linear", 3:benz}
    # g = molgen.build(topology=topology, mappings=maps, verbose=True)
    # # autografs.utils.view_graph(g)

    # autografs.utils.networkx_to_gulp(g, name="zul", write_to_file=True)
    graphs = molgen.build_all(refine_cell=False)
    with open("all_generated.bin", "wb") as uit:
        dill.dump(graphs, uit)
    # autografs.utils.view_graph(graphs[0])
    # for graph in graphs:
    #     autografs.utils.view_graph(graph)
    #     break
