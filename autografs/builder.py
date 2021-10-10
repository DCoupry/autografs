"""
Module docstrings are similar to class docstrings. Instead of classes and class methods being documented,
it’s now the module and any functions found within. Module docstrings are placed at the top of the file
even before any imports. Module docstrings should include the following:

A brief description of the module and its purpose
A list of any classes, exception, functions, and any other objects exported by the module
"""
import copy
import itertools
import logging
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple, Union

import dill
import joblib
import networkx
import numpy
import random
from pymatgen.analysis.molecule_matcher import HungarianOrderMatcher
from pymatgen.core.structure import Molecule
from scipy.optimize import minimize, brute
from tqdm.auto import tqdm

import autografs.data
import autografs.utils
from autografs.structure import Fragment, Topology

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")



class Autografs(object):
    """Framework maker class to generate ASE Atoms objects from topologies.

    AuToGraFS: Automatic Topological Generator for Framework Structures.
    Addicoat, M., Coupry, D. E., & Heine, T. (2014).
    The Journal of Physical Chemistry. A, 118(40), 9607–14.

    Attributes
    ----------

    Methods
    -------

    """
    topologies : Dict[str, Topology] = {}
    sbu : Dict[str, Fragment] = {}

    def __init__(self,
                 xyzfile: Optional[str] = None,
                 topofile: Optional[str] = None,
                 ) -> None:
        """
        Parameters
        ----------
        xyzfile : Optional[str], optional
            [description], by default None
        """
        super().__init__()
        logger.info(f"{'*':*^78}")
        logger.info(f"*{'AuToGraFS':^76}*")
        logger.info(f"*{'Automatic Topological Generator for Framework Structures':^76}*")
        logger.info(f"*{'Addicoat, M., Coupry, D. E., & Heine, T. (2014)':^76}*")
        logger.info(f"*{'The Journal of Physical Chemistry A, 118(40), 9607-14':^76}*")
        logger.info(f"{'*':*^78}")
        logger.info("")
        logger.info("Setting up libraries...")
        self._setup_sbu(xyzfile=xyzfile)
        self._setup_topologies(topofile=topofile)
        logger.info("")

    def build_all(self,
                  sbu_subset: Optional[List[str]] = None,
                  topology_subset: Optional[List[str]] = None,
                  refine_cell: bool = True,
                  ) -> List[networkx.Graph]:
        """
        Builds all available structures based on the SBU and Topologies libraries

        Parameters
        ----------
        sbu_subset : Optional[List[str]], optional
            The subset of SBU to build on, by default None
        topology_subset : Optional[List[str]], optional
            the subset of topologies to build on, by default None

        Returns
        -------
        List[networkx.Graph]
            the list of connected graphs produced by the building method
        """
        graphs = []
        logger.info("Building All Available Structures! This will take some time.")
        logger.info("============================================================")
        # logger.info(f"Number of cores for parrallel building = {cpu_count}")
        total_len = 0
        topologies = self.list_topologies(subset=topology_subset)
        for topology_name in tqdm(topologies, desc="topologies"):
            maps = []
            topology = self.topologies[topology_name].copy()
            sbu_dict = self.list_building_units(sieve=topology_name, verbose=False, subset=sbu_subset)
            if not all(sbu_dict.values()): continue
            for sbus in list(itertools.product(*sbu_dict.values())):
                maps.append((topology, dict(zip(sbu_dict.keys(), sbus))))
            total_len += len(maps)
            results_graphs = [self.build(*args, refine_cell=refine_cell, verbose=False) for args in tqdm(maps)]
            graphs += [g for g in results_graphs if g is not None]
        logger.info(f"\t[x] Generated a total of {len(graphs):^6} periodic graphs.")
        logger.info(f"\t[x] Rate of error: {1.0 - len(graphs)/total_len:2.2%}.")
        return graphs

    def build(self,
              topology: Topology,
              mappings: Dict[Union[Fragment, int], Union[Fragment, str]],
              refine_cell: bool = True,
              verbose: bool = False
              ) -> networkx.Graph:
        """
        Generates a graph from a mapping of SBU to topology slots.

        Parameters
        ----------
        topology : Topology
            the topology to consider
        mappings : Dict[Union[Fragment, int], Union[Fragment, str]]
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
        def opt_fun(scales: Tuple[float, float, float]) -> float:
            """
            [summary]

            Parameters
            ----------
            scales : Tuple[float, float, float]
                [description]

            Returns
            -------
            float
                [description]
            """
            this_topology = topology.copy()
            this_mapping = copy.deepcopy(mappings)
            _, rmse = self._align_all_mappings(this_topology, this_mapping, scales)
            return rmse

        # try:
        t0 = time.time()
        #  We need better initialization of search space for faster convergence
        abc_norm = sum([f.max_dummy_distance for f  in mappings.values()]) / 3.0
        abc_norm = numpy.ones(3) * abc_norm
        if refine_cell:
            x_min = abc_norm * 0.1
            x_max = abc_norm * 2.0
            best_scales, _, _, _ = brute(opt_fun, ranges=list(zip(x_min, x_max)), Ns=3, full_output=True)
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
            logger.info(f"\t[x] Aligned {len(topology.slots)} fragments in {time.time() - t0:.1f} seconds")
        graph = autografs.utils.fragments_to_networkx(best_alignment, cell=topology.cell.matrix)
        # except Exception as e:
            # logger.debug(e)
            # graph = None
        return graph

    def _validate_mappings(self,
                           topology: Topology,
                           mappings: Dict[Union[Fragment, int], Union[Fragment, str]]
                           ) -> Dict[int, Fragment]:
        """
        Returns a standard slot index to fragment object mappin

        Parameters
        ----------
        topology : Topology
            [description]
        mappings : Dict[Union[Fragment, int], Union[Fragment, str]]
            [description]

        Returns
        -------
        Dict[int, Fragment]
            [description]
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

    def _setup_sbu(self,
                   xyzfile: Optional[str] = None
                   ) -> None:
        """
        TODO: make that a class property

        Parameters
        ----------
        xyzfile : Optional[str], optional
            [description], by default None
        """
        t0 = time.time()
        default_path = os.path.join(autografs.data.__path__[0], "defaults.xyz")
        sbu = autografs.utils.xyz_to_sbu(default_path)
        logger.info(f"\t[x] loaded {len(sbu)} default building units in {time.time() - t0:.0f} seconds.")
        if xyzfile is not None:
            t0 = time.time()
            added_sbu = autografs.utils.xyz_to_sbu(xyzfile)
            sbu.update(added_sbu)
            logger.info(f"\t[x] loaded {len(added_sbu)} custom building units in {time.time() - t0:.0f} seconds.")
        self.sbu = sbu
        return None

    def _setup_topologies(self,
                          topofile: Optional[str]  = None) -> None:
        """
        TODO: make that a class property

        Parameters
        ----------
        topofile : Optional[str], optional
            [description], by default None
        """
        t0 = time.time()
        if topofile is None:
            topofile = os.path.join(autografs.data.__path__[0], "topologies.pkl")
        with open(topofile, 'rb') as topo:
            topologies = dill.load(topo)
        logger.info(f"\t[x] loaded {len(topologies)} topologies in {time.time() - t0:.0f} seconds.")
        self.topologies.update(topologies)
        return None

    def list_topologies(self,
                        sieve: Optional[str] = None,
                        verbose: bool = False,
                        subset: Optional[List[str]] = None,
                        ) -> List[str]:
        """
        [summary]

        Parameters
        ----------
        sieve : Optional[str], optional
            [description], by default None
        verbose : bool, optional
            [description], by default False
        subset : Optional[List[str]], optional
            [description], by default None

        Returns
        -------
        List[str]
            [description]
        """
        full_list = sorted(self.topologies.keys())
        if subset is not None:
            full_list = subset
        if sieve is not None:
            sieve = self.sbu[sieve]
            if verbose:
                logger.info(f"filtering topologies for compatibility with {sieve} building unit...")
            for topology in self.topologies.values():
                if topology.get_compatible_slots(candidate=sieve):
                    continue
                else:
                    full_list.remove(topology.name)
        if verbose:
            logger.info(f"\t[x] {len(full_list):>5} topologies available.")
        return full_list

    def list_building_units(self,
                            sieve: Optional[str] = None,
                            verbose: bool = False,
                            subset: Optional[List[str]] = None,
                            ) -> Dict[str, Fragment]:
        """
        [summary]

        Parameters
        ----------
        sieve : Optional[str], optional
            [description], by default None
        verbose : bool, optional
            [description], by default False
        subset : [type], optional
            [description], by default Optional[List[str]]

        Returns
        -------
        Dict[str, Fragment]
            [description]
        """
        full_list = []
        if subset is not None:
            sbus = [self.sbu[k] for k in subset]
        else:
            sbus = list(self.sbu.values())
        if sieve is not None:
            sieve = self.topologies[sieve]
            if verbose:
                logger.info(f"filtering building units for compatibility with {sieve} topology...")
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

    def _align_all_mappings(self,
                            topology: Topology,
                            mappings: Dict[int, Fragment],
                            scales: Tuple[float, float, float]
                            ):
        """
        [summary]

        Parameters
        ----------
        topology : Topology
            [description]
        mappings : Dict[int, Fragment]
            [description]
        scales : Tuple[float, float, float]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        all_aligned_fragments = []
        topology.scale_slots(scales)
        sum_rmse = 0.0
        for slot_index, this_fragment in mappings.items():
            slot = topology.slots[slot_index]
            slot_weight = this_fragment.max_dummy_distance / slot.max_dummy_distance
            aligned_fragment, rmse = self._align_slot(slot, this_fragment)
            all_aligned_fragments.append(aligned_fragment)
            score = slot_weight * rmse / (len(slot.atoms) * len(topology.mappings[slot]))
            sum_rmse += score
        logger.debug(f"Cell scaled with RMSE = {sum_rmse}")
        return all_aligned_fragments, sum_rmse

    def _align_slot(self,
                    slot: Fragment,
                    fragment: Fragment
                    ) -> Tuple[Fragment, float]:
        """
        [summary]

        Parameters
        ----------
        slot : Fragment
            [description]
        fragment : Fragment
            [description]

        Returns
        -------
        Tuple[Fragment, float]
            [description]
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
        fragment_tags = numpy.array([-1, ] * len(fragment.atoms))
        for j, frag_tag in enumerate(fragment_tags):
            fragment.atoms[j].properties["tags"] = frag_tag
        slot_tags = slot.atoms.site_properties["tags"]
        for i, slot_tag in enumerate(slot_tags):
            tag_by_dist = []
            for j in fragment.atoms.indices_from_symbol("X"):
                d = slot.atoms[i].coords - fragment.atoms[j].coords
                tag_by_dist.append((numpy.linalg.norm(d), j))
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
