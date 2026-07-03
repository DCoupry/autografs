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

import itertools
import logging
import time
from collections.abc import Mapping
from pathlib import Path

import dill
from scipy.optimize import minimize
from tqdm.auto import tqdm

import autografs.alignment
import autografs.data
import autografs.topology_io
import autografs.utils
from autografs.exceptions import AlignmentError
from autografs.fragment import Fragment
from autografs.framework import Framework
from autografs.topology import Topology

logger = logging.getLogger(__name__)

# Constants for cell parameter optimization
NELDER_MEAD_XATOL = 0.1  # Absolute tolerance for cell parameter convergence
NELDER_MEAD_FATOL = 0.01  # Absolute tolerance for RMSE convergence
NELDER_MEAD_MAXITER = 100  # Maximum iterations for optimization


class Autografs:
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
        self.topologies: Mapping[str, Topology] = {}
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
        max_rmsd: float | None = None,
    ) -> list[Framework]:
        """
        Builds all available structures based on the SBU and Topologies libraries

        Parameters
        ----------
        sbu_subset : list[str] | None, optional
            The subset of SBU to build on, by default None
        topology_subset : list[str] | None, optional
            the subset of topologies to build on, by default None
        refine_cell : bool, optional
            If True, optimizes cell parameters for each build, by default True
        max_rmsd : float or None, optional
            Per-slot alignment RMSD gate forwarded to build(); rejected
            structures are counted as errors, by default None

        Returns
        -------
        list[Framework]
            the frameworks produced by the building method
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
            # slot types with no compatible SBU are absent from the
            # dict entirely, so completeness needs an explicit check
            if len(sbu_dict) != len(topology.mappings):
                continue
            if not all(sbu_dict.values()):
                continue
            for sbus in list(itertools.product(*sbu_dict.values())):
                maps.append((topology, dict(zip(sbu_dict.keys(), sbus, strict=True))))
            total_len += len(maps)
            results_graphs = []
            for build_args in tqdm(maps):
                try:
                    g = self.build(
                        *build_args,
                        refine_cell=refine_cell,
                        max_rmsd=max_rmsd,
                        verbose=False,
                    )
                    results_graphs.append(g)
                except (AssertionError, AlignmentError, ValueError):
                    # expected failures: gated alignments, unfillable
                    # slots. Anything else is a bug and propagates.
                    continue
            graphs += [g for g in results_graphs if g is not None]
        logger.info(f"\t[x] Generated a total of {len(graphs):^6} periodic graphs.")
        logger.info(f"\t[x] Rate of error: {1.0 - len(graphs) / total_len:2.2%}.")
        return graphs

    def build(
        self,
        topology: Topology,
        mappings: dict[Fragment | int, Fragment | str],
        refine_cell: bool = True,
        verbose: bool = False,
        max_rmsd: float | None = None,
    ) -> Framework:
        """
        Generates a framework from a mapping of SBU to topology slots.

        Parameters
        ----------
        topology : Topology
            the topology to consider
        mappings : dict[Fragment | int, Fragment | str]
            the mappings to go from a slot to a compatible SBU
        refine_cell : bool, optional
            If True, optimizes the cell parameters so that bonded dummy
            pairs coincide, by default True. The analytic starting cell
            is already exact for uninodal isotropic nets.
        verbose : bool, optional
            If True, will log additional information, by default False
        max_rmsd : float or None, optional
            Maximum acceptable per-slot directional RMSD: the shape
            mismatch between the SBU's unit arm vectors and the slot's,
            independent of size (0 = perfect, 2 = opposite). If any
            slot exceeds it, AlignmentError is raised instead of
            silently returning a distorted structure. None (default)
            disables the gate.

        Returns
        -------
        Framework
            The built framework: structure, bond graph, and exports
            (write_cif, to_ase, to_gulp). The underlying networkx
            graph is available as Framework.graph.

        Raises
        ------
        AlignmentError
            If an SBU's connection count does not match its slot, or if
            max_rmsd is set and any slot alignment exceeds it.
        ValueError
            If the mappings leave topology slots unfilled.
        """
        mappings = self._validate_mappings(topology=topology, mappings=mappings)
        if verbose:
            logger.info("Starting building process:")
            logger.info(f"\tTopology =  {topology}")
            formatted_mappings = autografs.utils.format_mappings(mappings)
            logger.info(f"\tMappings =  {formatted_mappings}")
            logger.info("Aligning with cell scaling...")

        t0 = time.time()
        plan = autografs.alignment.prepare_build(topology, mappings)
        x0 = plan.initial_parameters()
        if refine_cell and plan.has_pairs:
            # Nelder-Mead on the pair-coincidence residual over the
            # crystal system's free parameters only (a cubic net
            # optimizes a single length); the objective is pure numpy,
            # no object copies per evaluation
            result = minimize(
                plan.residual,
                x0,
                method="Nelder-Mead",
                options={
                    "xatol": NELDER_MEAD_XATOL,
                    "fatol": NELDER_MEAD_FATOL,
                    "maxiter": NELDER_MEAD_MAXITER,
                },
            )
            best_parameters = result.x
        else:
            if verbose:
                logger.info("\t[x] No cell refinement performed.")
            best_parameters = x0
        best_alignment, lattice, slot_rmsds = plan.finalize(best_parameters)
        if max_rmsd is not None:
            bad_slots = {i: r for i, r in slot_rmsds.items() if r > max_rmsd}
            if bad_slots:
                worst = max(bad_slots.values())
                raise AlignmentError(
                    f"{len(bad_slots)} of {len(slot_rmsds)} slots exceed "
                    f"max_rmsd={max_rmsd:.3f} (worst: {worst:.3f}) "
                    f"on topology {topology.name}: {sorted(bad_slots)}"
                )
        if verbose:
            a, b, c = lattice.abc
            alpha, beta, gamma = lattice.angles
            logger.info("\t[x] Best cell parameters:")
            logger.info(f"\t\ta={a:<.2f} b={b:<.2f} c={c:<.2f}")
            logger.info(f"\t\talpha={alpha:<.1f} beta={beta:<.1f} gamma={gamma:<.1f}")
            logger.info(
                f"\t[x] Aligned {len(best_alignment)} fragments in {time.time() - t0:.1f} seconds"
            )
        graph = autografs.utils.fragments_to_networkx(
            best_alignment, cell=lattice.matrix
        )
        return Framework(graph, name=topology.name)

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
            Normalized mappings from slot indices to private Fragment
            copies. Neither the input dict nor the library fragments
            are aliased: alignment mutates fragments in place, and
            shared references would corrupt the SBU library.

        Raises
        ------
        ValueError
            If any topology slot is left uncovered by the mappings.
        """

        def to_fragment(value: Fragment | str) -> Fragment:
            return self.sbu[value] if isinstance(value, str) else value

        # slot-type keys first, so that index keys override them
        true_mappings: dict[int, Fragment] = {}
        for k, v in mappings.items():
            if not isinstance(k, int):
                fragment = to_fragment(v)
                for i in topology.mappings[k]:
                    true_mappings[i] = fragment.copy()
        for k, v in mappings.items():
            if isinstance(k, int):
                true_mappings[k] = to_fragment(v).copy()
        all_indices = set(itertools.chain(*topology.mappings.values()))
        missing = all_indices - true_mappings.keys()
        if missing:
            raise ValueError(f"Unfilled slots in mappings: {sorted(missing)}")
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
        default_path = Path(autografs.data.__path__[0]) / "defaults.xyz"
        sbu = autografs.utils.xyz_to_sbu(str(default_path))
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
        """Load topologies from a serialized library file.

        Loads topology blueprints from a JSON library (see
        autografs.topology_io) or, for backward compatibility, from a
        legacy dill pickle.

        Parameters
        ----------
        topofile : str or None, optional
            Path to a topology library (.json / .json.gz / legacy .pkl).
            If None, loads from the default package data location,
            preferring the JSON library when present.
        """
        t0 = time.time()
        if topofile is None:
            data_dir = Path(autografs.data.__path__[0])
            json_default = data_dir / "topologies.json.gz"
            if json_default.exists():
                topofile = json_default
            else:
                topofile = data_dir / "topologies.pkl"
        topofile = Path(topofile)
        if topofile.name.endswith((".json", ".json.gz")):
            topologies = autografs.topology_io.load_topologies(topofile)
        else:
            logger.warning(
                "Loading topologies from a pickle file. Pickles can execute "
                "arbitrary code and break across pymatgen versions; convert "
                "to JSON with autografs.topology_io.save_topologies."
            )
            with open(topofile, "rb") as topo:
                topologies = dill.load(topo)
        logger.info(
            f"\t[x] loaded {len(topologies)} topologies in {time.time() - t0:.0f} seconds."
        )
        # assignment, not dict.update(): the JSON loader returns a lazy
        # mapping, and update() would materialize every topology
        self.topologies = topologies
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
        if subset is not None:
            unknown = set(subset) - self.topologies.keys()
            if unknown:
                raise ValueError(f"Unknown topologies in subset: {sorted(unknown)}")
            full_list = sorted(subset)
        else:
            full_list = sorted(self.topologies.keys())
        if sieve is not None:
            candidate = self.sbu[sieve]
            if verbose:
                logger.info(
                    f"filtering topologies for compatibility with {candidate} building unit..."
                )
            full_list = [
                name
                for name in full_list
                if any(
                    indices
                    for indices in self.topologies[name]
                    .get_compatible_slots(candidate=candidate)
                    .values()
                )
            ]
        if verbose:
            logger.info(f"\t[x] {len(full_list):>5} topologies available.")
        return full_list

    def list_building_units(
        self,
        sieve: str | None = None,
        verbose: bool = False,
        subset: list[str] | None = None,
    ) -> dict[Fragment, list[str]]:
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
        dict[Fragment, list[str]]
            Compatible SBU names keyed by slot type (the topology's
            representative slot Fragments). Slot types with no
            compatible SBU are absent from the dict; without a sieve
            the dict is empty.

        Examples
        --------
        >>> mofgen = Autografs()
        >>> sbu_dict = mofgen.list_building_units(sieve="pcu")
        >>> for slot, sbus in sbu_dict.items():
        ...     print(f"Slot {slot}: {len(sbus)} compatible SBUs")
        """
        if subset is not None:
            sbus = [self.sbu[k] for k in subset]
        else:
            sbus = list(self.sbu.values())
        building_units: dict[Fragment, set[str]] = {}
        topology = self.topologies[sieve] if sieve is not None else None
        if topology is not None:
            if verbose:
                logger.info(
                    f"filtering building units for compatibility with {topology} topology..."
                )
            for sbu in sbus:
                compatible = topology.get_compatible_slots(candidate=sbu)
                for slot_type, indices in compatible.items():
                    if indices:
                        building_units.setdefault(slot_type, set()).add(sbu.name)
        out_dict = {k: sorted(v) for k, v in building_units.items()}
        if verbose:
            for k, v in out_dict.items():
                logger.info(f"\t[x] {len(v):>5} SBU available for slot {k}")
            # report slot types that no SBU can fill
            if topology is not None:
                for k in topology.mappings:
                    if k not in out_dict:
                        logger.info(f"\t[!] {0:>5} SBU available for slot {k}")
        return out_dict
