"""
UFF4MOF relaxation of built frameworks through LAMMPS.

lammps-interface (Boyd & Woo) types the framework and generates LAMMPS
inputs for the requested force field (UFF4MOF by default - the same
parameter set the rest of AuToGraFS speaks); the LAMMPS python module
then runs the alternating box-relax / FIRE minimization those inputs
define, in-process. The relaxed geometry is mapped back onto the
framework's bond graph, so the result is a normal Framework with the
same connectivity and updated coordinates, cell, and energy.

Both backends are optional::

    pip install "autografs[relax]"

On Windows, the LAMMPS wheel additionally needs the Microsoft MPI
runtime (https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi).

Notes
-----
lammps-interface replicates cells too small for the non-bonded cutoff
into a supercell. The relaxation preserves translational symmetry
(periodic starting point, deterministic minimizer), so the supercell
folds back exactly: the primitive cell is the relaxed supercell scaled
by the replication counts, and every atom maps onto its nearest
relaxed image, species-constrained and one-to-one.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import networkx
import numpy as np
from scipy.optimize import linear_sum_assignment

from autografs.exceptions import RelaxationError

if TYPE_CHECKING:
    from autografs.framework import Framework

logger = logging.getLogger(__name__)

# leading element symbol of a UFF4MOF type: C_R -> C, Zn4+2 -> Zn
_ELEMENT_OF_TYPE = re.compile(r"^([A-Z][a-z]?)")


@contextlib.contextmanager
def _muted_stderr_fd():
    """Silence the OS-level stderr file descriptor.

    Python-level redirect_stderr only reroutes sys.stderr;
    lammps_interface shells out to ``git rev-list HEAD`` inside
    site-packages at import time to stamp its version, and the child
    process writes 'fatal: not a git repository' straight to fd 2.
    The failure itself is caught and harmless - only the noise leaks.
    """
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def _import_backends():
    """Import the optional LAMMPS backends with a helpful error."""
    try:
        with _muted_stderr_fd():
            import lammps
            from lammps_interface.InputHandler import Options
            from lammps_interface.lammps_main import LammpsSimulation
            from lammps_interface.structure_data import from_CIF
    except ImportError as exc:
        raise RelaxationError(
            "UFF4MOF relaxation needs the optional LAMMPS backend: "
            'pip install "autografs[relax]". On Windows, the LAMMPS '
            "wheel also needs the Microsoft MPI runtime."
        ) from exc
    return lammps, Options, LammpsSimulation, from_CIF


def _make_options(options_cls, cif_path: str, force_field: str, cutoff: float):
    """Build a lammps-interface Options object programmatically.

    Options parses sys.argv, so the CLI arguments are staged there for
    the duration of the call; this tracks their schema instead of
    duplicating every default the simulation object reads.
    """
    staged = [
        "lammps-interface",
        "--minimize",
        "--force_field",
        force_field,
        "--cutoff",
        str(cutoff),
        cif_path,
    ]
    original = sys.argv
    sys.argv = staged
    try:
        return options_cls()
    finally:
        sys.argv = original


def _parse_type_elements(data_file: Path) -> dict[int, str]:
    """LAMMPS atom type id -> element, from the data file Masses block.

    lammps-interface comments every Masses line with the force-field
    type (``1 12.0107 # C_R``); the element is its leading symbol.
    """
    elements: dict[int, str] = {}
    in_masses = False
    for line in data_file.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("Masses"):
            in_masses = True
            continue
        if in_masses:
            if not stripped:
                if elements:
                    break
                continue
            parts = stripped.split()
            if not parts[0].isdigit():
                break
            if "#" not in stripped:
                raise RelaxationError(
                    f"Masses line without a type comment in {data_file.name}: "
                    f"{stripped!r}"
                )
            fftype = stripped.split("#", 1)[1].split()[0]
            match = _ELEMENT_OF_TYPE.match(fftype)
            if match is None:
                raise RelaxationError(f"Cannot read an element from {fftype!r}.")
            elements[int(parts[0])] = match.group(1)
    if not elements:
        raise RelaxationError(f"No Masses block found in {data_file.name}.")
    return elements


def _match_displacements(
    orig_frac: np.ndarray,
    orig_species: list[str],
    relaxed_frac: np.ndarray,
    relaxed_species: list[str],
    cell: np.ndarray,
) -> np.ndarray:
    """Fractional displacement of each original atom to its relaxed image.

    Assignment is species-constrained, one-to-one, and minimum-image:
    supercell replicas of the same source atom fold onto (nearly) the
    same fractional position, so any replica is a valid match.

    Parameters
    ----------
    orig_frac : np.ndarray
        (n, 3) wrapped fractional coordinates of the original atoms.
    orig_species, relaxed_species : list[str]
        Element symbols of both sets.
    relaxed_frac : np.ndarray
        (m, 3) wrapped fractional coordinates of the relaxed atoms in
        the primitive cell (m = n * replicas).
    cell : np.ndarray
        (3, 3) primitive cell matrix, used as the distance metric.

    Returns
    -------
    np.ndarray
        (n, 3) fractional displacements, minimum-image.
    """
    displacements = np.zeros_like(orig_frac)
    relaxed_species_arr = np.array(relaxed_species)
    for symbol in sorted(set(orig_species)):
        rows = np.flatnonzero(np.array(orig_species) == symbol)
        cols = np.flatnonzero(relaxed_species_arr == symbol)
        if len(cols) < len(rows):
            raise RelaxationError(
                f"Relaxed structure has {len(cols)} {symbol} atoms for "
                f"{len(rows)} originals; the atom mapping is inconsistent."
            )
        delta = relaxed_frac[cols][None, :, :] - orig_frac[rows][:, None, :]
        delta -= np.round(delta)
        cost = np.linalg.norm(delta @ cell, axis=2)
        row_idx, col_idx = linear_sum_assignment(cost)
        displacements[rows[row_idx]] = delta[row_idx, col_idx]
    return displacements


def relax_framework(
    framework: Framework,
    force_field: str = "UFF4MOF",
    cutoff: float = 12.5,
    verbose: bool = False,
) -> Framework:
    """Relax a framework's geometry and cell with LAMMPS.

    Parameters
    ----------
    framework : Framework
        The framework to relax; not modified.
    force_field : str, optional
        Force field passed to lammps-interface, by default "UFF4MOF".
        Other options include "UFF" and "Dreiding".
    cutoff : float, optional
        Non-bonded cutoff in Angstrom, by default 12.5. Cells too
        small for it are replicated into a supercell internally and
        folded back afterwards.
    verbose : bool, optional
        Pass the lammps-interface and LAMMPS output through instead of
        suppressing it.

    Returns
    -------
    Framework
        A new Framework with the same bond graph, relaxed coordinates
        and cell, and the UFF energy per unit cell (kcal/mol) in
        ``.energy``.

    Raises
    ------
    RelaxationError
        If the optional backends are missing, the structure contains
        free molecules lammps-interface cannot handle unattended, or
        the relaxed atoms cannot be mapped back onto the graph.
    """
    lammps, options_cls, simulation_cls, from_cif = _import_backends()
    # lammps-interface derives every file name from the cif basename
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", framework.name) or "framework"
    sink = io.StringIO()
    quiet: contextlib.AbstractContextManager = (
        contextlib.nullcontext() if verbose else contextlib.redirect_stdout(sink)
    )
    with tempfile.TemporaryDirectory(prefix="autografs_relax_") as tmp:
        workdir = Path(tmp)
        cif_path = workdir / f"{safe_name}.cif"
        framework.write_cif(cif_path)
        options = _make_options(options_cls, str(cif_path), force_field, cutoff)
        try:
            with quiet:
                sim = simulation_cls(options)
                cell, graph = from_cif(str(cif_path))
                sim.set_cell(cell)
                sim.set_graph(graph)
                sim.split_graph()
                sim.assign_force_fields()
                sim.compute_simulation_size()
                sim.merge_graphs()
                sim.write_lammps_files(wd=str(workdir))
        except EOFError as exc:
            # compute_simulation_size prompts interactively when it
            # detects free molecules; there is no API to answer it
            raise RelaxationError(
                f"lammps-interface found free molecules in "
                f"{framework.name!r}; relax() only handles connected "
                "frameworks."
            ) from exc
        supercell = np.array(sim.supercell, dtype=int)
        type_elements = _parse_type_elements(workdir / f"data.{safe_name}")

        try:
            lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none"])
        except OSError as exc:
            raise RelaxationError(
                "The LAMMPS runtime failed to load. On Windows, the "
                "LAMMPS wheel needs the Microsoft MPI runtime "
                "(winget install Microsoft.MSMPI)."
            ) from exc
        try:
            with quiet, contextlib.chdir(workdir):
                lmp.file(f"in.{safe_name}")
            # gather_atoms orders by atom id, matching the data file
            natoms = lmp.get_natoms()
            raw_x = lmp.gather_atoms("x", 1, 3)
            raw_t = lmp.gather_atoms("type", 0, 1)
            positions = np.array(raw_x[:], dtype=float).reshape(natoms, 3)
            types = np.array(raw_t[:], dtype=int)
            (xlo, ylo, zlo), (xhi, yhi, zhi), xy, yz, xz, *_ = lmp.extract_box()
            energy = float(lmp.get_thermo("pe"))
        finally:
            lmp.close()

    # LAMMPS convention: lattice vectors are the rows of a lower
    # triangular matrix; the primitive cell is the supercell scaled
    # down by the replication counts
    super_matrix = np.array(
        [
            [xhi - xlo, 0.0, 0.0],
            [xy, yhi - ylo, 0.0],
            [xz, yz, zhi - zlo],
        ]
    )
    prim_matrix = super_matrix / supercell[:, None]
    relaxed_frac = (positions @ np.linalg.inv(prim_matrix)) % 1.0
    relaxed_species = [type_elements[t] for t in types]

    orig_cell = framework.cell
    orig_frac = framework.cart_coords @ np.linalg.inv(orig_cell)
    displacements = _match_displacements(
        orig_frac % 1.0,
        framework.symbols,
        relaxed_frac,
        relaxed_species,
        prim_matrix,
    )
    moved = np.linalg.norm(displacements @ prim_matrix, axis=1)
    logger.info(
        f"Relaxed {framework.name!r} with {force_field}: energy "
        f"{energy / supercell.prod():.1f} kcal/mol per cell, max atom "
        f"displacement {moved.max():.2f} A."
    )
    # displacements apply to the unwrapped coordinates unchanged, so
    # bonded atoms stay cartesian neighbors in the graph
    new_frac = orig_frac + displacements
    new_coords = new_frac @ prim_matrix

    # nodes first, in sorted order, then edges: the relaxed graph gets
    # the same insertion order as every builder graph, so edge
    # iteration (and tuple orientation) matches the input exactly -
    # adding edges first would order nodes by edge encounter and flip
    # some reported orientations (networkx-internals-dependent, #145)
    relaxed = networkx.Graph(cell=prim_matrix)
    # cart_coords (and therefore new_coords) follow sorted node order
    for row, node in enumerate(sorted(framework.graph)):
        copied = dict(framework.graph.nodes[node])
        copied["coord"] = new_coords[row]
        relaxed.add_node(node, **copied)
    relaxed.add_edges_from(framework.graph.edges(data=True))
    from autografs.framework import Framework as FrameworkCls

    result = FrameworkCls(relaxed, name=framework.name)
    result.energy = energy / float(supercell.prod())
    return result
