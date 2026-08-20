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
import functools
import io
import logging
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import networkx
import numpy as np
from scipy.optimize import linear_sum_assignment

from autografs.data.uff4mof import UFF4MOF
from autografs.data.uff4mof import element_of as _element_of
from autografs.exceptions import RelaxationError

if TYPE_CHECKING:
    from autografs.framework import Framework

logger = logging.getLogger(__name__)

# leading element symbol of a UFF4MOF type: C_R -> C, Zn4+2 -> Zn
_ELEMENT_OF_TYPE = re.compile(r"^([A-Z][a-z]?)")


def _quiet(verbose: bool) -> contextlib.AbstractContextManager:
    """Suppress stdout unless verbose (LAMMPS and lammps-interface both
    narrate their setup to stdout)."""
    return (
        contextlib.nullcontext()
        if verbose
        else contextlib.redirect_stdout(io.StringIO())
    )


def _with_new_geometry(
    framework: Framework,
    new_coords: np.ndarray,
    cell: np.ndarray,
    energy: float | None = None,
) -> Framework:
    """The same framework graph with relaxed coordinates (and cell).

    Nodes first, in sorted order, then edges: the rebuilt graph gets
    the same insertion order as every builder graph, so edge iteration
    (and tuple orientation) matches the input exactly - adding edges
    first would order nodes by edge encounter and flip some reported
    orientations (networkx-internals-dependent, #145). ``new_coords``
    follows sorted node order.
    """
    rebuilt = networkx.Graph(cell=cell)
    for row, node in enumerate(sorted(framework.graph)):
        copied = dict(framework.graph.nodes[node])
        copied["coord"] = new_coords[row]
        rebuilt.add_node(node, **copied)
    rebuilt.add_edges_from(framework.graph.edges(data=True))
    from autografs.framework import Framework as FrameworkCls

    result = FrameworkCls(rebuilt, name=framework.name)
    result.energy = energy
    return result


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
    orig_species_arr = np.array(orig_species)
    relaxed_species_arr = np.array(relaxed_species)
    for symbol in sorted(set(orig_species)):
        rows = np.flatnonzero(orig_species_arr == symbol)
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


#: UFF symbols whose spelling differs between our table and
#: lammps-interface's for the SAME type. Only genuine renamings belong
#: here - anything else goes through the coordination substitution.
_TYPE_ALIASES = {
    # lawrencium: Lr is the current IUPAC symbol, Lw the historical one
    # the UFF paper used and lammps-interface kept
    "Lr6+3": "Lw6+3",
}


def _supported_types(force_field: str) -> set[str]:
    """Atom-type symbols the backend actually has parameters for.

    Our table (``data/uff4mof.py``, 227 symbols) and lammps-interface's
    (221) are both "UFF4MOF" and do not agree: 15 of ours are absent
    there, including ``Co6+2`` and ``N_3+4``. Handing one of those over
    raises KeyError before a single step runs, so the hand-off has to
    know what the receiver can take.
    """
    name = (force_field or "").lower()
    try:
        if "4mof" in name:
            from lammps_interface.uff4mof import UFF4MOF_DATA as table
        elif name.startswith("uff"):
            from lammps_interface.uff import UFF_DATA as table
        else:
            # Dreiding or anything else: no opinion, let it type
            return set()
    except ImportError:  # pragma: no cover - backend absent
        return set()
    return set(table)


@functools.cache
def _substitution_map(force_field: str) -> dict[str, str | None]:
    """Our type -> the closest type the backend supports.

    Same element always; among its supported types the one whose
    coordination is closest (ties broken by covalent radius), which
    keeps the connectivity information ``find_mmtypes`` derived even
    when the exact parameter set is missing downstream. None when the
    backend knows no type for that element at all - then the atom is
    left for lammps-interface to type, which is no worse than today.
    """
    supported = _supported_types(force_field)
    if not supported:
        return {}
    ours = {entry.symbol: entry for entry in UFF4MOF}
    by_element: dict[str, list] = {}
    for symbol in supported:
        entry = ours.get(symbol)
        if entry is not None:
            by_element.setdefault(_element_of(symbol), []).append(entry)
    mapping: dict[str, str | None] = {}
    for symbol, entry in ours.items():
        if symbol in supported:
            mapping[symbol] = symbol
            continue
        alias = _TYPE_ALIASES.get(symbol)
        if alias and alias in supported:
            mapping[symbol] = alias
            continue
        # UFF pads the element into a two-character field; our table
        # spells single-letter elements bare where lammps-interface pads
        # them ("K" vs "K_"), which is the same type, not a near miss
        padded = symbol[:2].ljust(2, "_") + symbol[2:]
        if padded != symbol and padded in supported:
            mapping[symbol] = padded
            continue
        candidates = by_element.get(_element_of(symbol))
        if not candidates:
            mapping[symbol] = None
            continue
        best = min(
            candidates,
            key=lambda other: (
                abs(other.coordination - entry.coordination),
                abs(other.radius - entry.radius),
            ),
        )
        mapping[symbol] = best.symbol
    return mapping


def _hand_off_atom_types(framework: Framework, graph, force_field: str) -> int:
    """Give lammps-interface the UFF types we already know.

    A Framework's bond graph is the source of truth for UFF4MOF atom
    types: ``utils.find_mmtypes`` assigns them from real connectivity,
    against the table in ``data/uff4mof.py``. Staging goes through a
    CIF, which cannot carry them, so lammps-interface re-perceives the
    bonding from geometry and re-types every atom - discarding work we
    have already done correctly and substituting a guess. On unusual
    coordination that guess can be a type absent from its own parameter
    tables (``S_3``, where UFF's sulfur types are ``S_3+2/+4/+6``),
    which raises KeyError before a single step runs.

    ``ForceFields.detect_ff_terms`` types an atom only when its
    ``force_field_type`` is still None, so filling it in here is the
    supported hand-off rather than a monkey-patch.

    The correspondence is positional - ``write_cif`` emits atoms in
    sorted node order - so it is **verified element by element** and
    abandoned wholesale on any mismatch, leaving lammps-interface's own
    typing in place. A wrong type is worse than a guessed one.

    Returns
    -------
    int
        How many atoms were typed from the framework; 0 when the
        correspondence did not check out.
    """
    ours = [
        framework.graph.nodes[n].get("ufftype") for n in sorted(framework.graph.nodes)
    ]
    symbols = [
        framework.graph.nodes[n].get("symbol") for n in sorted(framework.graph.nodes)
    ]
    theirs = list(graph.nodes_iter2(data=True))
    if len(theirs) != len(ours) or not all(ours):
        logger.debug(
            f"Not handing off atom types for {framework.name!r}: "
            f"{len(ours)} framework atoms against {len(theirs)} staged."
        )
        return 0
    for (_node, data), symbol in zip(theirs, symbols, strict=True):
        if data.get("element") != symbol:
            logger.debug(
                f"Not handing off atom types for {framework.name!r}: staged "
                f"element {data.get('element')} where the framework has {symbol}."
            )
            return 0
    # translate into the receiver's vocabulary; an atom we cannot express
    # there is left untyped rather than handed a symbol it will KeyError on
    mapping = _substitution_map(force_field)
    substitutions: Counter[str] = Counter()
    typed = 0
    for (_node, data), uff_type in zip(theirs, ours, strict=True):
        target = mapping.get(uff_type, uff_type) if mapping else uff_type
        if target is None:
            continue
        if target != uff_type:
            substitutions[f"{uff_type}->{target}"] += 1
        data["force_field_type"] = target
        typed += 1
    if substitutions:
        logger.info(
            f"Atom types unsupported by the {force_field} backend were "
            f"substituted for {framework.name!r}: {dict(substitutions)}."
        )
    if typed < len(ours):
        logger.info(
            f"{len(ours) - typed} atom(s) of {framework.name!r} have no "
            f"{force_field} counterpart and were left for lammps-interface."
        )
    return typed


def _write_lammps_inputs(
    framework: Framework,
    force_field: str,
    cutoff: float,
    workdir: Path,
    quiet: contextlib.AbstractContextManager,
) -> tuple[str, np.ndarray]:
    """Stage lammps-interface data/input files for a framework.

    Writes the framework as a CIF into ``workdir`` and runs the
    lammps-interface pipeline on it, leaving ``in.{name}`` and
    ``data.{name}`` behind. The input file performs the full
    alternating box-relax / FIRE minimization when run.

    Returns
    -------
    tuple[str, np.ndarray]
        The sanitized basename all files derive from, and the (3,)
        integer supercell replication lammps-interface chose.
    """
    _, options_cls, simulation_cls, from_cif = _import_backends()
    # lammps-interface derives every file name from the cif basename
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", framework.name) or "framework"
    cif_path = workdir / f"{safe_name}.cif"
    # hand our own bonds over rather than letting lammps-interface
    # re-perceive them from geometry: they come from the blueprint's
    # dummy correspondences and are exact, and from_CIF prefers a
    # _geom_bond_ loop over its own perception when one is present
    framework.write_cif(cif_path, write_bonds=True)
    options = _make_options(options_cls, str(cif_path), force_field, cutoff)
    try:
        with quiet:
            sim = simulation_cls(options)
            cell, graph = from_cif(str(cif_path))
            _hand_off_atom_types(framework, graph, force_field)
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
    return safe_name, np.array(sim.supercell, dtype=int)


#: input commands worth announcing: these are the long poles, and a
#: relaxation that reports nothing between them looks like a hung process
_LOUD_COMMANDS = ("minimize", "run", "fix", "velocity")


def _run_lammps_input(lmp, input_path: Path, name: str) -> None:
    """Feed a lammps-interface input file command by command.

    ``lmp.file()`` runs the whole script in one opaque call, so a
    framework of any size sits silent for minutes and is indistinguishable
    from a hung process. Replaying the commands individually costs
    nothing and lets the expensive ones be logged as they start.
    Continuation lines (``&``) are rejoined first, since LAMMPS treats
    them as a single command.
    """
    raw = input_path.read_text(encoding="utf-8").splitlines()
    commands: list[str] = []
    buffer = ""
    for line in raw:
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        if stripped.endswith("&"):
            buffer += stripped[:-1] + " "
            continue
        commands.append(buffer + stripped)
        buffer = ""
    if buffer:
        commands.append(buffer)
    total = sum(1 for c in commands if c.split()[0] in _LOUD_COMMANDS)
    done = 0
    for command in commands:
        head = command.split()[0]
        if head in _LOUD_COMMANDS:
            done += 1
            logger.info(f"Relaxing {name!r}: [{done}/{total}] {command[:60]}")
        lmp.command(command)


def _launch_lammps():
    """Create an in-process LAMMPS session with a helpful error."""
    lammps, *_ = _import_backends()
    try:
        return lammps.lammps(cmdargs=["-log", "none", "-screen", "none"])
    except OSError as exc:
        raise RelaxationError(
            "The LAMMPS runtime failed to load. On Windows, the "
            "LAMMPS wheel needs the Microsoft MPI runtime "
            "(winget install Microsoft.MSMPI)."
        ) from exc


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
    quiet = _quiet(verbose)
    # ignore_cleanup_errors: on Windows a handle on the LAMMPS data file
    # can outlive lmp.close(), and the resulting PermissionError from the
    # cleanup would discard a relaxation that had already succeeded. The
    # staging directory is disposable; the result is not.
    with tempfile.TemporaryDirectory(
        prefix="autografs_relax_", ignore_cleanup_errors=True
    ) as tmp:
        workdir = Path(tmp)
        safe_name, supercell = _write_lammps_inputs(
            framework, force_field, cutoff, workdir, quiet
        )
        type_elements = _parse_type_elements(workdir / f"data.{safe_name}")

        lmp = _launch_lammps()
        try:
            with quiet, contextlib.chdir(workdir):
                _run_lammps_input(lmp, Path(f"in.{safe_name}"), framework.name)
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

    return _with_new_geometry(
        framework, new_coords, prim_matrix, energy=energy / float(supercell.prod())
    )


#: Below this closest contact a UFF start is unusable: LJ goes as
#: r^-12, so an overlapping pair overflows the force, positions turn
#: non-finite and LAMMPS reports the symptom as a lost bond
#: ("Bond atoms N M missing on proc 0"). Measured on the generative
#: candidates: everything with a pair under 0.69 A failed this way,
#: everything over 0.86 A relaxed.
#: Run the push-off when the closest pair is under this, in Angstrom.
#: 1.3 is corpus-measured, not the "is it overlapping" intuition: 0.85
#: left 28 structures whose closest pair was 0.85-1.3 A to overflow the
#: full force field, and raising it rescued 27 of them while a control
#: of 14 that already relaxed was untouched (all 14 kept, median contact
#: 1.66 -> 1.68 A). A pair well inside a bond length is already enough
#: for r^-12 to dominate, so "not quite overlapping" is not safe.
SOFT_PUSHOFF_CONTACT = 1.3
#: Soft-potential prefactor, kcal/mol. Measured: 10 separates every
#: overlapping candidate tried (cds 0.44 -> 1.06, qzd 0.44 -> 1.04, lon
#: 0.17 -> 1.00) while 100 overshoots into a fresh collapse (cds -> 0.00)
#: and 1000 fails outright. Stronger is NOT safer here.
SOFT_PUSHOFF_PREFACTOR = 10.0
#: Soft cutoff; wide enough to reach a second-neighbour overlap.
SOFT_PUSHOFF_CUTOFF = 4.0
#: Below this the pair is close enough to coincident that the soft
#: force (which vanishes at r = 0) cannot separate it unaided.
SOFT_PUSHOFF_JITTER_BELOW = 0.3
#: Fixed seed: the jitter must not make a relaxation irreproducible.
SOFT_PUSHOFF_SEED = 20260806


def _pushoff_jitter(data) -> list[str]:
    """Break a coincident pair's degeneracy before the soft push.

    The soft force goes as sin(pi r / rc), so it VANISHES at r = 0 and
    an exactly-coincident pair would never separate. The seed is fixed:
    a relaxation must stay reproducible.
    """
    if data.closest_contact >= SOFT_PUSHOFF_JITTER_BELOW:
        return []
    return [f"displace_atoms all random 0.1 0.1 0.1 {SOFT_PUSHOFF_SEED} units box"]


def _push_off_overlaps(
    framework: Framework, force_field: str, verbose: bool = False
) -> Framework:
    """Separate overlapping atoms before the real relaxation runs.

    Only the PAIR term is softened; every bonded term stays. Dropping
    the angles, torsions and inversions to make the push-off
    "well-conditioned" was tried and is decisively worse - 79 of 210
    against 124, gaining 4 structures and losing 49. Without the angle
    terms the push-off deforms the units themselves (a ring folds, a
    tetrahedral centre flattens), and the full force field then starts
    from a chemically wrong geometry that is harder to fix than the
    overlap was. Do not re-try it.

    Returns the framework unchanged when nothing is overlapping.
    """
    from autografs.lammps_data import write_lammps_data

    _import_backends()
    data = write_lammps_data(framework, force_field=force_field)
    if data.closest_contact >= SOFT_PUSHOFF_CONTACT:
        return framework

    quiet = _quiet(verbose)
    with tempfile.TemporaryDirectory(
        prefix="autografs_pushoff_", ignore_cleanup_errors=True
    ) as tmp:
        path = (Path(tmp) / "data.pushoff").as_posix()
        Path(path).write_text(data.text, encoding="utf-8")
        lmp = _launch_lammps()
        try:
            with quiet:
                for command in (
                    "units real",
                    "atom_style full",
                    "boundary p p p",
                    f"bond_style {data.styles['bond_style']}",
                    f"angle_style {data.styles['angle_style']}",
                    f"dihedral_style {data.styles['dihedral_style']}",
                    f"improper_style {data.styles['improper_style']}",
                    f"special_bonds {data.styles['special_bonds']}",
                    f"comm_modify {data.styles['comm_modify']}",
                    f"pair_style soft {SOFT_PUSHOFF_CUTOFF}",
                    f"read_data {path}",
                    # the prefactor is set OUTRIGHT: the documented
                    # `fix adapt` + ramp() recipe is written for `run`
                    # and never applies during `minimize`
                    f"pair_coeff * * {SOFT_PUSHOFF_PREFACTOR}",
                    *_pushoff_jitter(data),
                    "min_style fire",
                    "min_modify dmax 0.02",
                    "minimize 1.0e-8 1.0e-8 2000 20000",
                ):
                    lmp.command(command)
                natoms = lmp.get_natoms()
                positions = np.array(
                    lmp.gather_atoms("x", 1, 3)[:], dtype=float
                ).reshape(natoms, 3)
                (lo, _hi, *_rest) = lmp.extract_box()
        finally:
            lmp.close()

    if natoms != len(framework.graph):
        raise RelaxationError(
            f"the push-off returned {natoms} atoms for a "
            f"{len(framework.graph)}-atom framework."
        )
    pushed = framework.graph.copy()
    for row, node in enumerate(sorted(framework.graph)):
        pushed.nodes[node]["coord"] = positions[row] - np.asarray(lo, dtype=float)
    from autografs.framework import Framework as FrameworkCls

    result = FrameworkCls(pushed, name=framework.name)
    logger.info(
        f"Pushed overlapping atoms in {framework.name!r} apart before "
        f"relaxing: closest pair {data.closest_contact:.2f} -> "
        f"{result.min_contact():.2f} A."
    )
    return result


#: A relaxation that leaves two atoms this close has not converged to
#: a structure, whatever the minimiser reported. Below any real bond.
COLLAPSE_DISTANCE = 0.5
#: Losing this much volume is a collapse, not a contraction. UFF's own
#: contraction is 2-15% (see the UFF4MOF notes), so 0.2 is far outside.
COLLAPSE_VOLUME_RATIO = 0.2


def _reject_collapse(before: Framework, after: Framework) -> None:
    """Refuse a "successful" relaxation that returned a collapsed cell.

    Measured need: relaxing a deliberately expanded start reported
    success while returning atoms 0.00 A apart. A minimiser converging
    on a degenerate structure is a failure, and it must not be handed
    back as a result.
    """
    coords = after.structure.distance_matrix.copy()
    np.fill_diagonal(coords, np.inf)
    closest = float(coords.min())
    ratio = float(after.structure.volume / before.structure.volume)
    if closest < COLLAPSE_DISTANCE:
        raise RelaxationError(
            f"the relaxation of {before.name!r} converged with two atoms "
            f"{closest:.2f} A apart (under {COLLAPSE_DISTANCE} A): the cell "
            "collapsed rather than relaxed."
        )
    if ratio < COLLAPSE_VOLUME_RATIO:
        raise RelaxationError(
            f"the relaxation of {before.name!r} kept only {ratio:.0%} of the "
            "built cell volume: the cell collapsed rather than relaxed."
        )


def relax_framework_native(
    framework: Framework,
    force_field: str = "UFF4MOF",
    verbose: bool = False,
    steps: int = 10000,
) -> Framework:
    """Relax through a data file we write ourselves (lammps_data).

    Same engine as ``relax_framework``, different staging: the topology
    comes from the framework's own bond graph rather than from
    lammps-interface's re-perception of the geometry. That removes the
    failure modes measured over 516 candidates (invalid dihedral IDs,
    C-level aborts, RecursionError, silent collapse) and, because the
    cutoff is chosen under half the box, the internal supercell and its
    fold-back with it - atoms come back in the order they were written.

    Validated against lammps-interface on MOF-5 at identical geometry:
    every bond coefficient equal to six decimals, total energy within
    0.5% (bonds 0.0%, angles 0.1%, dihedrals 0.7%; the van der Waals
    term differs by the cutoff).
    """
    from autografs.lammps_data import write_lammps_data

    lammps, _, _, _ = _import_backends()
    # separate any overlap FIRST, in a separate stage that softens only
    # the pair term (every bonded term kept), so the full force field
    # never sees the geometry that overflows its r^-12 term
    original = framework
    framework = _push_off_overlaps(framework, force_field, verbose=verbose)
    data = write_lammps_data(framework, force_field=force_field)
    quiet = _quiet(verbose)
    with tempfile.TemporaryDirectory(
        prefix="autografs_native_", ignore_cleanup_errors=True
    ) as tmp:
        path = (Path(tmp) / "data.framework").as_posix()
        Path(path).write_text(data.text, encoding="utf-8")
        lmp = _launch_lammps()
        try:
            with quiet:
                for command in (
                    "units real",
                    "atom_style full",
                    "boundary p p p",
                    f"pair_style {data.styles['pair_style']}",
                    f"pair_modify {data.styles['pair_modify']}",
                    f"bond_style {data.styles['bond_style']}",
                    f"angle_style {data.styles['angle_style']}",
                    f"dihedral_style {data.styles['dihedral_style']}",
                    f"improper_style {data.styles['improper_style']}",
                    f"special_bonds {data.styles['special_bonds']}",
                    # the ghost shell has to reach the far end of every
                    # bond, which in a small cell is wider than the pair
                    # cutoff; without it LAMMPS drops the bond at setup
                    f"comm_modify {data.styles['comm_modify']}",
                    f"read_data {path}",
                    "min_style fire",
                    "min_modify dmax 0.05",
                    f"minimize 1.0e-10 1.0e-10 {steps} {steps * 10}",
                    # box/relax refuses a damped-dynamics minimiser
                    "min_style cg",
                    "fix boxrelax all box/relax tri 0.0 vmax 0.001",
                    f"minimize 1.0e-10 1.0e-10 {steps} {steps * 10}",
                ):
                    lmp.command(command)
                natoms = lmp.get_natoms()
                positions = np.array(
                    lmp.gather_atoms("x", 1, 3)[:], dtype=float
                ).reshape(natoms, 3)
                (xlo, ylo, zlo), (xhi, yhi, zhi), xy, yz, xz, *_ = lmp.extract_box()
                energy = float(lmp.get_thermo("pe"))
        finally:
            lmp.close()

    if natoms != len(framework.graph):
        raise RelaxationError(
            f"LAMMPS returned {natoms} atoms for a {len(framework.graph)}-atom "
            "framework; the native data file was not read as written."
        )
    cell = np.array(
        [
            [xhi - xlo, 0.0, 0.0],
            [xy, yhi - ylo, 0.0],
            [xz, yz, zhi - zlo],
        ]
    )
    # no supercell, so no fold-back: LAMMPS atom i+1 IS sorted node i
    result = _with_new_geometry(
        framework, positions - np.array([xlo, ylo, zlo]), cell, energy=energy
    )
    _reject_collapse(original, result)
    logger.info(
        f"Relaxed {framework.name!r} natively with {force_field}: "
        f"{energy:.1f} kcal/mol, {data}."
    )
    return result
