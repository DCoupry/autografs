"""
ASE-calculator relaxation backends: the funnel levels above UFF4MOF.

``Framework.relax`` speaks LAMMPS/UFF4MOF natively (see ``relax.py``);
the multi-fidelity funnel needs higher levels — periodic tight-binding
(GFN-FF, GFN1-xTB) and DFTB+ — without one bespoke backend per code.
This module is the thin bridge: any ASE calculator that provides
energy, forces and (for cell relaxation) stress can relax a framework,
and the well-known ones are constructible by name:

======== ============================== ==============================
name     calculator                     install
======== ============================== ==============================
gfn-ff   xtb-python, method "GFNFF"     conda install -c conda-forge xtb-python
gfn1     tblite, method "GFN1-xTB"      pip install tblite
gfn2     rejected: GFN2-xTB has no periodic implementation
dftb     ase.calculators.dftb.Dftb      DFTB+ binary + Slater-Koster
                                        files (DFTB_PREFIX)
======== ============================== ==============================

The bond graph is preserved exactly as in the LAMMPS path: only
coordinates, cell and energy change, and the relaxed graph is built
nodes-first in sorted order so edge iteration matches the input
(#145). Energies are converted from ASE's eV to the kcal/mol per unit
cell convention ``Framework.energy`` uses everywhere.
"""

from __future__ import annotations

import contextlib
import io
import logging
from typing import TYPE_CHECKING, Any

import networkx
import numpy as np

from autografs.exceptions import RelaxationError

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

    from autografs.framework import Framework

logger = logging.getLogger(__name__)

EV_TO_KCAL_PER_MOL = 23.060548

# constructors are imported lazily; each entry is (builder, install hint)
_KNOWN_CALCULATORS = ("gfn-ff", "gfn1", "gfn2", "dftb")


def make_calculator(name: str, **kwargs: Any) -> Calculator:
    """Construct a named ASE calculator for periodic frameworks.

    Parameters
    ----------
    name : str
        One of "gfn-ff" (xtb-python), "gfn1" (tblite), "dftb"
        (DFTB+ through ASE). "gfn2" is rejected: GFN2-xTB has no
        periodic implementation. Case-insensitive; "gfnff",
        "gfn1-xtb", "dftb+" also resolve.
    **kwargs
        Passed to the calculator constructor (e.g. DFTB+ Hamiltonian
        settings).

    Returns
    -------
    Calculator
        A ready-to-use ASE calculator.

    Raises
    ------
    RelaxationError
        If the name is unknown, the backing package is not installed,
        or the method cannot treat periodic systems.
    """
    key = name.lower().replace("_", "-")
    if key in ("gfn-ff", "gfnff"):
        try:
            from xtb.ase.calculator import XTB
        except ImportError as exc:
            raise RelaxationError(
                "GFN-FF needs the xtb python bindings: "
                "conda install -c conda-forge xtb-python"
            ) from exc
        return XTB(method="GFNFF", **kwargs)
    if key in ("gfn1", "gfn1-xtb"):
        try:
            from tblite.ase import TBLite
        except ImportError as exc:
            raise RelaxationError(
                "GFN1-xTB needs tblite: pip install tblite"
            ) from exc
        return TBLite(method="GFN1-xTB", **kwargs)
    if key in ("gfn2", "gfn2-xtb"):
        raise RelaxationError(
            "GFN2-xTB has no periodic implementation; use 'gfn1' or "
            "'gfn-ff' for periodic frameworks."
        )
    if key in ("dftb", "dftb+"):
        try:
            from ase.calculators.dftb import Dftb
        except ImportError as exc:  # pragma: no cover - ships with ase
            raise RelaxationError("The ASE DFTB+ calculator is unavailable.") from exc
        try:
            return Dftb(**kwargs)
        except Exception as exc:
            raise RelaxationError(
                "DFTB+ could not be set up; it needs the dftb+ binary "
                "on PATH and Slater-Koster files (DFTB_PREFIX). "
                f"Underlying error: {exc}"
            ) from exc
    raise RelaxationError(
        f"Unknown calculator {name!r}; known names: "
        f"{', '.join(_KNOWN_CALCULATORS)} (or pass an ASE Calculator "
        "instance directly)."
    )


def relax_framework_ase(
    framework: Framework,
    calculator: Calculator | str,
    relax_cell: bool = True,
    fmax: float = 0.05,
    steps: int = 500,
    verbose: bool = False,
) -> Framework:
    """Relax a framework's geometry (and cell) with an ASE calculator.

    Parameters
    ----------
    framework : Framework
        The framework to relax; not modified.
    calculator : Calculator or str
        An ASE calculator instance, or a name ``make_calculator``
        understands ("gfn-ff", "gfn1", "dftb").
    relax_cell : bool, optional
        Optimize the cell along with the positions (through a
        FrechetCellFilter; the calculator must provide stress), by
        default True.
    fmax : float, optional
        Force convergence threshold in eV/Angstrom, by default 0.05.
    steps : int, optional
        Maximum optimizer steps, by default 500. Non-convergence is
        logged as a warning, not raised.
    verbose : bool, optional
        Stream the optimizer log instead of suppressing it.

    Returns
    -------
    Framework
        A new Framework with the same bond graph, relaxed coordinates
        and cell, and the energy per unit cell (kcal/mol, converted
        from eV) in ``.energy``.

    Raises
    ------
    RelaxationError
        If the calculator cannot be constructed or the calculation
        fails.
    """
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    if isinstance(calculator, str):
        calculator = make_calculator(calculator)

    atoms = framework.to_ase()
    atoms.calc = calculator
    initial_cell = np.array(atoms.cell)
    initial_frac = atoms.get_scaled_positions(wrap=False)

    target = FrechetCellFilter(atoms) if relax_cell else atoms
    sink = io.StringIO()
    quiet: contextlib.AbstractContextManager = (
        contextlib.nullcontext() if verbose else contextlib.redirect_stdout(sink)
    )
    try:
        with quiet:
            optimizer = FIRE(target, logfile="-" if verbose else None)
            converged = optimizer.run(fmax=fmax, steps=steps)
            energy_ev = float(atoms.get_potential_energy())
    except RelaxationError:
        raise
    except Exception as exc:
        raise RelaxationError(
            f"ASE relaxation of {framework.name!r} with "
            f"{type(calculator).__name__} failed: {exc}"
        ) from exc
    if not converged:
        logger.warning(
            f"ASE relaxation of {framework.name!r} did not reach "
            f"fmax={fmax} within {steps} steps; returning the last "
            "geometry."
        )

    new_cell = np.array(atoms.cell)
    # the optimizer moves atoms continuously (no wrapping), so the
    # fractional displacement is direct; applying it to the unwrapped
    # graph coordinates keeps bonded atoms cartesian neighbors
    displacement = atoms.get_scaled_positions(wrap=False) - initial_frac
    unwrapped_frac = framework.cart_coords @ np.linalg.inv(initial_cell)
    new_coords = (unwrapped_frac + displacement) @ new_cell

    moved = np.linalg.norm(
        (displacement @ new_cell),
        axis=1,
    )
    logger.info(
        f"Relaxed {framework.name!r} with {type(calculator).__name__}: "
        f"energy {energy_ev * EV_TO_KCAL_PER_MOL:.1f} kcal/mol per "
        f"cell, max atom displacement {moved.max():.2f} A."
    )

    # nodes first, in sorted order, then edges: same insertion order
    # as every builder graph, so edge iteration matches the input
    # exactly (#145)
    relaxed = networkx.Graph(cell=new_cell)
    for row, node in enumerate(sorted(framework.graph)):
        copied = dict(framework.graph.nodes[node])
        copied["coord"] = new_coords[row]
        relaxed.add_node(node, **copied)
    relaxed.add_edges_from(framework.graph.edges(data=True))

    from autografs.framework import Framework as FrameworkCls

    result = FrameworkCls(relaxed, name=framework.name)
    result.energy = energy_ev * EV_TO_KCAL_PER_MOL
    return result
