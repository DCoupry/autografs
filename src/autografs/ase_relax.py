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

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from autografs.exceptions import RelaxationError
from autografs.relax import _quiet, _with_new_geometry

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

    from autografs.framework import Framework

logger = logging.getLogger(__name__)

EV_TO_KCAL_PER_MOL = 23.060548

# calculator names the by-name dispatch accepts; the backends behind
# them are imported lazily, with install hints on failure
_KNOWN_CALCULATORS = ("gfn-ff", "gfn1", "gfn2", "dftb")


def _periodic_gfnff(xtb_cls: type, **kwargs: Any) -> Calculator:
    """GFN-FF that survives a periodic call, at the cost of the stress.

    xtb's ASE calculator fetches the virial for *any* periodic system,
    and GFN-FF does not produce one (only GFN0-xTB does), so every
    periodic call raises ``XTBException: Virial is not available``
    before returning the energy it had already computed. Energy and
    forces ARE available, so this keeps them and drops the stress -
    which makes fixed-cell relaxation work and leaves cell relaxation
    correctly impossible (``relax_framework_ase`` refuses it rather
    than silently pretending).

    Caveat worth knowing before choosing this backend: xtb's GFN-FF
    also crashes the interpreter outright on larger periodic cells
    (measured on this corpus: hard exits with ACCESS_VIOLATION at 76
    atoms and STACK_OVERFLOW from ~348 atoms up, which no Python-level
    handling can catch). GFN1-xTB via tblite has none of these
    problems and does provide stress.
    """
    from ase.calculators.calculator import all_changes

    class PeriodicGFNFF(xtb_cls):  # type: ignore[valid-type, misc]
        # no "stress": the cell cannot be relaxed with this method
        implemented_properties = ["energy", "free_energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            try:
                super().calculate(atoms, properties or ["energy"], system_changes)
            except Exception:
                # the virial fetch is the last thing xtb's calculator
                # does; anything earlier failing is a real error
                if "energy" not in self.results or "forces" not in self.results:
                    raise
            self.results.pop("stress", None)

    return PeriodicGFNFF(method="GFNFF", **kwargs)  # type: ignore[no-any-return]


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
        return _periodic_gfnff(XTB, **kwargs)
    if key in ("gfn1", "gfn1-xtb"):
        try:
            from tblite.ase import TBLite
        except ImportError as exc:
            raise RelaxationError("GFN1-xTB needs tblite: pip install tblite") from exc
        return TBLite(method="GFN1-xTB", **kwargs)  # type: ignore[no-any-return]
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

    if relax_cell and "stress" not in getattr(
        calculator, "implemented_properties", ("stress",)
    ):
        # FrechetCellFilter would ask for a stress the method cannot
        # give and fail deep inside the optimizer; say so up front
        raise RelaxationError(
            f"{type(calculator).__name__} provides no stress, so the cell "
            "cannot be relaxed with it (GFN-FF has no periodic virial - "
            "only GFN0-xTB does). Pass relax_cell=False to relax the "
            "positions at fixed cell, or use 'gfn1' (tblite), which does "
            "provide stress."
        )
    target = FrechetCellFilter(atoms) if relax_cell else atoms
    quiet = _quiet(verbose)
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

    return _with_new_geometry(
        framework, new_coords, new_cell, energy=energy_ev * EV_TO_KCAL_PER_MOL
    )
