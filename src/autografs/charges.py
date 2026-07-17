"""
Partial-charge assignment for built frameworks.

GCMC adsorption (and anything else electrostatic) needs point charges,
and the charge scheme is a controlled variable of the property funnel,
not an afterthought: EQeq systematically differs from DDEC, which
differs from ML charges. This module provides the schemes behind
``Framework.assign_charges`` through a small registry, with EQeq as
the built-in baseline; DDEC (from DFT output) or ML models plug in by
registering a callable.

EQeq — the extended charge equilibration method of Wilmer, Kim &
Snurr (J. Phys. Chem. Lett. 2012, 3, 2506-2511) — assigns charges by
minimizing an electrostatic energy whose per-element electronegativity
and hardness come from measured ionization energies (Taylor-expanded
around non-neutral charge centers for common framework metals), with
periodic Coulomb interactions summed exactly as in the reference
implementation: a damped real-space sum plus a reciprocal-space
correction over fixed cell ranges, and a short-range orbital-overlap
correction. No iteration, one linear solve. This is an independent
numpy port of the published method, validated against the reference
C++ code to the digits it prints.

Charges live on the bond graph (node attribute ``charge``), survive
``Framework.save``/``load`` and every editing operation, and flow into
the exports that can carry them (CIF ``_atom_site_charge``, ASE
initial charges, GULP).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import erfc

from autografs.data.eqeq import CHARGE_CENTERS, IONIZATION_ENERGIES

if TYPE_CHECKING:
    from autografs.framework import Framework

logger = logging.getLogger(__name__)

__all__ = [
    "CHARGE_METHODS",
    "assign_charges",
    "eqeq_charges",
    "register_charge_method",
]

# Coulomb constant 1/(4 pi eps0) in eV * Angstrom
COULOMB_K = 14.4

ChargeMethod = Callable[..., np.ndarray]

CHARGE_METHODS: dict[str, ChargeMethod] = {}


def register_charge_method(name: str) -> Callable[[ChargeMethod], ChargeMethod]:
    """Register a charge scheme under ``name`` (decorator).

    The callable receives the framework as its first argument plus any
    keyword arguments the user passes through ``assign_charges``, and
    returns one charge per atom in sorted-node order.
    """

    def decorator(func: ChargeMethod) -> ChargeMethod:
        CHARGE_METHODS[name] = func
        return func

    return decorator


def _element_parameters(
    symbols: list[str], hydrogen_affinity: float
) -> tuple[np.ndarray, np.ndarray]:
    """Electronegativity chi and hardness J (eV) for each atom.

    For an element centered at charge n, chi and J are the finite
    differences of the ionization-energy ladder around n; centering at
    a typical oxidation state (CHARGE_CENTERS) linearizes the energy
    where the atom actually sits. Hydrogen's electron affinity is the
    method's one empirical dial (``hydrogen_affinity``).
    """
    chi = np.empty(len(symbols))
    hardness = np.empty(len(symbols))
    for i, symbol in enumerate(symbols):
        if symbol == "H":
            ionization = IONIZATION_ENERGIES["H"][1]
            assert ionization is not None
            chi[i] = 0.5 * (ionization + hydrogen_affinity)
            hardness[i] = ionization - hydrogen_affinity
            continue
        try:
            ladder = IONIZATION_ENERGIES[symbol]
        except KeyError:
            raise ValueError(
                f"No EQeq ionization data for element {symbol!r}."
            ) from None
        center = CHARGE_CENTERS.get(symbol, 0)
        # the reference implementation treats unmeasured values as 0.0
        low = ladder[center] or 0.0
        high = ladder[center + 1] or 0.0
        chi[i] = 0.5 * (high + low) - center * (high - low)
        hardness[i] = high - low
    return chi, hardness


def _interaction_matrix(
    coords: np.ndarray,
    cell: np.ndarray,
    hardness: np.ndarray,
    scaling: float,
    eta: float,
    m_real: int,
    m_reciprocal: int,
) -> np.ndarray:
    """The periodic EQeq interaction matrix (eV per unit charge^2).

    Off-diagonal entries are the energy coupling of unit charges on
    atoms i and j; diagonal entries carry the atomic hardness plus the
    self-interaction with a charge's own periodic images. Sums follow
    the reference implementation exactly: real-space erfc-damped
    Coulomb and orbital-overlap terms over (2 m_real + 1)^3 cells,
    reciprocal-space term over (2 m_reciprocal + 1)^3 - 1 vectors, and
    the constant self-term -2/(eta sqrt(pi)).
    """
    n = len(coords)
    delta = coords[:, None, :] - coords[None, :, :]
    overlap_decay = np.sqrt(np.outer(hardness, hardness)) / COULOMB_K

    real_sum = np.zeros((n, n))
    orbital_sum = np.zeros((n, n))
    span = range(-m_real, m_real + 1)
    diagonal = np.eye(n, dtype=bool)
    for u in span:
        for v in span:
            for w in span:
                shift = u * cell[0] + v * cell[1] + w * cell[2]
                distance = np.linalg.norm(delta + shift, axis=2)
                # a charge does not interact with itself in the home
                # cell; mask instead of skipping so i != j pairs keep
                # their home-cell term
                excluded = diagonal & (u == 0 and v == 0 and w == 0)
                safe = np.where(excluded, 1.0, distance)
                real_sum += np.where(excluded, 0.0, erfc(safe / eta) / safe)
                overlap = np.exp(-((overlap_decay * safe) ** 2)) * (
                    2.0 * overlap_decay - overlap_decay**2 * safe - 1.0 / safe
                )
                orbital_sum += np.where(excluded, 0.0, overlap)

    volume = float(abs(np.linalg.det(cell)))
    reciprocal_cell = 2.0 * np.pi * np.linalg.inv(cell).T
    reciprocal_sum = np.zeros((n, n))
    span_k = range(-m_reciprocal, m_reciprocal + 1)
    for u in span_k:
        for v in span_k:
            for w in span_k:
                if u == 0 and v == 0 and w == 0:
                    continue
                vector = (
                    u * reciprocal_cell[0]
                    + v * reciprocal_cell[1]
                    + w * reciprocal_cell[2]
                )
                magnitude = float(np.linalg.norm(vector))
                envelope = np.exp(-((0.5 * magnitude * eta) ** 2)) / magnitude**2
                reciprocal_sum += np.cos(delta @ vector) * envelope
    reciprocal_sum *= 4.0 * np.pi / volume

    matrix = scaling * (COULOMB_K / 2.0) * (real_sum + reciprocal_sum + orbital_sum)
    matrix[diagonal] += hardness - scaling * COULOMB_K / (eta * np.sqrt(np.pi))
    return matrix


def _eqeq_solve(
    symbols: list[str],
    coords: np.ndarray,
    cell: np.ndarray,
    scaling: float,
    hydrogen_affinity: float,
    eta: float,
    m_real: int,
    m_reciprocal: int,
    total_charge: float,
) -> np.ndarray:
    """EQeq on raw arrays: one KKT solve for the equilibrated charges.

    Minimizes chi.Q + 1/2 Q.M.Q subject to sum(Q) = total_charge - the
    electronegativity-equalization conditions the reference code
    expresses as consecutive row differences, here solved with an
    explicit Lagrange multiplier (same solution).
    """
    chi, hardness = _element_parameters(symbols, hydrogen_affinity)
    matrix = _interaction_matrix(
        coords, cell, hardness, scaling, eta, m_real, m_reciprocal
    )
    n = len(symbols)
    system = np.zeros((n + 1, n + 1))
    system[:n, :n] = matrix
    system[:n, n] = 1.0
    system[n, :n] = 1.0
    rhs = np.concatenate([-chi, [total_charge]])
    return np.asarray(np.linalg.solve(system, rhs)[:n])


@register_charge_method("eqeq")
def eqeq_charges(
    framework: Framework,
    scaling: float = 1.2,
    hydrogen_affinity: float = -2.0,
    eta: float = 50.0,
    m_real: int = 2,
    m_reciprocal: int = 2,
    total_charge: float = 0.0,
) -> np.ndarray:
    """EQeq partial charges for every atom of a framework.

    Parameters
    ----------
    framework : Framework
        The framework to charge.
    scaling : float, optional
        Coulomb scaling (dielectric screening) parameter lambda, by
        default 1.2 (the published value).
    hydrogen_affinity : float, optional
        Effective electron affinity of hydrogen (eV), by default -2.0
        (the published value).
    eta : float, optional
        Damping length of the periodic summation (Angstrom), by
        default 50.0, matching the reference implementation.
    m_real, m_reciprocal : int, optional
        Real- and reciprocal-space cell ranges of the sums, by default
        2 (i.e. 5x5x5 cells), matching the reference implementation.
    total_charge : float, optional
        Net charge of the unit cell, by default 0.0.

    Returns
    -------
    np.ndarray
        One charge per atom, in sorted-node order; sums to
        ``total_charge`` exactly.
    """
    nodes = sorted(framework.graph)
    symbols = [framework.graph.nodes[n]["symbol"] for n in nodes]
    return _eqeq_solve(
        symbols,
        framework.cart_coords,
        np.asarray(framework.cell, dtype=float),
        scaling,
        hydrogen_affinity,
        eta,
        m_real,
        m_reciprocal,
        total_charge,
    )


def assign_charges(framework: Framework, method: str = "eqeq", **kwargs) -> Framework:
    """A new framework with per-atom charges on its bond graph.

    Parameters
    ----------
    framework : Framework
        The framework to charge; not modified.
    method : str, optional
        A key of ``CHARGE_METHODS``, by default "eqeq".
    **kwargs
        Passed through to the scheme (see e.g. ``eqeq_charges``).

    Returns
    -------
    Framework
        A copy whose graph nodes carry a ``charge`` attribute and
        whose graph records the scheme in ``charge_method``.

    Raises
    ------
    ValueError
        If the method is unknown or returns the wrong number of
        charges.
    """
    try:
        scheme = CHARGE_METHODS[method]
    except KeyError:
        available = ", ".join(sorted(CHARGE_METHODS))
        raise ValueError(
            f"Unknown charge method {method!r}; available: {available}."
        ) from None
    charges = np.asarray(scheme(framework, **kwargs), dtype=float)
    if charges.shape != (len(framework),):
        raise ValueError(
            f"Charge method {method!r} returned {charges.shape} values "
            f"for {len(framework)} atoms."
        )
    from autografs.framework import Framework as FrameworkCls

    graph = framework.graph.copy()
    for node, charge in zip(sorted(graph), charges, strict=True):
        graph.nodes[node]["charge"] = float(charge)
    graph.graph["charge_method"] = method
    result = FrameworkCls(graph, name=framework.name)
    result.energy = framework.energy
    logger.info(
        f"Assigned {method} charges to {framework.name!r}: "
        f"range [{charges.min():+.3f}, {charges.max():+.3f}] e."
    )
    return result
