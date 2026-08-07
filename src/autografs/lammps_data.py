"""LAMMPS data files written from a Framework's own bond graph.

``relax.py`` stages through lammps-interface, which re-derives the
topology from geometry. That step is where the failures live: over 516
generative candidates it lost 210 of them - 72 "Invalid atom ID in
Dihedrals section", 48 C-level aborts, 45 RecursionError, 43 silently
collapsed - none of which is a force-field problem.

None of that derivation is necessary here. A built Framework already
knows its bonds exactly (they come from the blueprint's dummy
correspondences, not a distance criterion) and its UFF4MOF atom types
(``utils.find_mmtypes``, from real connectivity). Everything a data file
needs beyond that is a deterministic walk of that graph plus the
published UFF combination rules. So this module writes the file itself.

What it does NOT reimplement is the parameter table: the 11 numbers per
atom type are literature values (Rappe 1992; Addicoat 2014; Coupry
2016), read from the MIT-licensed lammps-interface tables so there is
one transcription rather than two.

Scope and conventions
---------------------
* Topology is graph paths, not geometry. Bonds are the graph's edges;
  angles every i-j-k, dihedrals every i-j-k-l, impropers every
  3-coordinate centre of an inversion-active element. No periodic image
  bookkeeping appears in the file: LAMMPS resolves bonded interactions
  by minimum image, which is the same convention the graph already uses,
  so the one requirement is that no bond exceed half the shortest box
  length (checked, and raised as a RelaxationError rather than left to
  produce nonsense).
* Styles match lammps-interface's so the two are directly comparable:
  ``bond_style harmonic``, ``angle_style fourier``/``cosine/periodic``,
  ``dihedral_style harmonic``, ``improper_style fourier``,
  ``pair_style lj/cut`` with geometric mixing.
* Charges are written when the framework carries them (``assign_charges``)
  and zero otherwise; electrostatics are the caller's choice of pair
  style, not this writer's.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from autografs.exceptions import RelaxationError

if TYPE_CHECKING:
    from autografs.framework import Framework

logger = logging.getLogger(__name__)

__all__ = [
    "LammpsData",
    "enumerate_topology",
    "write_lammps_data",
]

#: UFF's bond-order correction constant (Rappe 1992, eq. 3).
LAMBDA_BOND_ORDER = 0.1332
#: Converts UFF's Z* products into kcal/mol/A^2 (Rappe 1992, eq. 6).
UFF_FORCE_CONSTANT = 664.12
#: LJ minimum-to-sigma conversion: x_i is the distance at the minimum.
SIGMA_FROM_XI = 2.0 ** (-1.0 / 6.0)

#: Elements whose 3-coordinate centres get a UFF inversion term.
_INVERSION_ELEMENTS = {"C", "N", "O", "P", "As", "Sb", "Bi"}


@dataclass
class LammpsData:
    """A written data file and the styles that go with it."""

    text: str
    n_atoms: int
    n_bonds: int
    n_angles: int
    n_dihedrals: int
    n_impropers: int
    styles: dict[str, str] = field(default_factory=dict)
    #: explicit ``pair_coeff i i eps sigma`` lines, so a caller can swap
    #: to a soft push-off style and restore these afterwards
    pair_coeffs: tuple[str, ...] = ()
    #: closest atom pair in the input, before any relaxation
    closest_contact: float = float("inf")
    #: types the parameter table did not cover, left out of the file
    missing_types: tuple[str, ...] = ()

    def __str__(self) -> str:
        return (
            f"LammpsData({self.n_atoms} atoms, {self.n_bonds} bonds, "
            f"{self.n_angles} angles, {self.n_dihedrals} dihedrals, "
            f"{self.n_impropers} impropers)"
        )


# --------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------

_FIELDS = (
    "r1",  # single-bond radius, A
    "theta0",  # ideal valence angle, degrees
    "x1",  # vdW distance at the minimum, A
    "D1",  # vdW well depth, kcal/mol
    "zeta",  # vdW scaling (unused by lj/cut)
    "Z1",  # effective charge
    "Vi",  # sp3 torsional barrier
    "Uj",  # sp2 torsional barrier
    "Xi",  # GMP electronegativity
    "hard",  # GMP hardness (unused here)
    "radius",  # GMP radius (unused here)
)


def uff_parameters(force_field: str = "UFF4MOF") -> dict[str, dict[str, float]]:
    """The published per-type parameters, keyed by atom type.

    Read from lammps-interface's tables (MIT) rather than transcribed a
    second time: these are literature values and a duplicate copy is a
    second place for a typo to hide.
    """
    try:
        if "4mof" in force_field.lower():
            from lammps_interface.uff4mof import UFF4MOF_DATA as table
        else:
            from lammps_interface.uff import UFF_DATA as table
    except ImportError as exc:  # pragma: no cover - optional backend
        raise RelaxationError(
            "Writing a UFF data file needs the parameter tables: "
            'pip install "autografs[relax]".'
        ) from exc
    return {
        symbol: dict(zip(_FIELDS, values, strict=False))
        for symbol, values in table.items()
    }


def _hybridisation(uff_type: str) -> int:
    """Coordination implied by a UFF type's third character.

    UFF encodes it: ``C_3`` sp3, ``C_2`` sp2, ``C_R`` resonant, ``C_1``
    sp. Metals carry a coordination digit instead (``Zn3+2`` is
    tetrahedral), which the torsion rules treat as sp3-like.
    """
    if len(uff_type) < 3:
        return 0
    marker = uff_type[2]
    if marker == "R":
        return 2
    if marker.isdigit():
        return int(marker)
    return 0


# --------------------------------------------------------------------
# topology, from the graph alone
# --------------------------------------------------------------------


def enumerate_topology(graph) -> dict[str, list[tuple]]:
    """Bonds, angles, dihedrals and impropers as index tuples.

    Pure graph walking - no geometry, no periodic images. LAMMPS applies
    the minimum image convention to bonded terms, which is exactly the
    convention the framework graph already uses, so a bond that crosses
    a cell boundary needs no special treatment here.
    """
    order = {node: i for i, node in enumerate(sorted(graph))}
    neighbours = {n: sorted(graph.neighbors(n)) for n in graph}

    bonds = [
        (order[a], order[b], float(data.get("bond_order", 1.0)))
        for a, b, data in graph.edges(data=True)
    ]

    angles: list[tuple[int, int, int]] = []
    for centre, around in neighbours.items():
        for left, right in itertools.combinations(around, 2):
            angles.append((order[left], order[centre], order[right]))

    dihedrals: list[tuple[int, int, int, int]] = []
    for b, c in graph.edges():
        for a in neighbours[b]:
            if a == c:
                continue
            for d in neighbours[c]:
                if d == b or d == a:
                    continue
                dihedrals.append((order[a], order[b], order[c], order[d]))

    impropers: list[tuple[int, int, int, int]] = []
    for centre, around in neighbours.items():
        if len(around) != 3:
            continue
        if graph.nodes[centre].get("symbol") not in _INVERSION_ELEMENTS:
            continue
        i, j, k = (order[n] for n in around)
        impropers.append((order[centre], i, j, k))

    return {
        "bonds": bonds,
        "angles": angles,
        "dihedrals": dihedrals,
        "impropers": impropers,
    }


# --------------------------------------------------------------------
# UFF combination rules
# --------------------------------------------------------------------


def uff_bond_order(type_i: str, type_j: str, perceived: float) -> float:
    """The bond order UFF's rules use, not the one perception measured.

    UFF works from a discrete order - 1.5 for a resonant bond, 1.41 for
    an amide, integers otherwise - while our graph carries *perceived*
    fractional values (1.736, 1.803, ...). Feeding those straight in
    shifted every aromatic r0: C_R-C_R came out at 1.351 A against the
    1.379 A UFF specifies.

    Resonance cannot be read off the types alone, though. MOF-5's BDC
    has two C_R-C_R bonds - the ring's (perceived 1.74, resonant) and
    the ring-to-carboxylate single bond (perceived 1.0, not) - and its
    carboxylate C_R-O_2 is resonant despite O_2 carrying no ``_R``. So
    the type says whether resonance is *possible* and the perceived
    order says whether it is *realised*. With this rule every bond
    coefficient matches lammps-interface's to six decimals.
    """
    resonant = type_i.endswith("_R") or type_j.endswith("_R")
    if resonant and perceived > 1.25:
        if {type_i, type_j} == {"C_R", "N_R"}:
            return 1.41  # amide
        return 1.5
    return float(max(1.0, round(perceived)))


def natural_bond_length(pi: dict, pj: dict, order: float) -> float:
    """UFF equilibrium bond length, with bond-order and EN corrections."""
    ri, rj = pi["r1"], pj["r1"]
    rbo = -LAMBDA_BOND_ORDER * (ri + rj) * math.log(max(order, 1e-6))
    chi_i, chi_j = pi["Xi"], pj["Xi"]
    denominator = chi_i * ri + chi_j * rj
    ren = (
        ri * rj * (math.sqrt(chi_i) - math.sqrt(chi_j)) ** 2 / denominator
        if denominator
        else 0.0
    )
    # Rappe's eq. 2 as published sums the corrections; the widely used
    # implementations subtract r_EN, which is what reproduces the
    # tabulated bond lengths - follow the implementations.
    return float(ri + rj + rbo - ren)


def bond_coefficients(pi: dict, pj: dict, order: float) -> tuple[float, float]:
    """(K, r0) for ``bond_style harmonic`` (E = K (r - r0)^2)."""
    r0 = natural_bond_length(pi, pj, order)
    force = UFF_FORCE_CONSTANT * pi["Z1"] * pj["Z1"] / r0**3
    return force / 2.0, r0


def angle_coefficients(
    pi: dict, pj: dict, pk: dict, order_ij: float, order_jk: float
) -> tuple[str, tuple[float, ...]]:
    """Angle term for the central type's ideal geometry.

    Linear, trigonal-planar, square-planar and octahedral centres get
    UFF's periodic form (``cosine/periodic``); everything else the
    three-term cosine expansion (``fourier``).
    """
    theta0 = math.radians(pj["theta0"])
    rij = natural_bond_length(pi, pj, order_ij)
    rjk = natural_bond_length(pj, pk, order_jk)
    rik = math.sqrt(rij**2 + rjk**2 - 2.0 * rij * rjk * math.cos(theta0))
    force = (
        UFF_FORCE_CONSTANT
        * pi["Z1"]
        * pk["Z1"]
        / rik**5
        * rij
        * rjk
        * (3.0 * rij * rjk * (1.0 - math.cos(theta0) ** 2) - rik**2 * math.cos(theta0))
    )

    degrees = round(pj["theta0"], 1)
    periodic = {180.0: 1, 120.0: 3, 90.0: 4}
    if degrees in periodic:
        n = periodic[degrees]
        # LAMMPS cosine/periodic: E = C[1 - B(-1)^n cos(n theta)]
        # UFF:                    E = (K/n^2)[1 - cos(n theta_0) cos(n theta)]
        b = 1 if degrees == 180.0 else -1
        return "cosine/periodic", (force / n**2 * 2.0, b, n)

    sin2 = math.sin(theta0) ** 2
    if sin2 < 1e-8:  # pragma: no cover - guarded by the table above
        return "cosine/periodic", (force * 2.0, 1, 1)
    c2 = 1.0 / (4.0 * sin2)
    c1 = -4.0 * c2 * math.cos(theta0)
    c0 = c2 * (2.0 * math.cos(theta0) ** 2 + 1.0)
    return "fourier", (force, c0, c1, c2)


def dihedral_coefficients(
    pj: dict, pk: dict, tj: str, tk: str, order_jk: float, multiplicity: int = 1
) -> tuple[float, int, int] | None:
    """(K, d, n) for ``dihedral_style harmonic`` (E = K[1 + d cos(n phi)]).

    UFF's rules turn on the hybridisation of the two *central* atoms;
    returns None where the pair carries no torsional term (a metal or a
    terminal centre), which is normal rather than an error.

    ``multiplicity`` is the number of torsions about that same j-k bond:
    UFF states one barrier for the bond and divides it among them, so
    omitting the division overcounts by exactly that factor (measured on
    MOF-5: 4x, benzene's C_R-C_R carrying four torsions each).
    """
    hj, hk = _hybridisation(tj), _hybridisation(tk)
    if hj == 3 and hk == 3:
        barrier = math.sqrt(pj["Vi"] * pk["Vi"])
        n, phi0 = 3, math.pi
    elif hj == 2 and hk == 2:
        barrier = (
            5.0
            * math.sqrt(pj["Uj"] * pk["Uj"])
            * (1.0 + 4.18 * math.log(max(order_jk, 1e-6)))
        )
        n, phi0 = 2, math.pi
    elif {hj, hk} == {2, 3}:
        barrier, n, phi0 = 1.0, 6, 0.0
    else:
        return None
    if barrier <= 0.0:
        return None
    # E = (V/2)[1 - cos(n phi_0) cos(n phi)]  ->  K = V/2, d = -cos(n phi_0)
    d = -1 if math.cos(n * phi0) > 0 else 1
    return barrier / (2.0 * max(multiplicity, 1)), d, n


def improper_coefficients(symbol: str) -> tuple[float, float, float, float] | None:
    """(K, C0, C1, C2) for ``improper_style fourier``.

    UFF's inversion barrier: 6 kcal/mol for a resonant/sp2 carbon or
    nitrogen centre, 22 for the group-15 pyramidal centres.
    """
    if symbol in ("C", "N"):
        force, c0, c1, c2 = 6.0 / 3.0, 1.0, -1.0, 0.0
    elif symbol in ("P", "As", "Sb", "Bi"):
        force, c0, c1, c2 = 22.0 / 3.0, 1.0, -1.0, 0.0
    elif symbol == "O":
        force, c0, c1, c2 = 6.0 / 3.0, 1.0, -1.0, 0.0
    else:  # pragma: no cover - filtered by _INVERSION_ELEMENTS
        return None
    return force, c0, c1, c2


def pair_coefficients(pi: dict) -> tuple[float, float]:
    """(epsilon, sigma) for ``pair_style lj/cut``; geometric mixing."""
    return pi["D1"], pi["x1"] * SIGMA_FROM_XI


# --------------------------------------------------------------------
# the writer
# --------------------------------------------------------------------


def _angle_style(styles: set[str]) -> str:
    """Declare hybrid only when both sub-styles are really used.

    LAMMPS rejects a hybrid whose sub-style no Angle Coeffs line
    references ("Angle hybrid sub-style fourier is not used"), so a
    framework with only linear/square-planar centres must ask for
    cosine/periodic alone.
    """
    ordered = sorted(styles)
    if len(ordered) == 1:
        return ordered[0]
    return "hybrid " + " ".join(ordered)


def _triclinic_box(cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LAMMPS lower-triangular cell, and the rotation that produced it.

    LAMMPS requires a = (xhi,0,0), b = (xy,yhi,0), c = (xz,yz,zhi), so a
    general cell is re-expressed in that frame by QR; positions are
    rotated with it, which is a rigid motion and changes no energy.
    """
    cell = np.asarray(cell, dtype=float)
    q, r = np.linalg.qr(cell.T)
    # fix signs so the diagonal is positive
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    r = r * signs[:, None]
    q = q * signs[None, :]
    return r.T, q


def write_lammps_data(
    framework: Framework,
    force_field: str = "UFF4MOF",
    path: str | None = None,
) -> LammpsData:
    """Write a complete LAMMPS data file for a built framework.

    Parameters
    ----------
    framework : Framework
        A built framework; its bond graph is taken as authoritative.
    force_field : str, optional
        "UFF4MOF" (default) or "UFF" - selects the parameter table.
    path : str or None, optional
        Write the text there as well as returning it.

    Returns
    -------
    LammpsData
        The file text, the section counts, and the styles it needs.

    Raises
    ------
    RelaxationError
        If a bond is longer than half the shortest box length (LAMMPS
        would resolve it through the wrong image), or if an atom's type
        has no parameters.
    """
    graph = framework.graph
    table = uff_parameters(force_field)
    nodes = sorted(graph)
    types = [graph.nodes[n].get("ufftype") for n in nodes]

    # the same translation relax.py applies before staging: our table
    # carries 15 symbols the parameter set does not (Co6+2, N_3+4, ...)
    from autografs.relax import _substitution_map

    substitution = _substitution_map(force_field)
    substituted: list[str] = []
    for row, uff_type in enumerate(types):
        if uff_type in table:
            continue
        target = substitution.get(uff_type)
        if target is None or target not in table:
            raise RelaxationError(
                f"No {force_field} parameters for atom type {uff_type!r}, "
                "and no same-element substitute exists."
            )
        substituted.append(f"{uff_type}->{target}")
        types[row] = target
    if substituted:
        logger.info(
            f"Substituted atom types unsupported by {force_field} for "
            f"{framework.name!r}: {sorted(set(substituted))}."
        )

    cell, rotation = _triclinic_box(np.asarray(graph.graph["cell"], dtype=float))
    coords = np.asarray(framework.cart_coords, dtype=float) @ rotation
    # the graph stores UNWRAPPED cartesians, which can sit far outside
    # the box; LAMMPS then cannot find a bonded partner in its ghost
    # shell ("Bond atoms N M missing on proc 0"). Wrap into the cell and
    # let the minimum-image convention resolve the bonds, which is the
    # convention the graph already uses.
    fractional = coords @ np.linalg.inv(cell)
    fractional -= np.floor(fractional)
    fractional[fractional >= 1.0] -= 1.0
    coords = fractional @ cell
    lengths = np.linalg.norm(cell, axis=1)

    topology = enumerate_topology(graph)

    # a bonded pair further apart than half the box would be resolved
    # through the wrong image by LAMMPS's minimum-image convention
    inverse = np.linalg.inv(cell)
    longest_bond = 0.0
    widest_wrapped = 0.0
    for a, b, _order in topology["bonds"]:
        delta = coords[a] - coords[b]
        delta -= np.round(delta @ inverse) @ cell
        longest_bond = max(longest_bond, float(np.linalg.norm(delta)))
        # the separation LAMMPS actually sees: once wrapped, a bond that
        # crosses a boundary has its two atoms at opposite ends of the
        # box, and the ghost shell must reach that far or the bond is
        # dropped at setup ("Bond atoms N M missing on proc 0")
        widest_wrapped = max(
            widest_wrapped, float(np.linalg.norm(coords[a] - coords[b]))
        )
        if np.linalg.norm(delta) > 0.5 * lengths.min():
            raise RelaxationError(
                f"Bond {a}-{b} spans {np.linalg.norm(delta):.2f} A, over half "
                f"the shortest box length ({lengths.min():.2f} A); LAMMPS "
                "would bond it through the wrong image. Use a supercell."
            )

    # ---- unique coefficient sets ---------------------------------
    atom_types = sorted(set(types))
    atom_type_id = {t: i + 1 for i, t in enumerate(atom_types)}

    bond_types: dict[tuple, int] = {}
    bond_rows: list[tuple[int, int, int]] = []
    for a, b, perceived in topology["bonds"]:
        order = uff_bond_order(types[a], types[b], perceived)
        key = (*sorted((types[a], types[b])), round(order, 3))
        bond_types.setdefault(key, len(bond_types) + 1)
        bond_rows.append((bond_types[key], a, b))

    angle_styles: set[str] = {"fourier"}
    angle_types: dict[tuple, int] = {}
    angle_rows: list[tuple[int, int, int, int]] = []
    orders = {
        tuple(sorted((a, b))): uff_bond_order(types[a], types[b], o)
        for a, b, o in topology["bonds"]
    }
    for i, j, k in topology["angles"]:
        oij = orders.get(tuple(sorted((i, j))), 1.0)
        ojk = orders.get(tuple(sorted((j, k))), 1.0)
        ends = sorted((types[i], types[k]))
        key = (ends[0], types[j], ends[1], round(oij, 3), round(ojk, 3))
        angle_types.setdefault(key, len(angle_types) + 1)
        angle_rows.append((angle_types[key], i, j, k))

    dihedral_types: dict[tuple, int] = {}
    dihedral_rows: list[tuple[int, int, int, int, int]] = []
    # UFF states ONE barrier per j-k bond and divides it among the
    # torsions about that bond; without the division MOF-5's benzene
    # overcounted by exactly 4x, its C_R-C_R carrying four each
    torsions_about = Counter(
        tuple(sorted((b, c))) for _a, b, c, _d in topology["dihedrals"]
    )
    for a, b, c, d in topology["dihedrals"]:
        obc = orders.get(tuple(sorted((b, c))), 1.0)
        multiplicity = torsions_about[tuple(sorted((b, c)))]
        key = (types[b], types[c], round(obc, 3), multiplicity)
        if (
            dihedral_coefficients(
                table[types[b]], table[types[c]], types[b], types[c], obc, multiplicity
            )
            is None
        ):
            continue
        dihedral_types.setdefault(key, len(dihedral_types) + 1)
        dihedral_rows.append((dihedral_types[key], a, b, c, d))

    improper_types: dict[str, int] = {}
    improper_rows: list[tuple[int, int, int, int, int]] = []
    for centre, i, j, k in topology["impropers"]:
        symbol = graph.nodes[nodes[centre]]["symbol"]
        improper_types.setdefault(symbol, len(improper_types) + 1)
        improper_rows.append((improper_types[symbol], centre, i, j, k))

    # ---- assemble -------------------------------------------------
    charges = framework.charges
    lines: list[str] = [
        f"LAMMPS data file for {framework.name!r}, {force_field}, "
        "written by autografs.lammps_data",
        "",
        f"{len(nodes)} atoms",
        f"{len(bond_rows)} bonds",
        f"{len(angle_rows)} angles",
        f"{len(dihedral_rows)} dihedrals",
        f"{len(improper_rows)} impropers",
        "",
        f"{len(atom_types)} atom types",
        f"{len(bond_types)} bond types",
        f"{len(angle_types)} angle types",
        f"{len(dihedral_types)} dihedral types",
        f"{len(improper_types)} improper types",
        "",
        f"0.0 {cell[0, 0]:.8f} xlo xhi",
        f"0.0 {cell[1, 1]:.8f} ylo yhi",
        f"0.0 {cell[2, 2]:.8f} zlo zhi",
        f"{cell[1, 0]:.8f} {cell[2, 0]:.8f} {cell[2, 1]:.8f} xy xz yz",
        "",
        "Masses",
        "",
    ]
    from ase.data import atomic_masses, atomic_numbers

    for uff_type in atom_types:
        element = uff_type[:2].rstrip("_0123456789+f")
        mass = atomic_masses[atomic_numbers.get(element, 1)]
        lines.append(f"{atom_type_id[uff_type]} {mass:.6f} # {uff_type}")

    lines += ["", "Pair Coeffs", ""]
    for uff_type in atom_types:
        epsilon, sigma = pair_coefficients(table[uff_type])
        lines.append(f"{atom_type_id[uff_type]} {epsilon:.6f} {sigma:.6f} # {uff_type}")

    if bond_types:
        lines += ["", "Bond Coeffs", ""]
        for key, index in sorted(bond_types.items(), key=lambda kv: kv[1]):
            ti, tj, order = key
            force, r0 = bond_coefficients(table[ti], table[tj], order)
            lines.append(f"{index} {force:.6f} {r0:.6f} # {ti}-{tj} bo={order}")

    if angle_types:
        # the sub-styles have to be known before the lines are written:
        # LAMMPS wants the style name on each Angle Coeffs line only when
        # the declared style is `hybrid`
        resolved = {
            index: angle_coefficients(
                table[key[0]], table[key[1]], table[key[2]], key[3], key[4]
            )
            for key, index in angle_types.items()
        }
        angle_styles = {style for style, _values in resolved.values()}
        hybrid = len(angle_styles) > 1
        lines += ["", "Angle Coeffs", ""]
        for key, index in sorted(angle_types.items(), key=lambda kv: kv[1]):
            ti, tj, tk, _oij, _ojk = key
            style, values = resolved[index]
            numbers = " ".join(
                f"{v:.6f}" if isinstance(v, float) else str(v) for v in values
            )
            prefix = f"{style} " if hybrid else ""
            lines.append(f"{index} {prefix}{numbers} # {ti}-{tj}-{tk}")

    if dihedral_types:
        lines += ["", "Dihedral Coeffs", ""]
        for key, index in sorted(dihedral_types.items(), key=lambda kv: kv[1]):
            tj, tk, order, multiplicity = key
            coefficients = dihedral_coefficients(
                table[tj], table[tk], tj, tk, order, multiplicity
            )
            assert coefficients is not None
            force, d, n = coefficients
            lines.append(f"{index} {force:.6f} {d} {n} # {tj}-{tk} x{multiplicity}")

    if improper_types:
        lines += ["", "Improper Coeffs", ""]
        for symbol, index in sorted(improper_types.items(), key=lambda kv: kv[1]):
            inversion = improper_coefficients(symbol)
            assert inversion is not None
            force, c0, c1, c2 = inversion
            lines.append(f"{index} {force:.6f} {c0:.6f} {c1:.6f} {c2:.6f} # {symbol}")

    lines += ["", "Atoms # full", ""]
    for row, _node in enumerate(nodes):
        charge = float(charges[row]) if charges is not None else 0.0
        x, y, z = coords[row]
        lines.append(
            f"{row + 1} 1 {atom_type_id[types[row]]} {charge:.6f} "
            f"{x:.8f} {y:.8f} {z:.8f} # {types[row]}"
        )

    for label, rows, width in (
        ("Bonds", bond_rows, 2),
        ("Angles", angle_rows, 3),
        ("Dihedrals", dihedral_rows, 4),
        ("Impropers", improper_rows, 4),
    ):
        if not rows:
            continue
        lines += ["", label, ""]
        for index, entry in enumerate(rows, 1):
            members = " ".join(str(a + 1) for a in entry[1 : 1 + width])
            lines.append(f"{index} {entry[0]} {members}")

    text = "\n".join(lines) + "\n"
    if path is not None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    # LAMMPS needs ghost atoms beyond the cutoff; keeping it under half
    # the shortest box length avoids the replication that corrupts the
    # lammps-interface path, and a truncated tail is the smaller error
    cutoff = min(12.5, 0.49 * float(lengths.min()))
    # LAMMPS drops a bond whose partner is outside the ghost shell
    # ("Bond atoms N M missing on proc 0"); the shell must reach the
    # longest bond, which in a small cell exceeds the pair cutoff
    # sized from the minimum-image bond, which is what LAMMPS actually
    # measures; the raw wrapped separation of a boundary-crossing bond
    # can exceed the box, and asking for a ghost shell wider than the
    # box is meaningless
    comm_cutoff = min(
        max(2.0 * longest_bond + 2.0, cutoff + 2.0), 0.99 * float(lengths.min())
    )
    styles = {
        "pair_style": f"lj/cut {cutoff:.3f}",
        "pair_modify": "mix geometric tail yes",
        "bond_style": "harmonic",
        "angle_style": _angle_style(angle_styles),
        "dihedral_style": "harmonic",
        "improper_style": "fourier",
        "special_bonds": "lj 0.0 0.0 1.0",
        "comm_modify": f"cutoff {comm_cutoff:.3f}",
    }
    pair_commands = tuple(
        f"pair_coeff {atom_type_id[t]} {atom_type_id[t]} "
        f"{pair_coefficients(table[t])[0]:.6f} {pair_coefficients(table[t])[1]:.6f}"
        for t in atom_types
    )
    closest = float("inf")
    if len(coords) > 1:
        separation = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        np.fill_diagonal(separation, np.inf)
        closest = float(separation.min())
    return LammpsData(
        text=text,
        n_atoms=len(nodes),
        n_bonds=len(bond_rows),
        n_angles=len(angle_rows),
        n_dihedrals=len(dihedral_rows),
        n_impropers=len(improper_rows),
        styles=styles,
        pair_coeffs=pair_commands,
        closest_contact=closest,
    )
