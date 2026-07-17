"""
Elastic constants of built frameworks at the force-field level.

The 6x6 elastic stiffness tensor is computed by finite differences of
the stress tensor under small strains, in-process through the same
LAMMPS/UFF4MOF backend as ``Framework.relax`` (the protocol of the
LAMMPS ``ELASTIC`` example): the lammps-interface minimization input
first relaxes cell and geometry, then each of the six Voigt strains is
applied in both directions with ``change_box``, the internal
coordinates re-minimized at fixed cell, and the stress read back.

The result is an :class:`ElasticProperties` — the symmetrized tensor
plus the standard derived quantities: Voigt/Reuss/Hill bulk and shear
moduli, Hill Young's modulus and Poisson ratio, and the directional
Young's modulus extrema (MOFs are often strongly anisotropic).

This is the screening level of the property funnel, not the truth
level: UFF4MOF stiffnesses carry force-field error, but rank
correlations against higher levels are the quantity of interest.

Notes
-----
Layer (2D) frameworks have a padded, non-physical c axis; their
out-of-plane components are meaningless at this level. The computation
does not forbid them, but only in-plane components should be read.
"""

from __future__ import annotations

import contextlib
import io
import logging
import tempfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from autografs.relax import _launch_lammps, _write_lammps_inputs

if TYPE_CHECKING:
    from pymatgen.core import Structure

    from autografs.framework import Framework

logger = logging.getLogger(__name__)

# LAMMPS real units report pressure in atmospheres
ATM_TO_GPA = 1.01325e-4

# Voigt order: xx, yy, zz, yz, xz, xy
_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
_VOIGT_INDEX = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])


def voigt_to_full(voigt: np.ndarray) -> np.ndarray:
    """Expand a 6x6 Voigt stiffness matrix to the full 3x3x3x3 tensor.

    Stiffness carries no Voigt weight factors; this is index expansion
    only (do not use it on a compliance matrix).
    """
    return np.asarray(np.asarray(voigt)[_VOIGT_INDEX[:, :, None, None], _VOIGT_INDEX])


def full_to_voigt(full: np.ndarray) -> np.ndarray:
    """Contract a full 3x3x3x3 stiffness tensor to its 6x6 Voigt form."""
    out = np.empty((6, 6))
    for a, (i, j) in enumerate(_VOIGT_PAIRS):
        for b, (k, l) in enumerate(_VOIGT_PAIRS):  # noqa: E741
            out[a, b] = full[i, j, k, l]
    return out


def symmetrized_stiffness(
    voigt: np.ndarray, structure: Structure, symprec: float = 0.01
) -> np.ndarray:
    """Project a stiffness matrix onto the structure's point group.

    Averages the full tensor over the cartesian rotations of the
    structure's space group — the crystal-system-aware symmetrization:
    a cubic framework comes out with the 3-constant cubic pattern, a
    triclinic one is only made major-symmetric. Falls back to plain
    ``(C + C.T) / 2`` with a warning when symmetry detection fails.
    """
    sym = np.asarray(0.5 * (np.asarray(voigt) + np.asarray(voigt).T))
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        operations = SpacegroupAnalyzer(
            structure, symprec=symprec
        ).get_symmetry_operations(cartesian=True)
    except Exception:
        logger.warning(
            "Symmetry detection failed; the elastic tensor is only "
            "made major-symmetric, not projected onto a crystal system."
        )
        return sym
    # space-group operations repeat each point-group rotation once per
    # inequivalent translation; average over the distinct rotations
    rotations = {
        tuple(np.round(op.rotation_matrix, 8).ravel()): op.rotation_matrix
        for op in operations
    }
    full = voigt_to_full(sym)
    averaged = np.zeros_like(full)
    for rot in rotations.values():
        averaged += np.einsum("ai,bj,ck,dl,ijkl->abcd", rot, rot, rot, rot, full)
    return full_to_voigt(averaged / len(rotations))


def _fibonacci_sphere(n: int) -> np.ndarray:
    """(n, 3) roughly uniform unit vectors (deterministic)."""
    k = np.arange(n, dtype=float)
    phi = k * np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (k + 0.5) / n
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


# high-symmetry directions where extrema usually sit exactly
_SPECIAL_DIRECTIONS = np.array(
    [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    + [(1, s, 0) for s in (1, -1)]
    + [(1, 0, s) for s in (1, -1)]
    + [(0, 1, s) for s in (1, -1)]
    + [(1, s, t) for s in (1, -1) for t in (1, -1)],
    dtype=float,
)


@dataclass(frozen=True, eq=False)
class ElasticProperties:
    """A symmetrized 6x6 stiffness tensor (GPa) and its derived moduli.

    Voigt component order is (xx, yy, zz, yz, xz, xy); shear entries
    are with respect to engineering strains.
    """

    stiffness: np.ndarray

    @cached_property
    def compliance(self) -> np.ndarray:
        """The 6x6 compliance matrix (1/GPa), inverse of the stiffness."""
        return np.linalg.inv(self.stiffness)

    @property
    def is_stable(self) -> bool:
        """Born stability: the stiffness matrix is positive definite."""
        return bool(np.all(np.linalg.eigvalsh(self.stiffness) > 0.0))

    @property
    def bulk_voigt(self) -> float:
        c = self.stiffness
        return float(
            (c[0, 0] + c[1, 1] + c[2, 2] + 2.0 * (c[0, 1] + c[0, 2] + c[1, 2])) / 9.0
        )

    @property
    def bulk_reuss(self) -> float:
        s = self.compliance
        return float(
            1.0 / (s[0, 0] + s[1, 1] + s[2, 2] + 2.0 * (s[0, 1] + s[0, 2] + s[1, 2]))
        )

    @property
    def bulk_hill(self) -> float:
        """Hill bulk modulus (GPa): Voigt/Reuss average."""
        return 0.5 * (self.bulk_voigt + self.bulk_reuss)

    @property
    def shear_voigt(self) -> float:
        c = self.stiffness
        return float(
            (
                c[0, 0]
                + c[1, 1]
                + c[2, 2]
                - (c[0, 1] + c[0, 2] + c[1, 2])
                + 3.0 * (c[3, 3] + c[4, 4] + c[5, 5])
            )
            / 15.0
        )

    @property
    def shear_reuss(self) -> float:
        s = self.compliance
        return float(
            15.0
            / (
                4.0 * (s[0, 0] + s[1, 1] + s[2, 2])
                - 4.0 * (s[0, 1] + s[0, 2] + s[1, 2])
                + 3.0 * (s[3, 3] + s[4, 4] + s[5, 5])
            )
        )

    @property
    def shear_hill(self) -> float:
        """Hill shear modulus (GPa): Voigt/Reuss average."""
        return 0.5 * (self.shear_voigt + self.shear_reuss)

    @property
    def young_hill(self) -> float:
        """Isotropic-average Young's modulus (GPa), from Hill B and G."""
        bulk, shear = self.bulk_hill, self.shear_hill
        return 9.0 * bulk * shear / (3.0 * bulk + shear)

    @property
    def poisson_hill(self) -> float:
        """Isotropic-average Poisson ratio, from Hill B and G."""
        bulk, shear = self.bulk_hill, self.shear_hill
        return (3.0 * bulk - 2.0 * shear) / (2.0 * (3.0 * bulk + shear))

    def young_modulus(self, direction: np.ndarray) -> float:
        """Young's modulus (GPa) for uniaxial stress along ``direction``."""
        return float(self._young_along(np.atleast_2d(direction))[0])

    def _young_along(self, directions: np.ndarray) -> np.ndarray:
        """Vectorized directional Young's modulus over (n, 3) directions."""
        n = np.asarray(directions, dtype=float)
        n = n / np.linalg.norm(n, axis=1, keepdims=True)
        # compliance in full-tensor form carries the Voigt weights:
        # one factor 2 per shear index pair
        weights = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        s_full = voigt_to_full(self.compliance / np.outer(weights, weights))
        inverse = np.einsum("ni,nj,nk,nl,ijkl->n", n, n, n, n, s_full)
        return np.asarray(1.0 / inverse)

    @cached_property
    def _young_extrema(self) -> tuple[float, float]:
        directions = np.vstack([_SPECIAL_DIRECTIONS, _fibonacci_sphere(4096)])
        values = self._young_along(directions)
        return float(values.min()), float(values.max())

    @property
    def young_min(self) -> float:
        """Smallest directional Young's modulus (GPa)."""
        return self._young_extrema[0]

    @property
    def young_max(self) -> float:
        """Largest directional Young's modulus (GPa)."""
        return self._young_extrema[1]

    def __repr__(self) -> str:
        return (
            f"ElasticProperties(bulk_hill={self.bulk_hill:.2f} GPa, "
            f"shear_hill={self.shear_hill:.2f} GPa, "
            f"young_hill={self.young_hill:.2f} GPa, "
            f"stable={self.is_stable})"
        )


def _strain_command(direction: int, box: dict[str, float], strain: float) -> str:
    """The ``change_box`` deltas realizing one signed Voigt strain.

    Deformations follow the LAMMPS ``ELASTIC`` example: normal strains
    scale one box length (and the tilt factors sharing its cartesian
    column); shear strains are applied as a single tilt displacement,
    i.e. engineering shear. All commands remap atoms affinely.
    """
    deltas = {
        0: f"x delta 0 {strain * box['lx']} "
        f"xy delta {strain * box['xy']} xz delta {strain * box['xz']}",
        1: f"y delta 0 {strain * box['ly']} yz delta {strain * box['yz']}",
        2: f"z delta 0 {strain * box['lz']}",
        3: f"yz delta {strain * box['lz']}",
        4: f"xz delta {strain * box['lz']}",
        5: f"xy delta {strain * box['ly']}",
    }
    return f"change_box all {deltas[direction]} remap units box"


def _finite_difference_stiffness(lmp, strain: float) -> np.ndarray:
    """Stress-strain finite differences on a relaxed LAMMPS session.

    Expects a session whose input has already been run to a relaxed,
    fix-free state. Returns the raw (unsymmetrized) 6x6 stiffness in
    GPa. The session is left in a strained state afterwards.
    """
    (xlo, ylo, zlo), (xhi, yhi, zhi), xy, yz, xz, *_ = lmp.extract_box()
    box = {
        "lx": xhi - xlo,
        "ly": yhi - ylo,
        "lz": zhi - zlo,
        "xy": xy,
        "yz": yz,
        "xz": xz,
    }
    restore_box = (
        f"change_box all x final {xlo} {xhi} y final {ylo} {yhi} "
        f"z final {zlo} {zhi} xy final {xy} xz final {xz} yz final {yz} "
        "units box"
    )
    # reference coordinates, in the ctypes layout scatter_atoms expects
    reference = lmp.gather_atoms("x", 1, 3)
    lmp.command("min_style fire")

    stiffness = np.zeros((6, 6))
    for direction in range(6):
        stresses = {}
        for sign in (1.0, -1.0):
            lmp.command(restore_box)
            lmp.scatter_atoms("x", 1, 3, reference)
            lmp.command(_strain_command(direction, box, sign * strain))
            lmp.command("minimize 1.0e-15 1.0e-15 10000 100000")
            # LAMMPS pressure is minus the stress
            stresses[sign] = -np.array(
                [
                    lmp.get_thermo(key)
                    for key in ("pxx", "pyy", "pzz", "pyz", "pxz", "pxy")
                ]
            )
        stiffness[:, direction] = (stresses[1.0] - stresses[-1.0]) / (2.0 * strain)
    return stiffness * ATM_TO_GPA


def elastic_properties(
    framework: Framework,
    force_field: str = "UFF4MOF",
    cutoff: float = 12.5,
    strain: float = 2e-3,
    symprec: float = 0.01,
    verbose: bool = False,
) -> ElasticProperties:
    """Compute the elastic stiffness tensor of a framework with LAMMPS.

    The framework is first fully relaxed (cell and geometry, same
    protocol as ``Framework.relax``), then strained; passing an
    already-relaxed framework just makes the first stage converge
    immediately.

    Parameters
    ----------
    framework : Framework
        The framework; not modified.
    force_field : str, optional
        Force field known to lammps-interface, by default "UFF4MOF".
    cutoff : float, optional
        Non-bonded cutoff in Angstrom, by default 12.5.
    strain : float, optional
        Finite-difference strain amplitude, by default 2e-3. Applied
        symmetrically (central differences).
    symprec : float, optional
        Symmetry tolerance for the crystal-system projection of the
        tensor, by default 0.01.
    verbose : bool, optional
        Pass the backend output through instead of suppressing it.

    Returns
    -------
    ElasticProperties
        The symmetrized stiffness tensor (GPa) and derived moduli.

    Raises
    ------
    RelaxationError
        If the optional LAMMPS backends are missing or fail.
    ValueError
        If ``strain`` is not positive.
    """
    if strain <= 0.0:
        raise ValueError(f"strain must be positive, got {strain}.")
    sink = io.StringIO()
    quiet: contextlib.AbstractContextManager = (
        contextlib.nullcontext() if verbose else contextlib.redirect_stdout(sink)
    )
    with tempfile.TemporaryDirectory(prefix="autografs_elastic_") as tmp:
        workdir = Path(tmp)
        safe_name, _ = _write_lammps_inputs(
            framework, force_field, cutoff, workdir, quiet
        )
        lmp = _launch_lammps()
        try:
            with quiet, contextlib.chdir(workdir):
                lmp.file(f"in.{safe_name}")
                raw = _finite_difference_stiffness(lmp, strain)
        finally:
            lmp.close()
    result = ElasticProperties(
        stiffness=symmetrized_stiffness(raw, framework.structure, symprec=symprec)
    )
    logger.info(
        f"Elastic tensor of {framework.name!r} with {force_field}: "
        f"B={result.bulk_hill:.2f} GPa, G={result.shear_hill:.2f} GPa, "
        f"E={result.young_hill:.2f} GPa (Hill), stable={result.is_stable}."
    )
    return result
