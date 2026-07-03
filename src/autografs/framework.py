"""
Framework: the result object of a build.

Wraps the bond graph produced by the builder together with convenient
crystallographic views of it: a pymatgen Structure (wrapped into the
cell), an ASE Atoms object, CIF export, and the GULP input generator
for UFF4MOF optimization. The graph remains the source of truth for
bonds, tags, and force-field types.

Examples
--------
>>> mof = mofgen.build(topology, mappings)
>>> mof.formula
'Zn8 O26 C48 H12'
>>> mof.write_cif("mof5.cif")
>>> atoms = mof.to_ase()
>>> gulp_input = mof.to_gulp(write_to_file=False)
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import networkx
import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

import autografs.utils

if TYPE_CHECKING:
    import ase

logger = logging.getLogger(__name__)


class Framework:
    """A built periodic framework.

    Attributes
    ----------
    graph : networkx.Graph
        Bond graph with node attributes ``symbol``, ``coord``
        (unwrapped cartesian), ``tag``, ``ufftype`` and edge attribute
        ``bond_order``. Source of truth for connectivity.
    name : str
        Human-readable identifier, usually the topology name.
    """

    def __init__(self, graph: networkx.Graph, name: str = "framework") -> None:
        """
        Parameters
        ----------
        graph : networkx.Graph
            The bond graph produced by the builder; must carry a
            ``cell`` graph attribute (3x3 matrix).
        name : str, optional
            Identifier used in exports and repr.
        """
        self.graph = graph
        self.name = name

    # ------------------------------------------------------------------
    # basic views
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.graph.number_of_nodes()

    def __repr__(self) -> str:
        abc = tuple(round(float(x), 2) for x in self.lattice.abc)
        return (
            f"Framework({self.name!r}, {self.structure.composition.formula!r}, "
            f"abc={abc})"
        )

    @property
    def cell(self) -> np.ndarray:
        """The 3x3 cell matrix."""
        return np.asarray(self.graph.graph["cell"], dtype=float)

    @property
    def lattice(self) -> Lattice:
        """The cell as a pymatgen Lattice."""
        return Lattice(self.cell)

    @property
    def symbols(self) -> list[str]:
        """Element symbols in node order."""
        return [self.graph.nodes[n]["symbol"] for n in sorted(self.graph)]

    @property
    def cart_coords(self) -> np.ndarray:
        """Unwrapped cartesian coordinates in node order."""
        return np.array([self.graph.nodes[n]["coord"] for n in sorted(self.graph)])

    @property
    def mmtypes(self) -> list[str]:
        """UFF4MOF atom types in node order."""
        return [self.graph.nodes[n]["ufftype"] for n in sorted(self.graph)]

    @property
    def bonds(self) -> list[tuple[int, int, float]]:
        """Bonds as (i, j, bond_order) triples in node order."""
        return [
            (min(i, j), max(i, j), data.get("bond_order", 1.0))
            for i, j, data in self.graph.edges(data=True)
        ]

    @property
    def formula(self) -> str:
        """Chemical formula of the unit cell."""
        return self.structure.composition.formula

    # ------------------------------------------------------------------
    # crystallographic exports
    # ------------------------------------------------------------------

    @functools.cached_property
    def structure(self) -> Structure:
        """The framework as a pymatgen Structure, wrapped into the cell.

        Site order matches graph node order; ``tags`` and ``ufftype``
        are carried as site properties. Bonds are not part of a
        Structure - use .graph for connectivity.
        """
        lattice = self.lattice
        frac = self.cart_coords @ np.linalg.inv(self.cell) % 1.0
        # x % 1.0 returns exactly 1.0 for tiny negative x
        frac[frac >= 1.0] -= 1.0
        tags = [self.graph.nodes[n]["tag"] for n in sorted(self.graph)]
        return Structure(
            lattice,
            self.symbols,
            frac,
            coords_are_cartesian=False,
            site_properties={"tags": tags, "ufftype": self.mmtypes},
        )

    def write_cif(self, path: str | Path, symprec: float | None = None) -> Path:
        """Write the framework to a CIF file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        symprec : float or None, optional
            If given, pymatgen attempts to detect symmetry with this
            tolerance and writes a symmetrized CIF; None (default)
            writes P1.

        Returns
        -------
        Path
            The written path.
        """
        from pymatgen.io.cif import CifWriter

        path = Path(path)
        CifWriter(self.structure, symprec=symprec).write_file(path)
        logger.info(f"Wrote {self!r} to {path}")
        return path

    def to_ase(self) -> ase.Atoms:
        """The framework as an ASE Atoms object (wrapped, periodic)."""
        from ase import Atoms

        return Atoms(
            symbols=self.symbols,
            scaled_positions=self.structure.frac_coords,
            cell=self.cell,
            pbc=True,
        )

    def to_gulp(self, write_to_file: bool = False) -> str:
        """GULP input for UFF4MOF optimization of this framework.

        Parameters
        ----------
        write_to_file : bool, optional
            If True, writes ``{name}.gin`` in the current directory.

        Returns
        -------
        str
            The GULP input file content.
        """
        return autografs.utils.networkx_to_gulp(
            self.graph, name=self.name, write_to_file=write_to_file
        )

    def view(self) -> None:
        """Open the framework in ASE's interactive viewer."""
        from ase.visualize import view

        view(self.to_ase())
