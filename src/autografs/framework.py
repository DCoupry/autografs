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
from autografs.exceptions import StackingError

if TYPE_CHECKING:
    import ase

logger = logging.getLogger(__name__)

# graphite-like default; typical COF interlayer range is 3.3-3.6 A
DEFAULT_INTERLAYER = 3.35

# in-plane fractional offset of the second layer per stacking mode
STACKING_OFFSETS: dict[str, tuple[float, float] | None] = {
    "AA": None,
    "AB": (1.0 / 3.0, 2.0 / 3.0),
    "serrated": (0.5, 0.0),
    "staggered": (0.5, 0.5),
}


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
        return int(self.graph.number_of_nodes())

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

    # ------------------------------------------------------------------
    # layer stacking
    # ------------------------------------------------------------------

    def stack(
        self,
        mode: str = "AA",
        interlayer: float = DEFAULT_INTERLAYER,
        offset: tuple[float, float] | None = None,
    ) -> Framework:
        """Stack this 2D layer into a COF crystal.

        A framework built on a layer net has c set to the slab padding,
        not a physical spacing: the interlayer geometry is dispersion-
        driven, a user variable, not a topology property. This replaces
        the padding with a real stacking:

        - ``AA``: eclipsed; the cell keeps one layer and c becomes
          ``interlayer``.
        - ``AB``, ``serrated``, ``staggered``: a two-layer cell with c
          = ``2 * interlayer`` and a second, in-plane-offset copy of
          the layer. Default fractional offsets: AB (1/3, 2/3),
          serrated (1/2, 0), staggered (1/2, 1/2).

        Layers are van-der-Waals stacked: intra-layer bonds are
        duplicated per layer and no inter-layer bonds are created.

        Parameters
        ----------
        mode : str, optional
            One of ``AA``, ``AB``, ``serrated``, ``staggered``.
        interlayer : float, optional
            Spacing between successive layer planes in Angstrom, by
            default 3.35 (graphite-like; typical COFs are 3.3-3.6).
        offset : tuple[float, float] or None, optional
            Fractional in-plane offset of the second layer, overriding
            the mode default. Not applicable to AA.

        Returns
        -------
        Framework
            A new Framework; this one is unchanged.

        Raises
        ------
        StackingError
            If the framework is not a flat layer with c perpendicular
            to it (e.g. a 3D net, or bonds crossing the slab).

        Examples
        --------
        >>> layer = mofgen.build(hcb, mappings)     # c = pad value
        >>> cof = layer.stack(mode="AA", interlayer=3.35)
        >>> cof = layer.stack(mode="serrated", offset=(0.5, 0))
        """
        if mode not in STACKING_OFFSETS:
            raise ValueError(
                f"Unknown stacking mode {mode!r}; "
                f"expected one of {sorted(STACKING_OFFSETS)}."
            )
        if interlayer <= 0:
            raise ValueError(f"Interlayer spacing must be positive, got {interlayer}.")
        cell = self.cell
        pad_c = float(cell[2, 2])
        # the padded-slab convention puts c along z, perpendicular to
        # the layer plane spanned by a and b
        if np.abs(cell[:2, 2]).max() > 1e-6 * self.lattice.a or np.abs(
            cell[2, :2]
        ).max() > 1e-6 * abs(pad_c):
            raise StackingError(
                f"Framework {self.name!r} has no c-axis perpendicular to "
                "the a-b plane; not a layer slab."
            )
        # a layer keeps all atoms in a thin z-window; coords are
        # unwrapped, so this also bounds every bond's z-span - a tag
        # pair bonded through a c-image cannot hide below it
        z = self.cart_coords[:, 2]
        thickness = float(z.max() - z.min())
        if thickness > 0.5 * pad_c:
            raise StackingError(
                f"Framework {self.name!r} spans {thickness:.2f} A along c "
                f"(cell {pad_c:.2f} A); not a 2D layer."
            )
        if mode == "AA":
            if offset is not None:
                raise ValueError("AA stacking takes no in-plane offset.")
            n_layers = 1
        else:
            if offset is None:
                offset = STACKING_OFFSETS[mode]
            n_layers = 2
        new_cell = cell.copy()
        new_cell[2] = [0.0, 0.0, n_layers * interlayer]
        stacked = networkx.Graph(cell=new_cell)
        stacked.add_nodes_from(self.graph.nodes(data=True))
        stacked.add_edges_from(self.graph.edges(data=True))
        if offset is not None:
            shift = offset[0] * cell[0] + offset[1] * cell[1]
            shift[2] += interlayer
            relabel = len(self.graph)
            # duplicated tags stay unique so tag-pair semantics survive
            tag_base = max(
                (data["tag"] for _, data in self.graph.nodes(data=True)), default=0
            )
            for node, data in self.graph.nodes(data=True):
                copied = dict(data)
                copied["coord"] = np.asarray(data["coord"], dtype=float) + shift
                if copied["tag"] > 0:
                    copied["tag"] += tag_base
                stacked.add_node(node + relabel, **copied)
            for i, j, data in self.graph.edges(data=True):
                stacked.add_edge(i + relabel, j + relabel, **data)
        return Framework(stacked, name=f"{self.name}_{mode}")
