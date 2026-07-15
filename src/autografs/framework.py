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
import math
from collections.abc import Iterable
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
    from pymatgen.core.structure import Molecule

    from autografs.topology import Topology

__all__ = [
    "Framework",
]

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
    energy : float or None
        Force-field energy per unit cell in kcal/mol; set by
        ``relax()``, None for as-built frameworks.

    Notes
    -----
    The graph is the source of truth, but the crystallographic views
    derived from it cache: ``structure`` (and everything built on it -
    ``formula``, ``min_contact``, exports, porosity descriptors) is
    computed once on first access. Mutating the graph of a Framework
    whose views have already been read leaves those views stale. The
    editing methods (``supercell``, ``defects``, ``functionalize``,
    ...) sidestep this by always returning a new Framework; direct
    graph surgery should construct a new ``Framework(graph)`` too.
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
        self.energy: float | None = None

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

    @property
    def slots(self) -> dict[int, str]:
        """SBU name per placed slot, keyed by slot index.

        The slot index identifies one placed SBU: pass it to
        ``rotate``, ``flip`` or ``defects`` to edit that unit.
        Slot ids of a supercell or stacked framework are offset
        per copy, so every placed unit stays individually
        addressable.
        """
        found: dict[int, str] = {}
        for _, data in self.graph.nodes(data=True):
            if "slot" in data:
                found[data["slot"]] = data["sbu"]
        return dict(sorted(found.items()))

    def min_contact(self, cutoff: float = 3.0) -> float:
        """Smallest periodic distance between non-bonded atoms.

        Screens for overlapping or interpenetrating output: every
        periodic image is considered, so an atom sitting too close to
        its own image in a collapsed cell is caught too. Pairs bonded
        in the graph are exempt (at every image - a bonded pair is
        never reported, even through a boundary).

        Blind spot: because the exemption ignores the image, a bonded
        pair whose atoms *also* approach each other through a different
        periodic image (possible in a strongly collapsed cell) is not
        reported either. Contacts between non-bonded pairs, including
        self-image contacts, are always seen.

        Parameters
        ----------
        cutoff : float, optional
            Search radius in Angstrom, by default 3.0. Contacts beyond
            it are not examined.

        Returns
        -------
        float
            The smallest non-bonded contact distance found, or
            ``math.inf`` when none is within ``cutoff``.
        """
        centers, points, _, distances = self.structure.get_neighbor_list(r=cutoff)
        if len(distances) == 0:
            return math.inf
        # pair key = lo * n + hi; self-image contacts (i == i) can
        # never collide with a bond key since the graph has no loops
        n = len(self)
        lo = np.minimum(centers, points).astype(np.int64)
        hi = np.maximum(centers, points).astype(np.int64)
        bonded = np.array(
            sorted({min(i, j) * n + max(i, j) for i, j in self.graph.edges()}),
            dtype=np.int64,
        )
        unbonded = ~np.isin(lo * n + hi, bonded)
        if not unbonded.any():
            return math.inf
        return float(distances[unbonded].min())

    def verify_net(self, topology: Topology) -> None:
        """Check that this as-built framework realizes its blueprint.

        Compares the labeled quotient graphs (one node per slot, one
        edge per inter-SBU bond with its periodic image voltage) as
        exact multisets - the check that catches mis-paired anchors or
        a bond through the wrong image, which the geometric gates
        cannot see. Also available at build time via
        ``build(..., verify_net=True)``.

        Only meaningful for as-built frameworks: supercells, stacks
        and defective frameworks intentionally change the quotient
        graph.

        Parameters
        ----------
        topology : Topology
            The blueprint this framework was built on.

        Raises
        ------
        NetMismatchError
            If the framework does not realize the blueprint's net.
        """
        from autografs.net import verify_net

        verify_net(self, topology)

    # ------------------------------------------------------------------
    # porosity descriptors (implementations in autografs.porosity)
    # ------------------------------------------------------------------

    @property
    def density(self) -> float:
        """Crystal density in g/cm3."""
        return float(self.structure.density)

    def void_fraction(self, probe_radius: float = 0.0, spacing: float = 0.4) -> float:
        """Fraction of the cell volume where a probe sphere fits.

        Parameters
        ----------
        probe_radius : float, optional
            Probe radius in Angstrom; 0.0 (default) gives the
            geometric void fraction outside the vdW surfaces. 1.2
            approximates a helium probe, 1.86 nitrogen.
        spacing : float, optional
            Grid resolution in Angstrom; smaller is more accurate and
            slower.

        Returns
        -------
        float
            Accessible-volume fraction, in [0, 1].
        """
        from autografs.porosity import void_fraction

        return void_fraction(self, probe_radius=probe_radius, spacing=spacing)

    def largest_cavity_diameter(self, spacing: float = 0.4) -> float:
        """Diameter of the largest sphere fitting in the structure (LCD).

        Parameters
        ----------
        spacing : float, optional
            Grid resolution in Angstrom.

        Returns
        -------
        float
            The LCD in Angstrom; 0.0 for a dense structure.
        """
        from autografs.porosity import largest_cavity_diameter

        return largest_cavity_diameter(self, spacing=spacing)

    def pore_limiting_diameter(self, spacing: float = 0.4) -> float:
        """Diameter of the largest sphere that can percolate through (PLD).

        The limiting bottleneck of the pore network: the widest probe
        that can travel through the periodic structure in any
        direction. Always <= the largest cavity diameter.

        Parameters
        ----------
        spacing : float, optional
            Grid resolution in Angstrom.

        Returns
        -------
        float
            The PLD in Angstrom; 0.0 when no channel percolates.
        """
        from autografs.porosity import pore_limiting_diameter

        return pore_limiting_diameter(self, spacing=spacing)

    # ------------------------------------------------------------------
    # crystallographic exports
    # ------------------------------------------------------------------

    @functools.cached_property
    def structure(self) -> Structure:
        """The framework as a pymatgen Structure, wrapped into the cell.

        Site order matches graph node order; ``tags`` and ``ufftype``
        are carried as site properties. Bonds are not part of a
        Structure - use .graph for connectivity.

        Cached on first access: mutating ``graph`` afterwards leaves
        this view (and everything derived from it) stale - build a new
        Framework instead (see the class Notes).
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

    def save(self, path: str | Path) -> Path:
        """Save the framework to a versioned JSON file.

        Unlike CIF export, this persists everything a Framework is:
        the bond graph with bond orders, UFF4MOF types, anchor tags,
        and the per-atom slot/SBU provenance - so a framework saved in
        one session can be reloaded with ``Framework.load`` and edited
        (supercell, defects, functionalize, ...) in another.

        Parameters
        ----------
        path : str or Path
            Output path; gzip-compressed and compact when it ends in
            ``.gz``, pretty-printed JSON otherwise.

        Returns
        -------
        Path
            The written path.
        """
        from autografs.framework_io import save_framework

        return save_framework(self, path)

    @classmethod
    def load(cls, path: str | Path) -> Framework:
        """Load a framework saved with ``save``.

        Parameters
        ----------
        path : str or Path
            Input path (.json or .json.gz).

        Returns
        -------
        Framework
            The loaded framework, editable exactly like the saved one.
        """
        from autografs.framework_io import load_framework

        return load_framework(path)

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

    def relax(
        self,
        force_field: str = "UFF4MOF",
        cutoff: float = 12.5,
        verbose: bool = False,
    ) -> Framework:
        """Relax geometry and cell with LAMMPS (UFF4MOF by default).

        Requires the optional backends: ``pip install "autografs[relax]"``
        (on Windows the LAMMPS wheel also needs the Microsoft MPI
        runtime). Runs the lammps-interface minimization protocol
        (alternating cell box-relax and FIRE) in-process and maps the
        relaxed geometry back onto this framework's bond graph.

        Parameters
        ----------
        force_field : str, optional
            Force field known to lammps-interface, by default
            "UFF4MOF"; "UFF" and "Dreiding" also work.
        cutoff : float, optional
            Non-bonded cutoff in Angstrom, by default 12.5. Cells too
            small for it are relaxed as an internal supercell and
            folded back.
        verbose : bool, optional
            Pass the backend output through instead of suppressing it.

        Returns
        -------
        Framework
            A new Framework with identical connectivity, relaxed
            coordinates and cell, and the force-field energy per unit
            cell (kcal/mol) in ``.energy``. This framework is
            unchanged.

        Raises
        ------
        RelaxationError
            If the backends are missing or the structure cannot be
            relaxed and mapped back.
        """
        from autografs.relax import relax_framework

        return relax_framework(
            self, force_field=force_field, cutoff=cutoff, verbose=verbose
        )

    # ------------------------------------------------------------------
    # post-build editing (implementations in autografs.editing)
    # ------------------------------------------------------------------

    def rotate(self, slot: int, theta: float) -> Framework:
        """Rotate one placed 2-connected SBU around its bond axis.

        The alignment of a 2-connected SBU (a linker) is
        underdetermined: any rotation around the axis through its two
        connection points fits the slot equally well. This picks a
        different one after the fact, e.g. to relieve a steric clash
        or break an artificial symmetry.

        Parameters
        ----------
        slot : int
            The placed SBU to rotate; see ``slots`` for the available
            indices and their SBU names.
        theta : float
            Rotation angle in radians, around the axis through the
            SBU's two connection (anchor) atoms. The anchors do not
            move, so all framework bonds are preserved exactly.

        Returns
        -------
        Framework
            A new Framework; this one is unchanged.

        Raises
        ------
        ValueError
            If the slot is unknown or the SBU is not 2-connected.
        """
        from autografs.editing import rotate_sbu

        return rotate_sbu(self, slot=slot, theta=theta)

    def flip(self, slot: int) -> Framework:
        """Mirror one placed SBU while keeping its connection points.

        Applies a reflection that fixes the SBU's anchor atoms: for a
        2-connected SBU, through a plane containing the anchor axis;
        for a higher-connected SBU whose anchors are coplanar, through
        the anchor plane. The result is the mirror image of the placed
        unit (chirality inverted), still bonded exactly as before.

        Parameters
        ----------
        slot : int
            The placed SBU to flip; see ``slots``.

        Returns
        -------
        Framework
            A new Framework; this one is unchanged.

        Raises
        ------
        ValueError
            If the slot is unknown, or no anchor-preserving mirror
            exists (non-coplanar anchors, or a strictly linear unit).
        """
        from autografs.editing import flip_sbu

        return flip_sbu(self, slot=slot)

    def interpenetrate(
        self,
        n: int = 2,
        offset: tuple[float, float, float] | str = "auto",
    ) -> Framework:
        """Generate an n-fold interpenetrated (catenated) framework.

        Places n displaced copies of this framework in the same cell -
        the classic catenation of open nets (IRMOF-9/-11, dia-c,
        pcu-c). Copy k is displaced by ``k * offset`` (fractional);
        copies are van-der-Waals interlocked, never bonded, and their
        tags and slot ids stay unique so every placed SBU remains
        addressable.

        Parameters
        ----------
        n : int, optional
            Number of interpenetrated nets, by default 2.
        offset : tuple[float, float, float] or "auto", optional
            Fractional displacement between successive nets. "auto"
            (default) tries the high-symmetry candidates (body center
            first, then face and edge centers) and keeps the one whose
            closest inter-net contact is largest.

        Returns
        -------
        Framework
            A new Framework; this one is unchanged. Check the result
            with ``min_contact()`` - a dense parent framework cannot
            host another net without clashes.

        Examples
        --------
        >>> catenated = mof.interpenetrate()                  # auto offset
        >>> catenated = mof.interpenetrate(2, (0.5, 0.5, 0.5))
        >>> catenated.min_contact()
        """
        from autografs.editing import interpenetrate

        return interpenetrate(self, n=n, offset=offset)

    def supercell(self, scale: int | tuple[int, int, int]) -> Framework:
        """Replicate the framework into a supercell.

        Bonds crossing the original cell boundary are remapped onto
        the correct periodic image, so the supercell's bond graph is
        exact - a prerequisite for statistical defects, which need
        whole SBUs removable without breaking the network bookkeeping.

        Parameters
        ----------
        scale : int or tuple[int, int, int]
            Multiplier along each cell vector; an int applies to all
            three.

        Returns
        -------
        Framework
            A new Framework; this one is unchanged. Slot indices are
            offset per copy (see ``slots``), tags stay pair-unique.
        """
        from autografs.editing import make_supercell

        return make_supercell(self, scale=scale)

    def defects(
        self,
        fraction: float | None = None,
        slots: Iterable[int] | None = None,
        sbu: str | None = None,
        cap: str | None = "H",
        seed: int | None = None,
    ) -> Framework:
        """Remove whole placed SBUs (missing-linker/node defects).

        Removes either an explicit list of slots or a seeded random
        fraction of them, and caps the dangling connection points left
        on the surviving neighbors. Combine with ``supercell`` for
        statistically defective crystals::

            defective = mof.supercell(2).defects(
                fraction=0.25, sbu="Benzene_linear", seed=42
            )

        Parameters
        ----------
        fraction : float or None, optional
            Fraction of the candidate slots to remove; the count is
            rounded to the nearest integer and the slots are drawn
            without replacement from a seeded generator.
        slots : Iterable[int] or None, optional
            Explicit slot indices to remove instead of sampling.
            Exactly one of ``fraction`` and ``slots`` must be given.
        sbu : str or None, optional
            Restrict the candidates to placed SBUs with this name
            (e.g. remove only linkers, only nodes...).
        cap : str or None, optional
            Element used to cap each dangling anchor, placed along
            the removed bond direction at the Cordero covalent
            distance; by default "H". None leaves open coordination
            sites (no capping).
        seed : int or None, optional
            Seed for the sampling generator.

        Returns
        -------
        Framework
            A new Framework; this one is unchanged.

        Raises
        ------
        ValueError
            On unknown slots, an sbu filter matching nothing, or an
            invalid fraction.
        """
        from autografs.editing import make_defects

        return make_defects(
            self, fraction=fraction, slots=slots, sbu=sbu, cap=cap, seed=seed
        )

    def functionalizable_sites(
        self, symbol: str = "H", sbu: str | None = None
    ) -> list[int]:
        """Atom indices where ``functionalize`` can graft a group.

        A site is functionalizable when it is a terminal atom of the
        requested element: single-bonded to exactly one neighbor and
        not itself a framework connection point.

        Parameters
        ----------
        symbol : str, optional
            Element to look for, by default "H".
        sbu : str or None, optional
            Only list sites on placed SBUs with this name.

        Returns
        -------
        list[int]
            Sorted atom indices, usable directly as ``index`` in
            ``functionalize``.
        """
        from autografs.editing import functionalizable_sites

        return functionalizable_sites(self, symbol=symbol, sbu=sbu)

    def functionalize(
        self,
        index: int | Iterable[int],
        functional_group: str | Molecule,
    ) -> Framework:
        """Replace terminal atoms with a functional group.

        The post-build counterpart of ``Fragment.functionalize``:
        grafts the group onto the framework itself, so one built
        structure can be decorated site by site (including sites made
        inequivalent by a supercell). The group is aligned along the
        existing bond direction and placed at the tabulated
        parent-to-group bond length.

        Parameters
        ----------
        index : int or Iterable[int]
            Atom indices to replace, from ``functionalizable_sites``.
            Each must be a terminal atom (exactly one bond) that is
            not a framework connection point.
        functional_group : str or Molecule
            A key of ``pymatgen.core.structure.FunctionalGroups``
            (e.g. "amine", "methyl", "nitro"), or a custom pymatgen
            Molecule containing exactly one dummy atom ("X") marking
            the attachment point.

        Returns
        -------
        Framework
            A new Framework; this one is unchanged.

        Raises
        ------
        ValueError
            If a site is not terminal, is a connection point, or the
            functional group has no unique attachment dummy.
        """
        from autografs.editing import functionalize

        return functionalize(self, index=index, functional_group=functional_group)

    # ------------------------------------------------------------------
    # layer stacking
    # ------------------------------------------------------------------

    def stack(
        self,
        mode: str = "AA",
        interlayer: float = DEFAULT_INTERLAYER,
        offset: tuple[float, float] | None = None,
        sequence: Iterable[tuple[float, float]] | None = None,
        n_layers: int | None = None,
        seed: int | None = None,
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
        - ``random``: ``n_layers`` copies with seeded uniform in-plane
          offsets - a turbostratic (stacking-disorder) model.
        - an explicit ``sequence`` of per-layer offsets for anything
          else (ABC and beyond).

        Layers are van-der-Waals stacked: intra-layer bonds are
        duplicated per layer and no inter-layer bonds are created.

        Parameters
        ----------
        mode : str, optional
            One of ``AA``, ``AB``, ``serrated``, ``staggered``,
            ``random``. Ignored when ``sequence`` is given.
        interlayer : float, optional
            Spacing between successive layer planes in Angstrom, by
            default 3.35 (graphite-like; typical COFs are 3.3-3.6).
        offset : tuple[float, float] or None, optional
            Fractional in-plane offset of the second layer, overriding
            the mode default. Only for the two-layer modes.
        sequence : iterable of (float, float), optional
            One fractional in-plane offset per layer; layer k sits at
            height ``k * interlayer`` and c becomes
            ``len(sequence) * interlayer``. Mutually exclusive with
            ``offset`` and non-default ``mode``.
        n_layers : int or None, optional
            Number of layers for ``mode="random"`` (>= 2); not
            applicable to the preset modes.
        seed : int or None, optional
            Seed for the random offsets; a fixed seed reproduces the
            same disorder model.

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
        >>> abc = layer.stack(sequence=[(0, 0), (1/3, 2/3), (2/3, 1/3)])
        >>> disordered = layer.stack(mode="random", n_layers=6, seed=42)
        """
        if interlayer <= 0:
            raise ValueError(f"Interlayer spacing must be positive, got {interlayer}.")
        offsets = self._stacking_offsets(mode, offset, sequence, n_layers, seed)
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
        # a puckered layer thicker than the spacing overlaps its own
        # stacked images; legal (interdigitated stackings exist) but
        # rarely intended, so say so instead of failing silently later
        if thickness > interlayer:
            logger.warning(
                f"Layer {self.name!r} is {thickness:.2f} A thick but the "
                f"interlayer spacing is {interlayer:.2f} A; stacked layers "
                "will interpenetrate. Check the result with min_contact()."
            )
        count = len(offsets)
        new_cell = cell.copy()
        new_cell[2] = [0.0, 0.0, count * interlayer]
        stacked = networkx.Graph(cell=new_cell)
        n_atoms = len(self.graph)
        # duplicated tags and slot ids stay unique per layer so tag-pair
        # semantics and placed-SBU identity survive
        tag_base = max(
            (data["tag"] for _, data in self.graph.nodes(data=True)), default=0
        )
        slot_base = (
            max(
                (data.get("slot", 0) for _, data in self.graph.nodes(data=True)),
                default=0,
            )
            + 1
        )
        for k, (fx, fy) in enumerate(offsets):
            shift = fx * cell[0] + fy * cell[1]
            shift[2] += k * interlayer
            for node, data in self.graph.nodes(data=True):
                copied = dict(data)
                copied["coord"] = np.asarray(data["coord"], dtype=float) + shift
                if copied["tag"] > 0:
                    copied["tag"] += k * tag_base
                if "slot" in copied:
                    copied["slot"] += k * slot_base
                stacked.add_node(node + k * n_atoms, **copied)
            for i, j, data in self.graph.edges(data=True):
                stacked.add_edge(i + k * n_atoms, j + k * n_atoms, **data)
        label = mode if sequence is None else f"stack{count}"
        return Framework(stacked, name=f"{self.name}_{label}")

    @staticmethod
    def _stacking_offsets(
        mode: str,
        offset: tuple[float, float] | None,
        sequence: Iterable[tuple[float, float]] | None,
        n_layers: int | None,
        seed: int | None,
    ) -> list[tuple[float, float]]:
        """Resolve the stacking arguments into one offset per layer."""
        if sequence is not None:
            if mode != "AA" or offset is not None:
                raise ValueError(
                    "Give either an explicit sequence or mode/offset, not both."
                )
            if n_layers is not None:
                raise ValueError("sequence fixes the layer count; drop n_layers.")
            offsets = [(float(fx), float(fy)) for fx, fy in sequence]
            if not offsets:
                raise ValueError("sequence needs at least one layer offset.")
            return offsets
        if mode == "random":
            if n_layers is None or n_layers < 2:
                raise ValueError("random stacking needs n_layers >= 2.")
            if offset is not None:
                raise ValueError("random stacking draws its own offsets.")
            rng = np.random.default_rng(seed)
            return [(0.0, 0.0)] + [
                (float(fx), float(fy)) for fx, fy in rng.random((n_layers - 1, 2))
            ]
        if mode not in STACKING_OFFSETS:
            raise ValueError(
                f"Unknown stacking mode {mode!r}; expected one of "
                f"{sorted(STACKING_OFFSETS)} or 'random'."
            )
        if n_layers is not None:
            raise ValueError(
                "n_layers only applies to mode='random'; repeat preset "
                "stackings by passing an explicit sequence."
            )
        if mode == "AA":
            if offset is not None:
                raise ValueError("AA stacking takes no in-plane offset.")
            return [(0.0, 0.0)]
        second = offset if offset is not None else STACKING_OFFSETS[mode]
        assert second is not None  # every non-AA preset has a default
        return [(0.0, 0.0), (float(second[0]), float(second[1]))]
