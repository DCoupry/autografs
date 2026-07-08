"""
Interactive command-line wizard for building MOF and COF structures.

Console entry point: ``autografs``. A guided session walks through the
common workflow - pick a topology, pick an SBU for each slot type,
build, stack 2D layers into a COF, export - without writing a script.
Browsing menus expose the topology and SBU libraries with the same
compatibility sieve the builder uses.

Non-interactive building (``--topology``/``--sbu`` flags) is a known
follow-up, deliberately left out: addressing slot types from a flag is
ambiguous on nets with several same-connectivity orbits, and scripted
use is already served by the Python API and ``Autografs.build_all``.

Examples
--------
$ autografs
$ autografs --xyz my_sbus.xyz
$ autografs --topofile tests/data/topologies_fixture.json
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import questionary
from rich.console import Console
from rich.table import Table

# the autografs package itself (and pymatgen behind it) is imported
# inside functions: Session.gen defers the expensive library setup
# until a menu actually needs it
if TYPE_CHECKING:
    from collections.abc import Mapping

    from autografs.builder import Autografs
    from autografs.fragment import Fragment
    from autografs.framework import Framework
    from autografs.topology import Topology

logger = logging.getLogger(__name__)

console = Console()

# above this many candidates, an arrow-key select becomes unusable and
# the prompt switches to type-to-search autocompletion
MAX_SELECT_CHOICES = 15

STRICT_MAX_RMSD = 0.5


# ----------------------------------------------------------------------
# pure helpers (unit-tested, no prompts)
# ----------------------------------------------------------------------


def connectivity(fragment: Fragment) -> int:
    """Number of connection points (dummy atoms) of a fragment."""
    return len(fragment.atoms.indices_from_symbol("X"))


def slot_label(fragment: Fragment, n_slots: int, show_orbit: bool = False) -> str:
    """Human-readable description of a slot type, e.g.
    ``3-connected, D3h - 2 slots``."""
    plural = "slot" if n_slots == 1 else "slots"
    label = (
        f"{connectivity(fragment)}-connected, {fragment.pointgroup}"
        f" - {n_slots} {plural}"
    )
    if show_orbit and fragment.equivalence_class is not None:
        label += f" (orbit {fragment.equivalence_class})"
    return label


def sorted_slot_types(topology: Topology) -> list[Fragment]:
    """Slot types in prompt order: high connectivity first (nodes
    before linkers), then point group and orbit for stability."""
    return sorted(
        topology.mappings,
        key=lambda f: (-connectivity(f), f.pointgroup, f.equivalence_class or 0),
    )


def slot_labels(topology: Topology) -> dict[Fragment, str]:
    """One label per slot type, with the orbit id appended whenever two
    slot types would otherwise read identically."""
    keys = sorted_slot_types(topology)
    shapes = Counter((connectivity(key), key.pointgroup) for key in keys)
    return {
        key: slot_label(
            key,
            len(topology.mappings[key]),
            show_orbit=shapes[(connectivity(key), key.pointgroup)] > 1,
        )
        for key in keys
    }


def default_output_name(topology_name: str, sbu_names: list[str]) -> str:
    """Suggested CIF filename for a build, filesystem-safe."""
    parts = [topology_name, *dict.fromkeys(sbu_names)]
    stem = "_".join(re.sub(r"[^\w.-]", "_", part) for part in parts)
    return f"{stem}.cif"


@dataclass(frozen=True)
class TopoInfo:
    """Cheap searchable metadata about one library topology."""

    name: str
    is_2d: bool
    connectivities: tuple[int, ...]


def build_topology_index(topologies: Mapping[str, Topology]) -> list[TopoInfo]:
    """Metadata index of a topology library, for filtering prompts.

    A LazyTopologyLibrary (see autografs.topology_io) keeps the parsed
    JSON payload in ``_raw``; scanning that costs nothing, whereas
    materializing ~2500 Topology objects takes ~10 s. Plain dicts
    (legacy pickle libraries) are iterated as-is.
    """
    raw = getattr(topologies, "_raw", None)
    infos = []
    if raw is not None:
        for name, data in raw.items():
            conns = {slot["species"].count("X") for slot in data["slots"]}
            infos.append(
                TopoInfo(name, bool(data.get("is_2d", False)), tuple(sorted(conns)))
            )
    else:
        for name, topology in topologies.items():
            conns = {connectivity(slot_type) for slot_type in topology.mappings}
            infos.append(TopoInfo(name, bool(topology.is_2d), tuple(sorted(conns))))
    return sorted(infos, key=lambda info: info.name)


def topology_summary_table(topology: Topology) -> Table:
    """Rich table describing one topology: cell, symmetry, slot types."""
    a, b, c, alpha, beta, gamma = topology.cell.parameters
    table = Table(title=f"Topology {topology.name}", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    if topology.is_2d:
        table.add_row("dimensionality", "2D layer (c is slab padding)")
        table.add_row("plane group", str(topology.spacegroup_number))
    else:
        table.add_row("dimensionality", "3D")
        table.add_row("space group", str(topology.spacegroup_number))
    table.add_row("cell", f"a={a:.2f} b={b:.2f} c={c:.2f} Angstrom")
    table.add_row("angles", f"alpha={alpha:.1f} beta={beta:.1f} gamma={gamma:.1f} deg")
    for label in slot_labels(topology).values():
        table.add_row("slot type", label)
    return table


# ----------------------------------------------------------------------
# prompt plumbing
# ----------------------------------------------------------------------


def _validate_positive_float(text: str) -> bool | str:
    try:
        value = float(text)
    except ValueError:
        return "Enter a number"
    return value > 0 or "Enter a positive number"


def _validate_positive_int(text: str) -> bool | str:
    try:
        value = int(text)
    except ValueError:
        return "Enter an integer"
    return value > 0 or "Enter a positive integer"


def _pick_name(message: str, names: list[str]) -> str | None:
    """Type-to-search prompt over a list of names; None on cancel."""
    return questionary.autocomplete(
        message,
        choices=names,
        match_middle=True,
        validate=lambda text: text in names or "Pick one of the suggested names",
    ).ask()


def _pick_max_rmsd() -> float | None | str:
    """Prompt for the alignment gate; returns 'cancel' on Ctrl-C/ESC
    (None already means 'no gate')."""
    pick = questionary.select(
        "Alignment quality gate (max RMSD)?",
        choices=[
            "No limit - always build (default)",
            f"{STRICT_MAX_RMSD} - strict, rejects distorted fits",
            "Custom...",
        ],
    ).ask()
    if pick is None:
        return "cancel"
    if pick.startswith("No limit"):
        return None
    if pick.startswith(str(STRICT_MAX_RMSD)):
        return STRICT_MAX_RMSD
    text = questionary.text(
        "max RMSD (0 = perfect shape match, 2 = opposite):",
        validate=_validate_positive_float,
    ).ask()
    if text is None:
        return "cancel"
    return float(text)


# ----------------------------------------------------------------------
# session
# ----------------------------------------------------------------------


@dataclass
class Session:
    """Lazily initialized Autografs instance plus derived caches."""

    xyzfile: str | None = None
    topofile: str | None = None
    _gen: Autografs | None = field(default=None, repr=False)
    _index: list[TopoInfo] | None = field(default=None, repr=False)

    @property
    def gen(self) -> Autografs:
        if self._gen is None:
            from autografs import Autografs, AutografsError

            try:
                with console.status(
                    "Loading SBU and topology libraries (about 10 s)..."
                ):
                    self._gen = Autografs(xyzfile=self.xyzfile, topofile=self.topofile)
            except (OSError, ValueError, AutografsError) as exc:
                console.print(f"[red]Could not load the libraries:[/red] {exc}")
                console.print(
                    "Generate a topology library with "
                    "[bold]autografs-topologies --use_rcsr -o topologies.json.gz"
                    "[/bold] and point at it with [bold]--topofile[/bold]."
                )
                raise SystemExit(1) from exc
        return self._gen

    @property
    def index(self) -> list[TopoInfo]:
        if self._index is None:
            self._index = build_topology_index(self.gen.topologies)
        return self._index


# ----------------------------------------------------------------------
# wizard steps
# ----------------------------------------------------------------------


def filter_topology_names(session: Session) -> list[str] | None:
    """Dimensionality + connectivity filter; None on cancel."""
    infos = session.index
    n_2d = sum(info.is_2d for info in infos)
    dim = questionary.select(
        "Which nets?",
        choices=[
            f"All ({len(infos)})",
            f"3D only ({len(infos) - n_2d})",
            f"2D layers only ({n_2d}) - stackable into COFs",
        ],
    ).ask()
    if dim is None:
        return None
    if dim.startswith("3D"):
        infos = [info for info in infos if not info.is_2d]
    elif dim.startswith("2D"):
        infos = [info for info in infos if info.is_2d]
    all_conns = sorted({c for info in infos for c in info.connectivities})
    pick = questionary.select(
        "Filter by slot connectivity?",
        choices=["Any connectivity"] + [f"has a {c}-connected slot" for c in all_conns],
    ).ask()
    if pick is None:
        return None
    if pick != "Any connectivity":
        wanted = int(pick.removeprefix("has a ").split("-")[0])
        infos = [info for info in infos if wanted in info.connectivities]
    return [info.name for info in infos]


def choose_topology(session: Session) -> Topology | None:
    """Filter, search, inspect, confirm; None on cancel."""
    while True:
        names = filter_topology_names(session)
        if names is None:
            return None
        if not names:
            console.print("[yellow]No topology matches those filters.[/yellow]")
            continue
        name = _pick_name("Topology name (type to search):", names)
        if name is None:
            return None
        topology = session.gen.topologies[name]
        console.print(topology_summary_table(topology))
        use = questionary.confirm("Use this topology?", default=True).ask()
        if use is None:
            return None
        if use:
            return topology


def choose_sbus(session: Session, topology: Topology) -> dict[Fragment, str] | None:
    """One SBU per slot type, from the compatibility sieve; None on
    cancel or when a slot type has no compatible SBU at all."""
    options = session.gen.list_building_units(sieve=topology.name)
    labels = slot_labels(topology)
    unfillable = [
        labels[slot_type]
        for slot_type in sorted_slot_types(topology)
        if not options.get(slot_type)
    ]
    if unfillable:
        console.print("[yellow]No SBU in the library fits these slot types:[/yellow]")
        for label in unfillable:
            console.print(f"  - {label}")
        console.print("Add candidates with [bold]--xyz your_sbus.xyz[/bold].")
        return None
    chosen: dict[Fragment, str] = {}
    for slot_type in sorted_slot_types(topology):
        candidates = options[slot_type]
        message = f"SBU for {labels[slot_type]}:"
        if len(candidates) <= MAX_SELECT_CHOICES:
            pick = questionary.select(message, choices=candidates).ask()
        else:
            pick = _pick_name(f"{message} (type to search)", candidates)
        if pick is None:
            return None
        chosen[slot_type] = pick
    return chosen


def choose_build_options() -> tuple[bool, float | None] | None:
    """(refine_cell, max_rmsd); None on cancel."""
    refine_cell = questionary.confirm(
        "Refine the cell parameters? (recommended)", default=True
    ).ask()
    if refine_cell is None:
        return None
    max_rmsd = _pick_max_rmsd()
    if max_rmsd == "cancel":
        return None
    return refine_cell, max_rmsd


def run_build(
    session: Session,
    topology: Topology,
    mappings: dict[Fragment, str],
    refine_cell: bool,
    max_rmsd: float | None,
) -> tuple[Framework, dict[Fragment, str]] | None:
    """Build with an interactive recovery loop on alignment failure.

    Returns the framework together with the mappings actually used
    (the recovery path can swap SBUs), or None on abort.
    """
    from autografs import AlignmentError, AutografsError

    while True:
        try:
            with console.status(f"Building on {topology.name}..."):
                framework = session.gen.build(
                    topology,
                    mappings=mappings,
                    refine_cell=refine_cell,
                    max_rmsd=max_rmsd,
                )
            console.print(f"[green]Built[/green] {framework!r}")
            return framework, mappings
        except AlignmentError as exc:
            console.print(f"[red]Alignment failed:[/red] {exc}")
            choices = []
            if max_rmsd is not None:
                choices.append("Retry without the max RMSD gate")
            choices += ["Pick different SBUs", "Back to main menu"]
            action = questionary.select("What now?", choices=choices).ask()
            if action == "Retry without the max RMSD gate":
                max_rmsd = None
            elif action == "Pick different SBUs":
                new_mappings = choose_sbus(session, topology)
                if new_mappings is None:
                    return None
                mappings = new_mappings
            else:
                return None
        except (AutografsError, ValueError) as exc:
            console.print(f"[red]Build failed:[/red] {exc}")
            return None


def maybe_stack(framework: Framework, topology: Topology) -> Framework:
    """Offer COF stacking for 2D layer builds; failures keep the layer."""
    if not topology.is_2d:
        return framework
    from autografs.exceptions import StackingError
    from autografs.framework import DEFAULT_INTERLAYER

    do_stack = questionary.confirm(
        "This is a 2D layer - stack it into a bulk COF crystal?", default=True
    ).ask()
    if not do_stack:
        return framework
    mode = questionary.select(
        "Stacking mode:",
        choices=[
            "AA - eclipsed (most common)",
            "AB - offset by (1/3, 2/3)",
            "serrated - offset by (1/2, 0)",
            "staggered - offset by (1/2, 1/2)",
        ],
    ).ask()
    if mode is None:
        return framework
    mode = mode.split()[0]
    text = questionary.text(
        "Interlayer spacing in Angstrom (typical COFs: 3.3-3.6):",
        default=str(DEFAULT_INTERLAYER),
        validate=_validate_positive_float,
    ).ask()
    if text is None:
        return framework
    try:
        stacked = framework.stack(mode=mode, interlayer=float(text))
    except StackingError as exc:
        console.print(
            f"[yellow]Stacking failed, keeping the single layer:[/yellow] {exc}"
        )
        return framework
    console.print(f"[green]Stacked[/green] {stacked!r}")
    return stacked


def export_step(framework: Framework, default_name: str) -> None:
    """Export loop: CIF, GULP input, ASE viewer; repeats until Done."""
    while True:
        action = questionary.select(
            "Export:",
            choices=[
                "Write CIF",
                "Write GULP input (UFF4MOF optimization)",
                "Open in the ASE viewer",
                "Done",
            ],
        ).ask()
        if action is None or action == "Done":
            return
        if action == "Write CIF":
            path = questionary.text("Output path:", default=default_name).ask()
            if not path:
                continue
            try:
                written = framework.write_cif(path)
            except OSError as exc:
                console.print(f"[red]Could not write {path}:[/red] {exc}")
                continue
            console.print(f"Wrote [bold]{written.resolve()}[/bold]")
        elif action.startswith("Write GULP"):
            framework.to_gulp(write_to_file=True)
            console.print(
                f"Wrote [bold]{Path.cwd() / (framework.name + '.gin')}[/bold]"
            )
        else:
            try:
                framework.view()
            except Exception as exc:  # viewer backends fail in many ways
                console.print(f"[yellow]Could not open the viewer:[/yellow] {exc}")


def build_wizard(session: Session) -> None:
    """The main flow: topology -> SBUs -> options -> build -> stack -> export."""
    topology = choose_topology(session)
    if topology is None:
        return
    mappings = choose_sbus(session, topology)
    if mappings is None:
        return
    options = choose_build_options()
    if options is None:
        return
    refine_cell, max_rmsd = options
    result = run_build(session, topology, mappings, refine_cell, max_rmsd)
    if result is None:
        return
    framework, mappings = result
    framework = maybe_stack(framework, topology)
    export_step(framework, default_output_name(topology.name, list(mappings.values())))


def browse_topologies(session: Session) -> None:
    """Inspect topologies: summary plus compatible SBUs per slot type."""
    while True:
        names = filter_topology_names(session)
        if names is None:
            return
        if not names:
            console.print("[yellow]No topology matches those filters.[/yellow]")
            continue
        name = _pick_name("Topology name (type to search):", names)
        if name is None:
            return
        topology = session.gen.topologies[name]
        console.print(topology_summary_table(topology))
        options = session.gen.list_building_units(sieve=name)
        labels = slot_labels(topology)
        table = Table(title="Compatible building units")
        table.add_column("slot type")
        table.add_column("SBUs")
        for slot_type in sorted_slot_types(topology):
            sbus = options.get(slot_type, [])
            table.add_row(
                labels[slot_type], ", ".join(sbus) if sbus else "[red]none[/red]"
            )
        console.print(table)
        if not questionary.confirm("Look at another topology?", default=True).ask():
            return


def browse_sbus(session: Session) -> None:
    """Inspect SBUs: composition, connectivity, compatible net count."""
    names = sorted(session.gen.sbu)
    while True:
        name = _pick_name("SBU name (type to search):", names)
        if name is None:
            return
        sbu = session.gen.sbu[name]
        table = Table(title=f"SBU {name}", show_header=False)
        table.add_column(style="bold")
        table.add_column()
        table.add_row("composition", sbu.atoms.composition.formula)
        table.add_row("connection points", str(connectivity(sbu)))
        table.add_row("dummy point group", sbu.pointgroup)
        console.print(table)
        with console.status("Sieving the topology library (slow the first time)..."):
            compatible = session.gen.list_topologies(sieve=name)
        console.print(
            f"Fits [bold]{len(compatible)}[/bold] of "
            f"{len(session.gen.topologies)} topologies."
        )
        if not questionary.confirm("Look at another SBU?", default=True).ask():
            return


def batch_build(session: Session) -> None:
    """Minimal front-end for Autografs.build_all: subset, cap, export."""
    text = questionary.text(
        "Topologies to attempt (comma-separated, empty = all):"
    ).ask()
    if text is None:
        return
    subset = [name.strip() for name in text.split(",") if name.strip()] or None
    if subset:
        unknown = [name for name in subset if name not in session.gen.topologies]
        if unknown:
            console.print(f"[red]Unknown topologies:[/red] {', '.join(unknown)}")
            return
    per_topology = questionary.text(
        "Max SBU combinations per topology:",
        default="5",
        validate=_validate_positive_int,
    ).ask()
    if per_topology is None:
        return
    max_rmsd = _pick_max_rmsd()
    if max_rmsd == "cancel":
        return
    outdir = questionary.text("Output directory:", default="frameworks").ask()
    if outdir is None:
        return
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # build_all reports its own tqdm progress; no spinner around it
    frameworks = session.gen.build_all(
        topology_subset=subset,
        max_per_topology=int(per_topology),
        max_rmsd=max_rmsd,
        n_jobs=1,
    )
    if not frameworks:
        console.print("[yellow]No structure could be built.[/yellow]")
        return
    table = Table(title=f"{len(frameworks)} structures written to {outdir}")
    table.add_column("topology")
    table.add_column("formula")
    table.add_column("file")
    for i, framework in enumerate(frameworks):
        # frameworks are named after their topology alone; the counter
        # keeps different SBU combinations from overwriting each other
        path = outdir / f"{framework.name}_{i}.cif"
        framework.write_cif(path)
        table.add_row(framework.name, framework.formula, str(path))
    console.print(table)


def main_menu(session: Session) -> None:
    while True:
        choice = questionary.select(
            "What would you like to do?",
            choices=[
                "Build a structure",
                "Browse topologies",
                "Browse building units",
                "Batch build (all compatible combinations)",
                "Quit",
            ],
        ).ask()
        if choice is None or choice == "Quit":
            return
        if choice == "Build a structure":
            build_wizard(session)
        elif choice == "Browse topologies":
            browse_topologies(session)
        elif choice == "Browse building units":
            browse_sbus(session)
        else:
            batch_build(session)


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("AuToGraFS")
    except PackageNotFoundError:
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    """Console entry point: autografs."""
    parser = argparse.ArgumentParser(
        prog="autografs",
        description="Interactively generate MOF and COF structures.",
    )
    parser.add_argument(
        "--xyz",
        type=str,
        default=None,
        help="XYZ file with custom SBUs; same-name entries override defaults.",
    )
    parser.add_argument(
        "--topofile",
        type=str,
        default=None,
        help=(
            "topology library (.json / .json.gz / legacy .pkl); "
            "defaults to the packaged RCSR library."
        ),
    )
    parser.add_argument("--version", action="version", version=_version())
    args = parser.parse_args(argv)
    # the wizard narrates through prompts; keep library logging quiet
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    session = Session(xyzfile=args.xyz, topofile=args.topofile)
    console.print("[bold]AuToGraFS[/bold] - interactive framework generator")
    try:
        main_menu(session)
    except (KeyboardInterrupt, EOFError):
        console.print()
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
