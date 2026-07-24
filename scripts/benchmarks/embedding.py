"""Compare a net's idealized embedding proportions against real crystals.

A build places every SBU at its blueprint slot centre and only ever
varies the cell's free parameters, so the *relative* placement of units
is whatever the net's idealized (maximum-symmetry, near-equal-edge)
embedding chose. For a real material that can be off, and no choice of
cell can repair a proportion.

This driver measures the gap, without needing a build. For a structure
and the library net it identifies as, both embeddings are reduced to
their **contracted quotient graph** (caps pruned, ditopic linkers and
blueprint edge centers contracted away, so the two are comparable
however they are decorated) and every edge gets a dimensionless
*reduced length*

    lambda = d / (fold * V_cell / n_vertices) ** (1/3)

- ``d``          straight-line distance between the two endpoint
                 centres, through the edge's periodic image;
- ``V/n``        cell volume per surviving vertex, the natural length
                 unit of the embedding;
- ``fold``       interpenetration, so catenated frameworks are compared
                 as the single nets they are.

lambda is invariant under uniform scaling *and* under supercell choice
(V and n scale together), which is what makes an idealized blueprint
and a real crystal in an unrelated setting directly comparable. The
two multisets need not be the same size - one cell may be a multiple of
the other - so they are compared on a normalized rank axis.

Two numbers come out:

- **size**   ``mean(lambda_real) / mean(lambda_ideal) - 1``. The
             embedding packs the same net at a different density than
             the real material. Because a build fixes ``d`` by closing
             bonds, this predicts the built cell volume error directly:
             ``V_built / V_real = (lambda_real / lambda_ideal) ** 3``.
- **shape**  the same comparison after dividing each multiset by its
             own mean: the spread a single scale factor *cannot* fix.

``--rebuild`` additionally builds each structure from its own harvested
fragments (gated by exact net verification) and records the realized
cell, closing the loop on the volume-ratio prediction above. Coverage
is much lower than the measurement itself - most structures have no
compatible fragment-to-slot mapping - so it is off by default. A
2-connected slot type that no harvested fragment fits is rebuilt
**empty** (#179), which is what brings the edge-decorated nets
(``tbo``-class) into scoring range at all.

Deterministic (no sampling, sorted corpus order); machine-readable JSON
out plus a summary table on stdout.

Usage:
    python scripts/benchmarks/embedding.py "corpus/*.cif" -o embedding.json
    python scripts/benchmarks/embedding.py corpus_dir --rebuild --limit 200
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np

from autografs import Autografs
from autografs.exceptions import AutografsError
from autografs.net import contract_quotient_edges, topology_quotient_edges

# normalized ranks at which the two reduced-length distributions are
# compared; offset from 0 and 1 so a grid point never lands exactly on
# a step boundary of either multiset
RANK_GRID = (np.arange(101) + 0.5) / 101.0

# fragment-to-slot assignments tried per net candidate under --rebuild
MAX_MAPPINGS_PER_NET = 8


def reduced_lengths(
    centers: dict[int, np.ndarray],
    edges: Counter,
    matrix: np.ndarray,
    volume: float,
    fold: int = 1,
) -> np.ndarray | None:
    """Sorted reduced edge lengths of one embedding.

    Parameters
    ----------
    centers : dict[int, np.ndarray]
        Fractional position of every quotient vertex, keyed as in
        ``edges``. Vertices contracted away are simply never read.
    edges : Counter
        Labeled quotient graph, uncontracted.
    matrix : np.ndarray
        (3, 3) cell matrix the fractional centres refer to.
    volume : float
        Cell volume, Angstrom^3.
    fold : int, optional
        Interpenetration fold; the vertex count is divided by it so a
        catenated structure is measured as the single net it repeats.

    Returns
    -------
    np.ndarray or None
        Ascending reduced lengths, one entry per contracted edge
        (multiplicities included), or None when nothing survives
        contraction (a pure cycle, or an empty graph).
    """
    contracted = contract_quotient_edges(edges)
    if not contracted:
        return None
    vertices = {node for edge in contracted for node in edge[:2]}
    reference = (fold * volume / len(vertices)) ** (1.0 / 3.0)
    lengths: list[float] = []
    for (node_a, node_b, voltage), multiplicity in contracted.items():
        if node_a not in centers or node_b not in centers:
            return None
        delta = (
            centers[node_a] - centers[node_b] - np.asarray(voltage, dtype=float)
        ) @ matrix
        lengths.extend([float(np.linalg.norm(delta))] * multiplicity)
    return np.sort(np.asarray(lengths)) / reference


def _on_rank_grid(values: np.ndarray) -> np.ndarray:
    """Sample an ascending multiset at RANK_GRID by nearest rank.

    Two embeddings of the same net have edge multiplicities in a fixed
    ratio (one cell is a multiple of the other), so sampling both on a
    normalized rank axis lines their orbits up whatever the cell
    choice. Nearest rank, not interpolation: a step between two edge
    orbits is real structure, not something to smooth over.
    """
    index = np.minimum((RANK_GRID * len(values)).astype(int), len(values) - 1)
    return values[index]


def compare(real: np.ndarray, ideal: np.ndarray) -> dict:
    """Size and shape mismatch between two reduced-length multisets."""
    real_grid = _on_rank_grid(real)
    ideal_grid = _on_rank_grid(ideal)
    size = float(real.mean() / ideal.mean())
    shape = (real_grid / real.mean()) / (ideal_grid / ideal.mean())
    return {
        "size_ratio": size,
        "size_error": abs(size - 1.0),
        "shape_error": float(np.abs(shape - 1.0).max()),
        "predicted_volume_ratio": size**3,
        "mean_lambda_real": float(real.mean()),
        "mean_lambda_ideal": float(ideal.mean()),
        "n_edges_real": int(len(real)),
        "n_edges_ideal": int(len(ideal)),
    }


# fractional centres are rounded to this many decimals before wrapping,
# matching autografs.net._WRAP_DECIMALS: a unit sitting exactly on a
# cell boundary (a node on a special position, which is the common case
# in a high-symmetry net) must wrap to the same side the quotient
# graph's voltages were measured against
_WRAP_DECIMALS = 6


def real_centers(result) -> dict[int, np.ndarray]:
    """Home-cell heavy-atom centroid of every building unit, fractional.

    Hydrogens are left out: they hang off a linker asymmetrically and
    would shift its centre away from the ring the blueprint slot sits
    on. Each unit is unwrapped onto its own first atom before
    averaging, so a unit straddling a cell boundary keeps its shape,
    then the centroid is wrapped back into the home cell - the gauge
    ``deconstruct`` measured the quotient voltages in, so a centre and
    a voltage compose into the right periodic image.
    """
    structure = result.structure
    frac = np.asarray(structure.frac_coords)
    centers: dict[int, np.ndarray] = {}
    for index, unit in enumerate(result.units):
        heavy = [
            i for i in unit.atom_indices if structure[i].specie.symbol != "H"
        ] or unit.atom_indices
        block = frac[heavy]
        block = block - np.round(block - block[0])
        centroid = np.round(block.mean(axis=0), _WRAP_DECIMALS)
        centers[index] = centroid - np.floor(centroid)
    return centers


def blueprint_centers(topology) -> dict[int, np.ndarray]:
    """Slot centres (dummy centroid), fractional - where a build puts
    the SBU it maps onto that slot."""
    centers: dict[int, np.ndarray] = {}
    for index, slot in enumerate(topology.slots):
        dummies = [i for i, site in enumerate(slot.atoms) if site.specie.symbol == "X"]
        cart = slot.atoms.cart_coords[dummies]
        centers[index] = topology.cell.get_fractional_coords(cart).mean(axis=0)
    return centers


def _candidate_mappings(topology, fragments):
    """Fragment-per-slot-type assignments compatible by geometry.

    A 2-connected slot type that *nothing* fits is offered as **empty**
    (#179) rather than abandoning the net. That is not a fallback of
    convenience: many blueprints subdivide their edges with
    2-connected centers the real material does not have - HKUST-1
    bonds its paddlewheels straight onto BTC while ``tbo`` decorates
    every edge - and an absent decoration is exactly what a
    contracted-tier identification means. Without this, the whole
    edge-decorated family (``tbo``-class, and the ``n_free`` 3-5 nets
    that carry the free proportions #174 is about) can never rebuild
    faithfully inside the benchmark, so relaxation has nothing real to
    be scored against.

    Emptying is offered only where nothing fits, never as an extra
    option alongside a fitting fragment: it would multiply the
    combination count for nets that already build, and a decoration
    the material *does* have should be placed.
    """
    slot_types = list(topology.mappings)
    options = []
    for slot_type in slot_types:
        fitting = [
            f for f in fragments.values() if f.has_compatible_symmetry(slot_type)
        ]
        if not fitting:
            if len(slot_type.atoms.indices_from_symbol("X")) != 2:
                return  # only 2-connected slots may be emptied
            fitting = [None]
        options.append(fitting)
    combos = itertools.product(*options)
    for combo in itertools.islice(combos, MAX_MAPPINGS_PER_NET):
        yield dict(zip(slot_types, combo, strict=True))


def is_buildable(topology, fragments) -> bool:
    """Would the forward pipeline even attempt this structure?

    The population that matters for #174 is the one that *builds* -
    right topology, right composition, wrong packing. A structure whose
    units cannot be placed on the blueprint's slots at all fails for an
    unrelated reason (a square-planar node measured against a
    tetrahedral dia slot) and its reduced lengths say nothing about the
    embedding. Same geometric test the builder applies, before any cell
    work - including the empty-slot option for undecorated edges.
    """
    return next(_candidate_mappings(topology, fragments), None) is not None


def bond_residuals(framework) -> dict:
    """Deviation of every inter-SBU bond from its covalent target.

    This is what the cell optimizer minimizes, so it separates the two
    ways a build can be wrong. Residuals near zero next to a large
    ``size_error`` is the #174 signature: the objective is *already
    satisfied* and the packing is still wrong, so freeing more degrees
    of freedom against the same objective has nothing to pull on. Large
    residuals instead mean the alignment itself failed, and the
    embedding is not the story.
    """
    from pymatgen.analysis.local_env import CovalentRadius

    graph = framework.graph
    cell = np.asarray(graph.graph["cell"], dtype=float)
    inverse = np.linalg.inv(cell)
    deviations: list[float] = []
    for node_a, node_b in graph.edges():
        data_a = graph.nodes[node_a]
        data_b = graph.nodes[node_b]
        delta = np.asarray(data_a["coord"], float) - np.asarray(data_b["coord"], float)
        crossing = np.round(delta @ inverse)
        if data_a.get("slot") == data_b.get("slot") and not np.any(crossing):
            # internal to one SBU: fixed by the fragment, not the cell.
            # A same-slot bond that *does* cross is a unit bonded to its
            # own periodic image - real, and what emptied edge centers
            # produce (#179), so it must be measured.
            continue
        delta -= crossing @ cell
        target = CovalentRadius.radius.get(
            data_a["symbol"], 0.0
        ) + CovalentRadius.radius.get(data_b["symbol"], 0.0)
        deviations.append(abs(float(np.linalg.norm(delta)) - target))
    if not deviations:
        return {}
    ordered = sorted(deviations)
    return {
        "median": statistics.median(ordered),
        "max": ordered[-1],
        "n_bonds": len(ordered),
    }


def _rebuild(
    mofgen, topology, result, max_rmsd: float, relax_embedding: bool = False
) -> dict | None:
    """Build from the structure's own fragments; verified net only.

    No overlap gate: a compressed packing is exactly what this
    benchmark is here to measure, so it must not be rejected.
    """
    for mappings in _candidate_mappings(topology, result.fragments):
        try:
            framework = mofgen.build(
                topology,
                mappings=mappings,
                max_rmsd=max_rmsd,
                verify_net=True,
                relax_embedding=relax_embedding,
            )
        except AutografsError:
            continue
        built = framework.structure
        experimental = result.structure
        return {
            "abc": [round(x, 4) for x in built.lattice.abc],
            "angles": [round(x, 2) for x in built.lattice.angles],
            "volume_per_atom": built.volume / len(built),
            "volume_ratio": (built.volume / len(built))
            / (experimental.volume / len(experimental) * result.n_periodic_components),
            "min_contact": framework.min_contact(),
            "bond_residual": bond_residuals(framework),
            # which blueprint slots were rebuilt empty, so a result on
            # an edge-decorated net is interpretable
            "empty_slots": sorted(framework.graph.graph.get("empty_slots", ())),
            # a compatible mapping is not necessarily the *right* one -
            # a net with several interchangeable slot types can take a
            # fragment in the wrong place. Only a rebuild that
            # reproduces the material's own formula is evidence about
            # its packing; the reduced formula is invariant to cell
            # multiples and to interpenetration fold alike.
            "composition_match": (
                built.composition.reduced_formula
                == experimental.composition.reduced_formula
            ),
            "formula": built.composition.reduced_formula,
            "formula_experimental": experimental.composition.reduced_formula,
        }
    return None


def measure_one(
    mofgen: Autografs,
    source: Path,
    rebuild: bool,
    max_rmsd: float,
    relax_embedding: bool = False,
) -> dict:
    """Measure one structure's embedding gap; failures are data."""
    record: dict = {"outcome": None, "net": None, "error": None}
    t0 = time.perf_counter()
    try:
        result = mofgen.deconstruct(source)
    except (AutografsError, ValueError, KeyError, IndexError) as exc:
        record["outcome"] = "deconstruction_failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = time.perf_counter() - t0
        return record
    record["fold"] = result.n_periodic_components
    record["n_units"] = len(result.units)
    if result.rod_units:
        # a rod has no finite unit centre to compare against a slot;
        # the rod pipeline pins its own axis parameter anyway (#158)
        record["outcome"] = "rod"
        record["seconds"] = time.perf_counter() - t0
        return record
    if not result.net_candidates:
        record["outcome"] = "unidentified"
        record["seconds"] = time.perf_counter() - t0
        return record
    record["net"] = result.net_candidates
    record["tier"] = result.subframework_nets[0].tier
    real = reduced_lengths(
        real_centers(result),
        result.quotient_edges,
        result.structure.lattice.matrix,
        result.structure.volume,
        result.n_periodic_components,
    )
    if real is None:
        record["outcome"] = "no_contracted_graph"
        record["seconds"] = time.perf_counter() - t0
        return record
    record["nets"] = {}
    for net in result.net_candidates:
        topology = mofgen.topologies[net]
        if topology.is_2d:
            # a layer blueprint's c is slab padding, so its cell volume
            # is arbitrary and the reduced length is meaningless
            continue
        ideal = reduced_lengths(
            blueprint_centers(topology),
            topology_quotient_edges(topology),
            topology.cell.matrix,
            topology.cell.volume,
        )
        if ideal is None:
            continue
        entry = compare(real, ideal)
        entry["spacegroup"] = topology.spacegroup_number
        entry["buildable"] = is_buildable(topology, result.fragments)
        if rebuild and entry["buildable"]:
            entry["rebuild"] = _rebuild(
                mofgen, topology, result, max_rmsd, relax_embedding
            )
        record["nets"][net] = entry
    if not record["nets"]:
        record["outcome"] = (
            "layer_net"
            if all(mofgen.topologies[net].is_2d for net in result.net_candidates)
            else "no_contracted_graph"
        )
        record["seconds"] = time.perf_counter() - t0
        return record
    # the best-fitting candidate is the fair one to score: a structure
    # is not penalized for a runner-up the identifier also allowed, and
    # a buildable candidate always beats an unbuildable one
    best = min(
        record["nets"],
        key=lambda net: (
            not record["nets"][net]["buildable"],
            record["nets"][net]["size_error"],
        ),
    )
    record["best_net"] = best
    record["buildable"] = record["nets"][best]["buildable"]
    record["size_error"] = record["nets"][best]["size_error"]
    record["shape_error"] = record["nets"][best]["shape_error"]
    record["outcome"] = "measured"
    record["seconds"] = time.perf_counter() - t0
    return record


def _crystal_system(number: int | None) -> str:
    """Crystal system of a spacegroup number, for grouping."""
    number = number or 0
    if number >= 195:
        return "cubic"
    if number >= 143:
        return "trigonal/hexagonal"
    if number >= 75:
        return "tetragonal"
    if number >= 16:
        return "orthorhombic"
    if number >= 3:
        return "monoclinic"
    if number >= 1:
        return "triclinic"
    return "unknown"


def _stats(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": ordered[min(int(0.9 * len(ordered)), len(ordered) - 1)],
        "max": ordered[-1],
    }


def _summarize(records: dict) -> dict:
    """Aggregate the measured structures over two populations.

    **buildable** is the population that matters - a compatible mapping
    exists, so the pipeline would produce this structure and its
    packing is the pipeline's responsibility. **all** additionally
    counts structures that identify but could never be placed; their
    gap is real but not attributable to the embedding.
    """
    measured = [r for r in records.values() if r["outcome"] == "measured"]
    buildable = [r for r in measured if r["buildable"]]
    summary: dict = {"n_measured": len(measured), "n_buildable": len(buildable)}
    for label, population in (("buildable", buildable), ("all", measured)):
        if not population:
            continue
        by_system: dict[str, list[float]] = {}
        for record in population:
            system = _crystal_system(record["nets"][record["best_net"]]["spacegroup"])
            by_system.setdefault(system, []).append(record["size_error"])
        summary[label] = {
            "size_error": _stats([r["size_error"] for r in population]),
            "shape_error": _stats([r["shape_error"] for r in population]),
            # signed, so the systematic direction is visible: an
            # absolute error hides whether the idealization is
            # consistently more open than real chemistry or less
            "size_bias": _stats(
                [
                    record["nets"][record["best_net"]]["size_ratio"] - 1.0
                    for record in population
                ]
            ),
            "size_error_by_system": {
                system: _stats(values) for system, values in sorted(by_system.items())
            },
        }
    rebuilds = [
        entry["rebuild"]
        for record in measured
        for entry in record["nets"].values()
        if entry.get("rebuild")
    ]
    # only rebuilds that reproduced the material's own formula say
    # anything about its packing (see _rebuild)
    faithful = [r for r in rebuilds if r["composition_match"]]
    if rebuilds:
        summary["n_rebuilt"] = len(rebuilds)
        summary["n_rebuilt_faithful"] = len(faithful)
    if faithful:
        summary["rebuilt_min_contact"] = _stats([r["min_contact"] for r in faithful])
        residuals = [
            r["bond_residual"]["median"] for r in faithful if r.get("bond_residual")
        ]
        if residuals:
            summary["rebuilt_bond_residual"] = _stats(residuals)
    return summary


def run(
    corpus: list[Path],
    mofgen: Autografs,
    rebuild: bool = False,
    max_rmsd: float = 0.5,
    verbose: bool = True,
    relax_embedding: bool = False,
) -> dict:
    """Measure every structure; returns the results payload."""
    records: dict[str, dict] = {}
    for path in sorted(corpus):
        record = measure_one(mofgen, path, rebuild, max_rmsd, relax_embedding)
        records[Path(path).name] = record
        if verbose:
            detail = (
                f"{record['best_net']:<8} size {record['size_error']:+.3f} "
                f"shape {record['shape_error']:.3f}"
                f"{'' if record['buildable'] else '  (not buildable)'}"
                if record["outcome"] == "measured"
                else ""
            )
            print(f"  {Path(path).name:<32} {record['outcome']:<22} {detail}")
    outcomes = Counter(record["outcome"] for record in records.values())
    return {
        "benchmark": "embedding",
        "rebuild": rebuild,
        "n_structures": len(records),
        "outcomes": dict(sorted(outcomes.items())),
        "summary": _summarize(records),
        "structures": records,
    }


def _collect(spec: str) -> list[Path]:
    path = Path(spec)
    if path.is_dir():
        return sorted(path.glob("*.cif"))
    if any(char in spec for char in "*?["):
        base = Path(spec).parent
        return sorted(base.glob(Path(spec).name))
    return [path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", help="directory, glob, or single CIF")
    parser.add_argument("-o", "--output", default="embedding.json")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="also build from the harvested fragments and record the cell",
    )
    parser.add_argument("--max-rmsd", type=float, default=0.5)
    parser.add_argument(
        "--relax-embedding",
        action="store_true",
        help="rebuild with embedding relaxation (#174): slot displacements "
        "+ anchor-direction objective",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="measure only the first N structures"
    )
    parser.add_argument("--topofile", default=None, help="topology library override")
    args = parser.parse_args()

    corpus = _collect(args.corpus)
    if args.limit is not None:
        corpus = corpus[: args.limit]
    if not corpus:
        raise SystemExit(f"no structures matched {args.corpus!r}")
    mofgen = Autografs(topofile=args.topofile)
    payload = run(
        corpus,
        mofgen,
        rebuild=args.rebuild,
        max_rmsd=args.max_rmsd,
        relax_embedding=args.relax_embedding,
    )
    Path(args.output).write_text(json.dumps(payload, indent=1, default=str))
    print(f"\n{payload['n_structures']} structures -> {args.output}")
    for outcome, count in payload["outcomes"].items():
        print(f"  {outcome:<24} {count}")
    summary = payload["summary"]
    for label in ("buildable", "all"):
        if label not in summary:
            continue
        block = summary[label]
        print(f"\n  {label} ({block['size_error']['n']} structures)")
        for key in ("size_error", "shape_error"):
            values = block[key]
            print(
                f"    {key:<12} median {values['median']:.3f}  "
                f"p90 {values['p90']:.3f}  max {values['max']:.3f}"
            )
        values = block["size_bias"]
        print(
            f"    {'size_bias':<12} median {values['median']:+.3f}  "
            f"({values['min']:+.3f} .. {values['max']:+.3f})"
        )
        print("    size error by crystal system:")
        for system, values in block["size_error_by_system"].items():
            print(
                f"      {system:<20} n={values['n']:<4} median {values['median']:.3f}"
                f"  p90 {values['p90']:.3f}"
            )
    if "n_rebuilt" in summary:
        print(
            f"\n  rebuilt {summary['n_rebuilt']} "
            f"({summary['n_rebuilt_faithful']} reproduced the real formula)"
        )
    if "rebuilt_min_contact" in summary:
        values = summary["rebuilt_min_contact"]
        print(
            f"  rebuilt min_contact  n={values['n']}  "
            f"median {values['median']:.2f} A  worst {values['min']:.2f} A"
        )
    if "rebuilt_bond_residual" in summary:
        values = summary["rebuilt_bond_residual"]
        print(
            f"  rebuilt bond residual  median {values['median']:.3f} A  "
            f"worst {values['max']:.3f} A"
        )


if __name__ == "__main__":
    main()
