"""Round-trip closure driver: deconstruct -> rebuild -> verify.

For every structure in a corpus, deconstruct it, then try to rebuild it
from its own extracted fragments on every identified net candidate,
gating the rebuild with the exact net-verification gate AND a
composition gate: verify_net proves the graph realizes the net, but a
net with interchangeable slot types can accept a fragment on the wrong
orbit, building a topologically correct framework that is not the input
material. Only a rebuild that also reproduces the experimental reduced
formula closes the loop. The per-structure outcome taxonomy is the
useful output - failures are data:

- ``closed``            rebuild passed verify_net AND reproduced the
                        experimental reduced formula
- ``closed_uncapped``   the rebuild reproduces the framework once its
                        **monotopic** units are set aside. Those are
                        pendant substituents and capping residues on
                        the metal-free (branch-point) path, and no
                        blueprint has a 1-connected slot to host them,
                        so a rebuild is short by exactly those atoms.
                        The net and every polytopic unit are right;
                        this is a scope limit, not a wrong material
- ``closed_wrong_composition``  verified rebuilds exist, but none with
                        the right formula even allowing for that -
                        topologically closed, chemically not the same
                        material (a fragment on the wrong orbit, which
                        is what this gate was added to catch)
- ``rebuild_failed``    every candidate/mapping combination was gated
- ``no_mapping``        no compatible fragment-to-slot assignment
- ``rod``               identified, but contains 1-periodic units this
                        driver does not rebuild: rods go through
                        ``Autografs.build_rod``, not the slot-type
                        mappings enumerated here
- ``unidentified``      deconstructed, but no library net matched
- ``deconstruction_failed``  with the error message

Deterministic (no sampling, sorted corpus order); machine-readable
JSON out plus a summary table on stdout.

Usage:
    python scripts/benchmarks/roundtrip.py "corpus/*.cif" -o results.json
    python scripts/benchmarks/roundtrip.py corpus_dir -o results.json --max-rmsd 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from _corpus import collect as _collect
from _mapping_order import graded_indices, n_combinations

# sibling driver: the same inter-unit bond measurement, so the two
# benchmarks report the identical quantity. Imported at module level
# like _mapping_order - callers put this directory on sys.path only
# for the import, so a deferred import would not find it.
from embedding import bond_residuals
from pymatgen.core.composition import Composition

from autografs import Autografs
from autografs.exceptions import AutografsError

# fragment-to-slot assignments tried per net candidate before giving
# up; multi-orbit nets with many interchangeable fragments explode
# combinatorially and a closure benchmark must terminate
MAX_MAPPINGS_PER_NET = 16


def candidate_mappings(topology, fragments: dict):
    """Yield fragment-per-slot-type assignments compatible by geometry.

    Enumerated in graded order (see ``_mapping_order``) rather than
    ``itertools.product`` order, so the budget varies every slot type
    instead of only the last one.
    """
    slot_types = list(topology.mappings)
    options = []
    for slot_type in slot_types:
        fitting = [
            fragment
            for fragment in fragments.values()
            if fragment.has_compatible_symmetry(slot_type)
        ]
        if not fitting:
            return
        options.append(fitting)
    sizes = [len(fitting) for fitting in options]
    total = n_combinations(sizes)
    if total > MAX_MAPPINGS_PER_NET:
        print(
            f"    [!] {topology.name}: {total} compatible assignments, "
            f"trying the {MAX_MAPPINGS_PER_NET} nearest the first choice"
        )
    for indices in graded_indices(sizes, MAX_MAPPINGS_PER_NET):
        combo = [option[index] for option, index in zip(options, indices, strict=True)]
        yield dict(zip(slot_types, combo, strict=True))


def uncapped_composition(result) -> tuple[str | None, str | None]:
    """(reduced formula without the monotopic units, their formula).

    A ``cap`` unit is an organic unit of external degree 1 - a pendant
    substituent or capping residue, found on the metal-free
    branch-point path. No blueprint carries a 1-connected slot, so a
    rebuild is short by exactly those atoms however well the net and
    every polytopic unit match; scoring that as "chemically not the
    same material" conflates a scope limit with the wrong-orbit
    assignment the composition gate exists to catch.

    (Metal-bound solvent is *not* a cap: the metal-oxo rule clusters
    C-free metal-bound atoms into the node, so a capped MOF round-trips
    with its full formula and never reaches this path.)

    Returns ``(None, None)`` when the structure has no cap units.
    """
    cap_atoms = [
        index
        for unit in result.units
        if unit.kind == "cap"
        for index in unit.atom_indices
    ]
    if not cap_atoms:
        return None, None
    structure = result.structure
    caps = Composition(Counter(structure[index].specie.symbol for index in cap_atoms))
    kept = Counter(
        site.specie.symbol
        for index, site in enumerate(structure)
        if index not in set(cap_atoms)
    )
    if not kept:
        return None, caps.reduced_formula
    return Composition(kept).reduced_formula, caps.reduced_formula


def _geometry(framework, result) -> dict:
    """Packing and closure of a rebuild, against the experimental cell.

    ``volume_ratio`` is per-atom and divided by the interpenetration
    fold, so it is invariant to supercell choice and compares a single
    net against a catenated crystal. It is the quantity the embedding
    relaxation of #174 moves; ``bond_residual`` is what that movement
    costs. Reported together because the trade between them is the
    whole point - either one alone is misleading.
    """
    built = framework.structure
    experimental = result.structure
    fold = result.n_periodic_components
    return {
        "volume_ratio": (built.volume / len(built))
        / (experimental.volume / len(experimental) * fold),
        "min_contact": framework.min_contact(),
        "bond_residual": bond_residuals(framework),
        "empty_slots": sorted(framework.graph.graph.get("empty_slots", ())),
    }


def roundtrip_one(
    mofgen: Autografs,
    source,
    max_rmsd: float,
    relax_embedding: bool = False,
) -> dict:
    """Deconstruct one structure and attempt the verified rebuild."""
    record: dict = {"outcome": None, "net": None, "tier": None, "error": None}
    t0 = time.perf_counter()
    try:
        result = mofgen.deconstruct(source)
    except (AutografsError, ValueError, KeyError, IndexError) as exc:
        record["outcome"] = "deconstruction_failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = time.perf_counter() - t0
        return record
    record["n_fragments"] = len(result.fragments)
    record["fold"] = result.n_periodic_components
    record["guests"] = result.guest_formulas
    if result.net_candidates:
        record["net"] = result.net_candidates
        record["tier"] = result.subframework_nets[0].tier
    if result.rod_units:
        # out of scope for *this* driver, not for the pipeline: rods
        # build through Autografs.build_rod (#158/#168/#173), which
        # takes a RodFragment and a run rather than the slot-type
        # mappings this round trip enumerates
        record["outcome"] = "rod" if result.net_candidates else "unidentified"
        record["seconds"] = time.perf_counter() - t0
        return record
    if not result.net_candidates:
        record["outcome"] = "unidentified"
        record["seconds"] = time.perf_counter() - t0
        return record
    experimental_formula = result.structure.composition.reduced_formula
    uncapped_formula, cap_formula = uncapped_composition(result)
    record["caps"] = cap_formula
    saw_mapping = False
    saw_verified = False
    saw_uncapped = False
    for net in result.net_candidates:
        topology = mofgen.topologies[net]
        for mappings in candidate_mappings(topology, result.fragments):
            saw_mapping = True
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
            saw_verified = True
            # the reduced formula is invariant to supercell choice and
            # interpenetration fold, so it compares across cells
            built_formula = framework.structure.composition.reduced_formula
            if built_formula == experimental_formula:
                record["outcome"] = "closed"
                record["rebuilt_net"] = net
                record.update(_geometry(framework, result))
                record["seconds"] = time.perf_counter() - t0
                return record
            if uncapped_formula is not None and built_formula == uncapped_formula:
                # everything but the monotopic units: no blueprint has a
                # 1-connected slot for those, so keep looking for a full
                # match and fall back to this if none turns up
                saw_uncapped = True
                record["rebuilt_net"] = net
            # else: right topology, wrong material - keep trying
    if saw_uncapped:
        record["outcome"] = "closed_uncapped"
    elif saw_verified:
        record["outcome"] = "closed_wrong_composition"
    elif saw_mapping:
        record["outcome"] = "rebuild_failed"
    else:
        record["outcome"] = "no_mapping"
    record["seconds"] = time.perf_counter() - t0
    return record


# one Autografs per worker process: the topology library costs seconds
# to load and is read-only during a run, so it is built once in the
# initializer rather than per structure
_WORKER: dict = {}


def _init_worker(topofile: str | None, max_rmsd: float, relax: bool) -> None:
    _WORKER["mofgen"] = Autografs(topofile=topofile)
    _WORKER["max_rmsd"] = max_rmsd
    _WORKER["relax"] = relax


def _run_worker(path: str) -> tuple[str, dict]:
    """Round-trip one structure in a worker; never raises.

    A corpus-scale sweep must not lose 4000 results to one pathological
    CIF, and an unexpected exception is itself data - it becomes a
    ``driver_error`` outcome rather than a traceback that kills the run.
    """
    name = Path(path).name
    try:
        record = roundtrip_one(
            _WORKER["mofgen"], path, _WORKER["max_rmsd"], _WORKER["relax"]
        )
    except Exception as exc:  # noqa: BLE001 - deliberate: see docstring
        record = {
            "outcome": "driver_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return name, record


def _load_checkpoint(path: Path) -> dict:
    """Read a JSONL checkpoint into ``{name: record}``.

    Truncated trailing lines (a killed run mid-write) are dropped
    rather than fatal.
    """
    records: dict = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            records[entry["name"]] = entry["record"]
    return records


def run(
    corpus: list[Path],
    mofgen: Autografs | None = None,
    max_rmsd: float = 0.5,
    *,
    relax_embedding: bool = False,
    topofile: str | None = None,
    n_jobs: int = 1,
    checkpoint: Path | None = None,
) -> dict:
    """Round-trip every structure; returns the results payload.

    Deterministic in sorted corpus order regardless of ``n_jobs`` - the
    results are keyed by structure name, so worker completion order
    does not reach the output.

    ``mofgen`` is used only by the serial path; workers build their own
    (an Autografs does not survive pickling to a spawned process, and
    the library load is why they are built in an initializer rather
    than per structure).
    """
    records = _load_checkpoint(checkpoint) if checkpoint is not None else {}
    if records:
        print(f"  resuming: {len(records)} structures already done")
    pending = [path for path in sorted(corpus) if path.name not in records]
    handle = checkpoint.open("a", encoding="utf-8") if checkpoint is not None else None
    try:
        if n_jobs > 1 and pending:
            executor = ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=_init_worker,
                initargs=(topofile, max_rmsd, relax_embedding),
            )
            with executor:
                stream = executor.map(
                    _run_worker, [str(path) for path in pending], chunksize=1
                )
                for index, (name, record) in enumerate(stream, 1):
                    records[name] = record
                    _report(index, len(pending), name, record, handle)
        elif pending:
            if mofgen is None:
                _init_worker(topofile, max_rmsd, relax_embedding)
            else:
                _WORKER.update(mofgen=mofgen, max_rmsd=max_rmsd, relax=relax_embedding)
            for index, path in enumerate(pending, 1):
                name, record = _run_worker(str(path))
                records[name] = record
                _report(index, len(pending), name, record, handle)
    finally:
        if handle is not None:
            handle.close()
    outcomes = Counter(record["outcome"] for record in records.values())
    return {
        "benchmark": "roundtrip",
        "max_rmsd": max_rmsd,
        "relax_embedding": relax_embedding,
        "n_structures": len(records),
        "outcomes": dict(sorted(outcomes.items())),
        "structures": dict(sorted(records.items())),
    }


def _report(index: int, total: int, name: str, record: dict, handle) -> None:
    print(f"  [{index}/{total}] {name:<40} {record['outcome']}", flush=True)
    if handle is not None:
        handle.write(json.dumps({"name": name, "record": record}) + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", help="directory, glob, or single CIF")
    parser.add_argument("-o", "--output", default="roundtrip.json")
    parser.add_argument("--max-rmsd", type=float, default=0.5)
    parser.add_argument(
        "--relax-embedding",
        action="store_true",
        help="rebuild with embedding relaxation (#174): symmetry-allowed "
        "slot displacements + anchor-direction objective",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="round-trip only the first N"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="worker processes (default 1; -1 uses all cores)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="JSONL file of finished structures; resumes from it if present",
    )
    parser.add_argument("--topofile", default=None, help="topology library override")
    args = parser.parse_args()

    corpus = _collect(args.corpus)
    if args.limit is not None:
        corpus = corpus[: args.limit]
    if not corpus:
        raise SystemExit(f"no structures matched {args.corpus!r}")
    n_jobs = os.cpu_count() or 1 if args.n_jobs == -1 else args.n_jobs
    payload = run(
        corpus,
        max_rmsd=args.max_rmsd,
        relax_embedding=args.relax_embedding,
        topofile=args.topofile,
        n_jobs=n_jobs,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
    )
    Path(args.output).write_text(json.dumps(payload, indent=1, default=str))
    print(f"\n{payload['n_structures']} structures -> {args.output}")
    for outcome, count in payload["outcomes"].items():
        print(f"  {outcome:<24} {count}")


if __name__ == "__main__":
    main()
