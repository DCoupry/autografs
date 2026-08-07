"""Do the recombined frameworks survive a force field?

``recombine.py`` produces hypothetical frameworks whose bonds close and
whose atoms clear each other, but every one of those is a *geometric*
claim made by a builder that knows about covalent bond lengths and
nothing else - no angles, no torsions, no van der Waals, no
electrostatics. This driver asks the next question: put each one in
UFF4MOF and minimize, and see what is still a framework afterwards.

Two independent survival tests, because they fail differently:

**Geometric** - the structure relaxes without the cell collapsing or
atoms piling up. Reported as closest contact and cell volume before and
after, plus the RMS displacement the field had to apply. A structure
that barely moves was already right; one that moves a long way was not,
even if it ends somewhere plausible.

**Topological** - the relaxed *geometry* is re-deconstructed from
scratch and re-identified. This is the strict test and the reason the
driver does not simply call ``verify_net``: relaxation preserves the
bond graph by construction, so any graph-based check passes tautologically.
Perceiving the bonds afresh from the relaxed coordinates is what can
actually fail, and a framework that no longer reads as its own net has
not survived however good its contacts look.

LAMMPS exposes no step or force control (it runs lammps-interface's own
minimization protocol), so there is no ladder here - one full
minimization, one endpoint. Do not add a ``steps`` ladder: those
arguments reach the ASE bridge only, and passing them to the LAMMPS
path measured nothing when it was tried.

Usage:
    python scripts/generative/relax_recombined.py recombine.json ... \\
        --frameworks usable-filled --frameworks usable-empty -o relaxed.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from autografs.framework import Framework  # noqa: E402

# the contact floor recombine.py used to call a build usable; a relaxed
# structure that falls back under it has been made worse, not better
CONTACT_FLOOR = 1.5
# a cell that changes by more than this factor either way has not been
# refined, it has been re-imagined
VOLUME_BAND = (0.5, 2.0)
# any two atoms closer than this are coincident, not bonded: a relaxation
# that returns them has failed, whatever its other numbers say
COINCIDENT = 0.3


def _displacement(before: Framework, after: Framework) -> float:
    """RMS atom displacement in Angstrom, min-imaged in the relaxed cell.

    Measured in fractional coordinates and pushed through the relaxed
    cell: the cell itself moves during the relaxation, so a cartesian
    difference would report the box change as atom motion.
    """
    cell_before = np.asarray(before.graph.graph["cell"], dtype=float)
    cell_after = np.asarray(after.graph.graph["cell"], dtype=float)
    nodes = sorted(before.graph.nodes())
    coords_b = np.array([before.graph.nodes[n]["coord"] for n in nodes], dtype=float)
    coords_a = np.array([after.graph.nodes[n]["coord"] for n in nodes], dtype=float)
    frac_b = coords_b @ np.linalg.inv(cell_before)
    frac_a = coords_a @ np.linalg.inv(cell_after)
    delta = frac_a - frac_b
    delta -= np.round(delta)
    cartesian = delta @ cell_after
    return float(np.sqrt((cartesian**2).sum(axis=1).mean()))


def safe_cutoff(framework: Framework, default: float = 12.5) -> float:
    """Largest cutoff that does not trigger lammps-interface's supercell.

    lammps-interface replicates any cell smaller than twice the
    non-bonded cutoff, and the topology it generates for that supercell
    is **corrupt on these structures**: at the default 12.5 A a bnn
    build came back with 33 atom pairs inside 0.3 A, bonded pairs among
    them at 1e-8 A, while the identical framework at cutoff 8 or 6 -
    where no replication happens - relaxed cleanly to 1.047 A. The fold
    back is not the culprit: the supercell's replicas were measured to
    agree to 0.0009 A, so translational symmetry held and relax.py's
    assumption is sound. Other structures fail outright inside the same
    code path ("Invalid atom ID in Dihedrals section").

    Shortening the cutoff is a real approximation and is recorded per
    structure, not hidden - but a truncated dispersion tail is a far
    smaller error than a collapsed framework.
    """
    lengths = framework.structure.lattice.abc
    # a hair under half, so equality does not trip the resize
    return float(min(default, 0.5 * min(lengths) - 0.01))


def relax_one(
    path: Path,
    net: str,
    force_field: str,
    topofile: str | None,
    auto_cutoff: bool = True,
    cutoff: float = 12.5,
    calculator: str | None = None,
    steps: int = 150,
    fmax: float = 0.1,
) -> dict:
    """Relax one framework and measure what survived."""
    record: dict = {"framework": path.name, "net": net, "outcome": None}
    try:
        before = Framework.load(str(path))
    except Exception as exc:  # noqa: BLE001 - a bad artifact is data
        record["outcome"] = "load_failed"
        record["error"] = f"{type(exc).__name__}: {exc}"[:160]
        return record
    structure = before.structure
    record["formula"] = structure.composition.reduced_formula
    record["n_atoms"] = len(structure)
    record["n_bonds"] = before.graph.number_of_edges()
    record["contact_before"] = round(float(before.min_contact()), 3)
    record["volume_before"] = round(float(structure.volume), 2)

    used = safe_cutoff(before, cutoff) if auto_cutoff else cutoff
    record["cutoff"] = round(used, 2)
    record["cutoff_shortened"] = bool(used < cutoff - 1e-9)
    record["backend"] = calculator or force_field
    start = time.perf_counter()
    try:
        if calculator:
            # the ASE bridge honours steps/fmax (the LAMMPS path does
            # not), and relaxes the cell when the method has a stress
            after = before.relax(
                calculator=calculator, steps=steps, fmax=fmax, relax_cell=True
            )
        else:
            after = before.relax(force_field=force_field, cutoff=used)
    except Exception as exc:  # noqa: BLE001 - a relaxation failure is data
        record["outcome"] = "relax_failed"
        record["error"] = f"{type(exc).__name__}: {exc}"[:200]
        record["traceback"] = traceback.format_exc(limit=3)
        record["seconds"] = round(time.perf_counter() - start, 1)
        return record
    record["seconds"] = round(time.perf_counter() - start, 1)
    record["force_field"] = force_field
    relaxed = after.structure
    record["contact_after"] = round(float(after.min_contact()), 3)
    record["volume_after"] = round(float(relaxed.volume), 2)
    record["volume_ratio"] = round(relaxed.volume / structure.volume, 4)
    record["energy"] = (
        round(float(after.energy), 2) if after.energy is not None else None
    )
    record["n_bonds_after"] = after.graph.number_of_edges()
    try:
        record["rms_displacement"] = round(_displacement(before, after), 3)
    except Exception:  # noqa: BLE001 - a diagnostic must not sink the record
        record["rms_displacement"] = None

    # ---- corruption check -----------------------------------------
    # atoms on top of each other are not a relaxation result, they are a
    # broken one, and they must never be reported as a survivor on the
    # strength of some other metric looking fine
    distances = relaxed.distance_matrix.copy()
    np.fill_diagonal(distances, np.inf)
    record["closest_any_pair"] = round(float(distances.min()), 4)
    record["coincident_pairs"] = int((distances < COINCIDENT).sum() // 2)
    if record["coincident_pairs"]:
        record["outcome"] = "collapsed"
        record["geometric_survivor"] = False
        record["topological_survivor"] = False
        record["survivor"] = False
        return record

    # ---- geometric survival ---------------------------------------
    contact_ok = record["contact_after"] >= CONTACT_FLOOR
    volume_ok = VOLUME_BAND[0] <= record["volume_ratio"] <= VOLUME_BAND[1]
    record["contact_held"] = bool(contact_ok)
    record["volume_held"] = bool(volume_ok)
    record["geometric_survivor"] = bool(contact_ok and volume_ok)

    # ---- topological survival -------------------------------------
    # re-perceive the bonding from the RELAXED coordinates: the bond
    # graph rides through relaxation untouched, so only a fresh
    # deconstruction can tell us the structure still reads as its net
    cif = path.with_name(path.stem + "-relaxed.cif")
    try:
        after.write_cif(str(cif))
        record["relaxed_cif"] = cif.name
        from autografs import Autografs

        mofgen = Autografs(topofile=topofile) if topofile else Autografs()
        # a 2D layer net's c axis is slab padding, not a real lattice
        # parameter, so a 3D cell relaxation is free to squeeze the
        # vacuum and stack the layers onto each other. Recorded, not
        # excluded: the result is real, it just is not a statement about
        # the material.
        record["is_2d"] = bool(mofgen.topologies[net].is_2d) if net else None
        result = mofgen.deconstruct(str(cif))
        candidates = list(result.net_candidates)
        record["renet"] = candidates
        record["n_units_after"] = len(result.units)
        record["topological_survivor"] = net in candidates
    except Exception as exc:  # noqa: BLE001 - re-identification failure is data
        record["renet"] = None
        record["renet_error"] = f"{type(exc).__name__}: {exc}"[:160]
        record["topological_survivor"] = False

    record["survivor"] = bool(
        record["geometric_survivor"] and record["topological_survivor"]
    )
    record["outcome"] = "relaxed"
    return record


def _isolated(
    path: Path,
    net: str,
    force_field: str,
    topofile: str | None,
    timeout: float,
    cutoff: float = 12.5,
    fixed_cutoff: bool = False,
    calculator: str | None = None,
    steps: int = 150,
    fmax: float = 0.1,
) -> dict:
    """Relax one framework in a child process.

    LAMMPS reports some malformed inputs by aborting at C level, which
    takes the interpreter with it - one bad structure would otherwise
    end the sweep and silently truncate the population. The child's exit
    status becomes a recorded outcome instead.
    """
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--one",
        str(path),
        "--net",
        net,
        "--force-field",
        force_field,
        "--cutoff",
        str(cutoff),
    ]
    if calculator:
        argv += ["--calculator", calculator, "--steps", str(steps), "--fmax", str(fmax)]
    if fixed_cutoff:
        argv.append("--fixed-cutoff")
    if topofile:
        argv += ["--topofile", topofile]
    proc = subprocess.run(  # noqa: S603 - fixed argv, our own module
        argv, capture_output=True, text=True, timeout=timeout, check=False
    )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                break
    return {
        "framework": path.name,
        "net": net,
        "outcome": "aborted",
        "error": (proc.stderr or proc.stdout or "")[-200:],
    }


def collect_targets(
    reports: list[Path], directories: list[Path], genuine_only: bool = True
) -> list[dict]:
    """The written frameworks worth relaxing.

    A ``recombine.py`` report is filtered down to its usable, novel,
    genuine builds. A ``materialize.py`` manifest is taken as-is: that
    file exists precisely to define a population, and re-applying the
    ``usable`` test to it would silently drop the clashing structures
    the relaxation is meant to fix - which is most of them.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from recombine import is_genuine_framework

    targets: dict[str, dict] = {}
    for report in reports:
        payload = json.loads(report.read_text(encoding="utf-8"))
        preselected = payload.get("benchmark") == "materialize"
        for record in payload["records"]:
            if not record.get("written"):
                continue
            if not preselected:
                if not record.get("usable") or not record.get("novel", False):
                    continue
                if genuine_only and not is_genuine_framework(record):
                    continue
            for directory in directories:
                candidate = directory / f"{record['written']}.json"
                if candidate.exists():
                    # keyed by fingerprint: the same assembly written by
                    # two arms is one hypothetical material, not two
                    targets.setdefault(
                        record.get("fingerprint", record["written"]),
                        {
                            "path": candidate,
                            "net": record["net"],
                            "n_atoms": record.get("n_atoms"),
                            "formula": record.get("formula"),
                        },
                    )
                    break
    return sorted(targets.values(), key=lambda t: t["n_atoms"] or 0)


def summarize(records: list[dict]) -> dict:
    relaxed = [r for r in records if r["outcome"] == "relaxed"]

    def _stats(values: list[float]) -> dict | None:
        values = [v for v in values if v is not None and math.isfinite(v)]
        if not values:
            return None
        values = sorted(values)
        return {
            "n": len(values),
            "median": round(statistics.median(values), 3),
            "min": round(values[0], 3),
            "max": round(values[-1], 3),
        }

    return {
        "attempted": len(records),
        "relaxed": len(relaxed),
        "outcomes": {
            outcome: sum(1 for r in records if r["outcome"] == outcome)
            for outcome in sorted({r["outcome"] for r in records})
        },
        "geometric_survivors": sum(1 for r in relaxed if r.get("geometric_survivor")),
        "topological_survivors": sum(
            1 for r in relaxed if r.get("topological_survivor")
        ),
        "survivors": sum(1 for r in relaxed if r.get("survivor")),
        # 2D layer nets are reported apart: their c axis is slab padding,
        # so a 3D cell relaxation is not a statement about the material
        "survivors_3d": sum(
            1 for r in relaxed if r.get("survivor") and not r.get("is_2d")
        ),
        "n_2d": sum(1 for r in relaxed if r.get("is_2d")),
        "cutoff_shortened": sum(1 for r in records if r.get("cutoff_shortened")),
        "contact_before": _stats([r.get("contact_before") for r in relaxed]),
        "contact_after": _stats([r.get("contact_after") for r in relaxed]),
        "volume_ratio": _stats([r.get("volume_ratio") for r in relaxed]),
        "rms_displacement": _stats([r.get("rms_displacement") for r in relaxed]),
        "seconds": _stats([r.get("seconds") for r in relaxed]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="*", help="recombine.py output JSON files")
    parser.add_argument(
        "--frameworks",
        action="append",
        default=[],
        help="directory written by recombine.py --write-usable (repeatable)",
    )
    parser.add_argument("-o", "--output", default="relaxed.json")
    parser.add_argument("--force-field", default="UFF4MOF")
    parser.add_argument("--cutoff", type=float, default=12.5)
    parser.add_argument(
        "--fixed-cutoff",
        action="store_true",
        help="do not shorten the cutoff to avoid lammps-interface's supercell",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--calculator",
        default=None,
        help="ASE backend instead of LAMMPS: 'gfn1' (tblite) or 'gfn-ff' (xtb)",
    )
    parser.add_argument("--steps", type=int, default=150, help="ASE path only")
    parser.add_argument("--fmax", type=float, default=0.1, help="ASE path only")
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="skip candidates larger than this (the ASE backends scale steeply)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="JSONL of finished records; a rerun skips what it already has",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--topofile", default=None)
    parser.add_argument(
        "--all-usable", action="store_true", help="skip the linker test"
    )
    parser.add_argument("--one", default=None, help="internal: relax one framework")
    parser.add_argument("--net", default=None, help="internal: its intended net")
    args = parser.parse_args()

    if args.one:  # child process: emit one record as JSON and exit
        print(
            json.dumps(
                relax_one(
                    Path(args.one),
                    args.net or "",
                    args.force_field,
                    args.topofile,
                    auto_cutoff=not args.fixed_cutoff,
                    cutoff=args.cutoff,
                    calculator=args.calculator,
                    steps=args.steps,
                    fmax=args.fmax,
                ),
                default=str,
            )
        )
        return

    targets = collect_targets(
        [Path(p) for p in args.reports],
        [Path(d) for d in args.frameworks],
        genuine_only=not args.all_usable,
    )
    if args.max_atoms:
        kept = [t for t in targets if (t["n_atoms"] or 0) <= args.max_atoms]
        print(
            f"size filter: {len(kept)} of {len(targets)} at or below "
            f"{args.max_atoms} atoms"
        )
        targets = kept
    if args.limit:
        targets = targets[: args.limit]
    if not targets:
        raise SystemExit("no usable novel frameworks found in those reports")

    # a relaxation sweep is long and every structure is independent, so
    # finished work is checkpointed and never repeated
    done: dict[str, dict] = {}
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                done[entry["framework"]] = entry
        print(f"resuming: {len(done)} already relaxed")
    todo = [t for t in targets if t["path"].name not in done]
    handle = checkpoint.open("a", encoding="utf-8") if checkpoint else None
    print(
        f"relaxing {len(todo)} of {len(targets)} frameworks with "
        f"{args.force_field} on {args.n_jobs} worker(s)\n",
        flush=True,
    )

    def _one(target: dict) -> dict:
        try:
            record = _isolated(
                target["path"],
                target["net"],
                args.force_field,
                args.topofile,
                args.timeout,
                cutoff=args.cutoff,
                fixed_cutoff=args.fixed_cutoff,
                calculator=args.calculator,
                steps=args.steps,
                fmax=args.fmax,
            )
        except subprocess.TimeoutExpired:
            record = {
                "framework": target["path"].name,
                "net": target["net"],
                "outcome": "timeout",
                "error": f"exceeded {args.timeout}s",
            }
        record.setdefault("n_atoms", target["n_atoms"])
        return record

    records = list(done.values())
    started = time.perf_counter()
    if args.n_jobs > 1:
        # threads, not processes: every relaxation already runs in its
        # own child process for LAMMPS crash isolation, so the parent
        # only waits on them. A process pool would also die whole when
        # one child aborts at C level, which is the thing being guarded.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.n_jobs) as pool:
            # as_completed order, not map order: one slow structure must
            # not stall the checkpoint behind it
            futures = {pool.submit(_one, t): t for t in todo}
            for index, future in enumerate(as_completed(futures), 1):
                record = future.result()
                records.append(record)
                _report(index, len(todo), record)
                _checkpoint(handle, record)
    else:
        for index, target in enumerate(todo, 1):
            record = _one(target)
            records.append(record)
            _report(index, len(todo), record)
            _checkpoint(handle, record)
    if handle:
        handle.close()

    payload = {
        "benchmark": "relax_recombined",
        "force_field": args.force_field,
        "calculator": args.calculator,
        "contact_floor": CONTACT_FLOOR,
        "volume_band": list(VOLUME_BAND),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "summary": summarize(records),
        "records": records,
    }
    Path(args.output).write_text(json.dumps(payload, indent=1, default=str))
    print(f"\n-> {args.output}")
    for key, value in payload["summary"].items():
        print(f"  {key:<22} {value}")


def _checkpoint(handle, record: dict) -> None:
    if handle is None:
        return
    handle.write(json.dumps(record, default=str) + "\n")
    handle.flush()


def _report(index: int, total: int, record: dict) -> None:
    if record["outcome"] == "relaxed":
        marks = (
            f"contact {record['contact_before']}->{record['contact_after']} "
            f"V x{record['volume_ratio']} "
            f"{'GEO' if record.get('geometric_survivor') else 'geo-fail'} "
            f"{'TOPO' if record.get('topological_survivor') else 'topo-fail'}"
        )
    else:
        marks = f"{record['outcome']}: {str(record.get('error'))[:70]}"
    print(
        f"[{index}/{total}] {record['net']:<8} {record.get('n_atoms', '?'):>5} atoms  {marks}",
        flush=True,
    )


if __name__ == "__main__":
    main()
