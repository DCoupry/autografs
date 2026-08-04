"""How far is a recombined rod framework from a usable structure?

The swap driver produces frameworks whose bonds close but whose atoms
often do not clear each other: a host blueprint's proportions belong to
its own units, and neither the single transverse scale nor the
symmetry-allowed slot displacements can always re-proportion it for a
different one. The remaining question is practical rather than
conceptual - is that residual a force field's job, and how much of one?

This driver answers it as a *ladder* instead of a single relaxation.
Each framework is relaxed in increasing step budgets, carrying the
result forward, and the closest contact is measured after each rung. A
structure that clears the contact floor after 10 steps was essentially
already right; one that needs 500 was not, and one that never does is
telling us the recombination itself is wrong rather than merely tight.

Reporting the ladder matters: a single "relaxed" number would conflate
those three populations, and the cost of a rung is what decides whether
a generative pipeline can afford to screen with it.

Usage:
    python scripts/rods/relax_swaps.py swaps_dir -o relax-study.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from autografs.framework import Framework

# the same floor the swap driver uses to call a build a structure
CONTACT_FLOOR = 1.5
# cumulative: each rung continues from the previous one's result, so the
# x axis is total steps spent, not steps per attempt
LADDER = (10, 25, 50, 100, 250, 500)


def study_one(path: Path, ladder: tuple[int, ...], fmax: float) -> dict:
    """Relax one framework up the ladder, measuring contact at each rung."""
    record: dict = {"name": path.name, "rungs": []}
    try:
        framework = Framework.load(str(path))
    except Exception as exc:  # noqa: BLE001 - a bad artifact is data
        record["outcome"] = "load_failed"
        record["error"] = f"{type(exc).__name__}: {exc}"[:160]
        return record
    record["n_atoms"] = len(framework.structure)
    record["formula"] = framework.structure.composition.reduced_formula
    try:
        record["contact_0"] = round(framework.min_contact(), 3)
    except Exception:  # noqa: BLE001
        record["contact_0"] = None
    # UFF4MOF extends UFF with metal types but lammps_interface's table
    # is missing some main-group ones (S_3, for instance), so a sulfonate
    # linker raises KeyError before a single step runs. Plain UFF covers
    # those, and which one was used is recorded rather than hidden.
    spent = 0
    for budget in ladder:
        step = budget - spent
        start = time.perf_counter()
        for field in ("UFF4MOF", "UFF"):
            try:
                framework = framework.relax(force_field=field, steps=step, fmax=fmax)
            except KeyError as exc:
                record.setdefault("missing_types", []).append(str(exc))
                continue
            except Exception as exc:  # noqa: BLE001 - relaxation failure is data
                record["outcome"] = "relax_failed"
                record["error"] = f"{type(exc).__name__}: {exc}"[:160]
                record["traceback"] = traceback.format_exc(limit=3)
                return record
            record["force_field"] = field
            break
        else:
            record["outcome"] = "no_force_field"
            record["error"] = f"no parameters: {record.get('missing_types')}"
            return record
        spent = budget
        contact = framework.min_contact()
        record["rungs"].append(
            {
                "steps": budget,
                "contact": round(float(contact), 3),
                "energy": round(float(framework.energy), 3)
                if framework.energy is not None
                else None,
                "seconds": round(time.perf_counter() - start, 2),
                "clash_free": bool(contact >= CONTACT_FLOOR),
            }
        )
        if contact >= CONTACT_FLOOR:
            record["outcome"] = "clash_free"
            record["steps_to_clear"] = budget
            record["final_contact"] = round(float(contact), 3)
            record["best_contact"] = round(float(contact), 3)
            return record
    # contact is NOT monotonic in the step budget: from a severe overlap
    # the field's repulsion plus a relaxing cell can drive the structure
    # through worse geometry before it recovers (measured: 1.917 -> 0.555
    # in ten steps on a clean synthetic rod). Reporting only the endpoint
    # would call that a failure and hide a trajectory that was fine in
    # between, so the best rung is carried alongside the last.
    contacts = [rung["contact"] for rung in record["rungs"]]
    record["outcome"] = "still_clashing"
    record["final_contact"] = contacts[-1] if contacts else None
    record["best_contact"] = max(contacts) if contacts else None
    record["best_steps"] = (
        record["rungs"][contacts.index(max(contacts))]["steps"] if contacts else None
    )
    return record


def _isolated(path: Path, fmax: float, timeout: float) -> dict:
    """Study one framework in a child process.

    LAMMPS reports some malformed inputs by aborting at C level, which
    takes the interpreter with it - one bad recombination would
    otherwise end the sweep and silently truncate the population. The
    child's exit status becomes a recorded outcome instead.
    """
    proc = subprocess.run(  # noqa: S603 - fixed argv, our own module
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--one",
            str(path),
            "--fmax",
            str(fmax),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                break
    return {
        "name": path.name,
        "outcome": "aborted",
        "rungs": [],
        "error": (proc.stderr or proc.stdout or "")[-160:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("swaps_dir", nargs="?", help="output dir of swap_rod_units.py")
    parser.add_argument("-o", "--output", default="relax-study.json")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--one", default=None, help="internal: study one framework")
    args = parser.parse_args()

    if args.one:  # child process: emit one record as JSON and exit
        print(json.dumps(study_one(Path(args.one), LADDER, args.fmax), default=str))
        return
    if not args.swaps_dir:
        raise SystemExit("swaps_dir is required")

    directory = Path(args.swaps_dir)
    frameworks = sorted(p for p in directory.glob("*.json") if p.name != "swaps.json")
    if args.limit:
        frameworks = frameworks[: args.limit]
    if not frameworks:
        raise SystemExit(f"no saved frameworks in {directory}")
    print(f"relaxing {len(frameworks)} recombined frameworks up {LADDER}\n")

    records = []
    for index, path in enumerate(frameworks, 1):
        try:
            record = _isolated(path, args.fmax, args.timeout)
        except subprocess.TimeoutExpired:
            record = {
                "name": path.name,
                "outcome": "timeout",
                "rungs": [],
                "error": f"exceeded {args.timeout}s",
            }
        records.append(record)
        if record["outcome"] == "clash_free":
            tag = (
                f"clear at {record['steps_to_clear']} steps "
                f"({record['contact_0']} -> {record['final_contact']} A)"
            )
        elif record["outcome"] == "still_clashing":
            tag = (
                f"still clashing ({record['contact_0']} -> {record['final_contact']} A)"
            )
        else:
            tag = record["outcome"]
        print(f"[{index}/{len(frameworks)}] {record['name'][:44]}: {tag}", flush=True)

    cleared = [r for r in records if r["outcome"] == "clash_free"]
    started_clean = [r for r in cleared if (r.get("contact_0") or 0) >= CONTACT_FLOOR]
    payload = {
        "study": "ff-relaxation-ladder",
        "contact_floor": CONTACT_FLOOR,
        "ladder": list(LADDER),
        "n": len(records),
        "clash_free": len(cleared),
        "already_clean": len(started_clean),
        "rescued": len(cleared) - len(started_clean),
        "still_clashing": sum(1 for r in records if r["outcome"] == "still_clashing"),
        "failed": sum(
            1 for r in records if r["outcome"] not in ("clash_free", "still_clashing")
        ),
        "by_rung": {
            str(b): sum(1 for r in cleared if r.get("steps_to_clear") == b)
            for b in LADDER
        },
        # structures whose best rung cleared the floor even though the
        # last one did not - a relaxation that overshot, not one that
        # failed
        "cleared_transiently": sum(
            1
            for r in records
            if r["outcome"] == "still_clashing"
            and (r.get("best_contact") or 0) >= CONTACT_FLOOR
        ),
        "records": records,
    }
    times = [
        rung["seconds"] for r in records for rung in r["rungs"] if rung["steps"] == 10
    ]
    if times:
        payload["seconds_first_rung_median"] = round(statistics.median(times), 2)
    Path(args.output).write_text(json.dumps(payload, indent=1, default=str))
    print(
        f"\n{payload['clash_free']}/{payload['n']} clash-free "
        f"({payload['rescued']} rescued by relaxation), "
        f"{payload['still_clashing']} still clashing -> {args.output}"
    )
    print(f"cleared at each rung: {payload['by_rung']}")


if __name__ == "__main__":
    main()
