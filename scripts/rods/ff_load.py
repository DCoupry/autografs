"""Does our embedding relaxation reduce the force field's workload?

The geometric pipeline and the force field both move atoms, and the
question is whether the first spares the second any effort. The two swap
arms make it a paired experiment: identical recombinations, built once
with the slot centres fixed and once with the symmetry-allowed
displacements freed, then each handed the *same* full UFF minimization.

FF load is measured three ways, because they can disagree:

* **wall time** - what a screening pipeline actually pays. Fair here
  only because the comparison is paired: the same recombination has the
  same atom count in both arms.
* **max atom displacement** and **RMSD** - how far the field had to move
  the structure. This is the physical measure of "how wrong was the
  starting geometry", and unlike time it is hardware-independent.
* **final energy** - whether the extra freedom also lands in a better
  minimum, not merely a cheaper path to the same one.

A build that closes its bonds well can still start far from a force
field's minimum, so none of these is implied by the closure figures
reported elsewhere.

Usage:
    python scripts/rods/ff_load.py swaps-fixed swaps-relaxed -o ff-load.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from autografs.framework import Framework


def measure(path: Path) -> dict:
    """Full UFF minimization of one framework, with its cost."""
    framework = Framework.load(str(path))
    before = np.asarray(framework.cart_coords, dtype=float)
    record: dict = {
        "name": path.name,
        "n_atoms": len(before),
        "contact_before": round(float(framework.min_contact()), 3),
        "volume_before": round(float(framework.structure.volume), 2),
    }
    start = time.perf_counter()
    try:
        relaxed = framework.relax()
    except Exception as exc:  # noqa: BLE001 - a failure is data
        record["outcome"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"[:150]
        return record
    record["seconds"] = round(time.perf_counter() - start, 1)
    after = np.asarray(relaxed.cart_coords, dtype=float)
    # same connectivity and atom order by construction, so the
    # displacement is a direct atom-by-atom comparison
    delta = np.linalg.norm(after - before, axis=1)
    record.update(
        {
            "outcome": "relaxed",
            "contact_after": round(float(relaxed.min_contact()), 3),
            "volume_after": round(float(relaxed.structure.volume), 2),
            "max_displacement": round(float(delta.max()), 3),
            "rmsd_displacement": round(float(np.sqrt((delta**2).mean())), 3),
            "energy": round(float(relaxed.energy), 2)
            if relaxed.energy is not None
            else None,
        }
    )
    return record


def _isolated(path: Path, timeout: float) -> dict:
    """Measure in a child process; LAMMPS can abort at C level."""
    proc = subprocess.run(  # noqa: S603 - fixed argv, our own module
        [sys.executable, str(Path(__file__).resolve()), "--one", str(path)],
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
        "error": (proc.stderr or "")[-150:],
    }


def _frameworks(directory: Path) -> dict[str, Path]:
    return {
        p.name: p
        for p in sorted(directory.glob("*.json"))
        if p.name not in ("swaps.json", "rods.json")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fixed", nargs="?", help="swap dir built with fixed slots")
    parser.add_argument("relaxed", nargs="?", help="swap dir built with relaxation")
    parser.add_argument("-o", "--output", default="ff-load.json")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--one", default=None, help="internal: measure one framework")
    args = parser.parse_args()

    if args.one:
        print(json.dumps(measure(Path(args.one)), default=str))
        return
    if not (args.fixed and args.relaxed):
        raise SystemExit("both swap directories are required")

    arms = {
        "fixed": _frameworks(Path(args.fixed)),
        "relaxed": _frameworks(Path(args.relaxed)),
    }
    # the same recombination is written under the same stem in both
    # arms; anything unpaired cannot answer the question and is skipped
    paired = sorted(set(arms["fixed"]) & set(arms["relaxed"]))
    print(f"{len(paired)} recombinations built in both arms\n")

    results: dict[str, dict] = {"fixed": {}, "relaxed": {}}
    for index, name in enumerate(paired, 1):
        for arm in ("fixed", "relaxed"):
            record = _isolated(arms[arm][name], args.timeout)
            results[arm][name] = record
        f, r = results["fixed"][name], results["relaxed"][name]
        print(
            f"[{index}/{len(paired)}] {name[:34]:36s} "
            f"fixed {f.get('seconds', '-')}s/{f.get('max_displacement', '-')}A  "
            f"relaxed {r.get('seconds', '-')}s/{r.get('max_displacement', '-')}A",
            flush=True,
        )

    both = [
        n
        for n in paired
        if results["fixed"][n].get("outcome") == "relaxed"
        and results["relaxed"][n].get("outcome") == "relaxed"
    ]

    def _median(arm: str, key: str) -> float | None:
        values = [
            results[arm][n][key] for n in both if results[arm][n].get(key) is not None
        ]
        return round(statistics.median(values), 3) if values else None

    payload = {
        "study": "ff-load-vs-embedding-relaxation",
        "paired": len(paired),
        "relaxed_in_both": len(both),
        "medians": {
            arm: {
                key: _median(arm, key)
                for key in (
                    "seconds",
                    "max_displacement",
                    "rmsd_displacement",
                    "energy",
                    "contact_before",
                    "contact_after",
                )
            }
            for arm in ("fixed", "relaxed")
        },
        "wins": {
            key: {
                "relaxed_lower": sum(
                    1
                    for n in both
                    if results["relaxed"][n][key] < results["fixed"][n][key]
                ),
                "fixed_lower": sum(
                    1
                    for n in both
                    if results["relaxed"][n][key] > results["fixed"][n][key]
                ),
            }
            for key in ("seconds", "max_displacement", "rmsd_displacement")
        },
        "records": results,
    }
    Path(args.output).write_text(json.dumps(payload, indent=1, default=str))
    print(f"\nrelaxed in both arms: {len(both)}/{len(paired)}")
    print(f"medians: {json.dumps(payload['medians'], indent=1)}")
    print(f"wins: {json.dumps(payload['wins'])}")


if __name__ == "__main__":
    main()
