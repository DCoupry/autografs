"""Does the force field erase what the embedding relaxation does?

The library-net version of this test was invalid: auto-picked SBUs give
clashing, disconnected builds that no force field will touch, so it
measured the SBU choice rather than the relaxation. Self-templates are
the right population - real deconstructed structures rebuilt on their
own blueprints, which are faithful by construction (the 1901-structure
E9 result) and whose P1 blueprints give the relaxation plenty of free
displacements to use.

Each structure is built **twice** from the same blueprint and mapping -
slots fixed, slots freed - and both are handed the same full UFF
minimization. The endpoint is what matters:

* **rmsd_endpoints**: atom-by-atom between the two FF-relaxed
  structures (identical graph and atom order, so exact). Near zero
  means the force field erased the difference and the geometric step
  buys nothing wherever one follows.
* **energy_gap**: same minimum, or genuinely different ones?
* **rmsd_built** as the reference: if the builds differ but the
  endpoints do not, that is redundancy; if neither differs, the
  relaxation simply did nothing here.

Usage:
    python scripts/benchmarks/embedding_vs_ff_selftemplate.py MANIFEST \
        -o embedding-vs-ff-self.json --limit 20
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _corpus import collect as _collect  # noqa: E402

from autografs import Autografs  # noqa: E402
from autografs.builder import build_framework  # noqa: E402
from autografs.exceptions import AutografsError  # noqa: E402
from autografs.extract_topology import topology_from_deconstruction  # noqa: E402


def compare(path: str) -> dict:
    """Build one structure fixed and freed, FF both, compare endpoints."""
    record: dict = {"name": Path(path).name, "outcome": "failed"}
    mofgen = Autografs()
    try:
        result = mofgen.deconstruct(path)
        if result.rod_units:
            record["outcome"] = "skipped_rod"
            return record
        topology, mapping = topology_from_deconstruction(result)
    except (AutografsError, ValueError, KeyError, IndexError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"[:120]
        return record
    built = {}
    for arm, relax in (("fixed", False), ("freed", True)):
        try:
            built[arm] = build_framework(
                topology,
                {i: copy.deepcopy(result.fragments[n]) for i, n in mapping.items()},
                max_rmsd=0.5,
                relax_embedding=relax,
            )
        except (AutografsError, ValueError, KeyError) as exc:
            record["outcome"] = f"build_failed_{arm}"
            record["error"] = f"{type(exc).__name__}: {exc}"[:120]
            return record
    record["n_atoms"] = len(built["fixed"].structure)
    for arm in ("fixed", "freed"):
        record[f"contact_{arm}"] = round(float(built[arm].min_contact()), 3)
    before = {a: np.asarray(f.cart_coords, float) for a, f in built.items()}
    record["rmsd_built"] = round(
        float(np.sqrt(((before["freed"] - before["fixed"]) ** 2).sum(1).mean())), 4
    )
    relaxed = {}
    for arm in ("fixed", "freed"):
        start = time.perf_counter()
        try:
            relaxed[arm] = built[arm].relax()
        except Exception as exc:  # noqa: BLE001 - an FF failure is data
            record["outcome"] = f"relax_failed_{arm}"
            record["error"] = f"{type(exc).__name__}: {exc}"[:120]
            return record
        record[f"seconds_{arm}"] = round(time.perf_counter() - start, 1)
        energy = relaxed[arm].energy
        record[f"energy_{arm}"] = (
            round(float(energy), 3) if energy is not None else None
        )
    after = {a: np.asarray(f.cart_coords, float) for a, f in relaxed.items()}
    record["rmsd_endpoints"] = round(
        float(np.sqrt(((after["freed"] - after["fixed"]) ** 2).sum(1).mean())), 4
    )
    if record["energy_fixed"] is not None and record["energy_freed"] is not None:
        record["energy_gap"] = round(record["energy_freed"] - record["energy_fixed"], 3)
    # the decisive comparison: distance to the crystal each build came
    # from, before and after the force field
    for arm in ("fixed", "freed"):
        record[f"exp_{arm}_built"] = _rmsd_to_experiment(built[arm], result.structure)
        record[f"exp_{arm}_ff"] = _rmsd_to_experiment(relaxed[arm], result.structure)
    record["outcome"] = "compared"
    return record


def _rmsd_to_experiment(framework, experimental) -> float | None:
    """Per-atom RMSD of a build against the crystal it came from.

    The decisive metric. UFF energy only says which minimum the force
    field prefers; a self-template has a known right answer, so the
    question is which build lands nearer it.

    Atom order differs between a build and its source, so the
    correspondence is the same per-element Hungarian match on minimum
    -image fractional coordinates that ``relax`` uses to map relaxed
    atoms back onto the graph. Both arms are compared identically, so
    whatever the metric's absolute bias, the *difference* between them
    is meaningful.
    """
    from autografs.relax import _match_displacements

    matrix = np.asarray(experimental.lattice.matrix, dtype=float)
    reference = np.asarray(experimental.frac_coords, dtype=float) % 1.0
    ref_symbols = [s.specie.symbol for s in experimental]
    built = framework.structure
    if len(built) != len(experimental):
        return None
    candidate = np.asarray(built.frac_coords, dtype=float) % 1.0
    symbols = [s.specie.symbol for s in built]
    if sorted(symbols) != sorted(ref_symbols):
        return None
    try:
        displacements = _match_displacements(
            reference, ref_symbols, candidate, symbols, matrix
        )
    except Exception:  # noqa: BLE001 - an unmatched structure is data
        return None
    moved = np.linalg.norm(displacements @ matrix, axis=1)
    return round(float(np.sqrt((moved**2).mean())), 4)


def _med(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.median(values), 4) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", nargs="?")
    parser.add_argument("-o", "--output", default="embedding-vs-ff-self.json")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--one", default=None)
    args = parser.parse_args()

    if args.one:
        print(json.dumps(compare(args.one), default=str))
        return
    if not args.corpus:
        raise SystemExit("corpus is required")

    paths = _collect(args.corpus)[: args.limit]
    print(f"{len(paths)} structures\n", flush=True)
    records = []
    for index, path in enumerate(paths, 1):
        record: dict = {"name": Path(str(path)).name, "outcome": "timeout"}
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, our own module
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--one",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
            for line in reversed(proc.stdout.splitlines()):
                if line.startswith("{"):
                    record = json.loads(line)
                    break
        except subprocess.TimeoutExpired:
            pass
        records.append(record)
        print(
            f"[{index}/{len(paths)}] {record['name'][:32]:34s} {record['outcome']:18s} "
            f"built {record.get('rmsd_built', '-')}  endpoints "
            f"{record.get('rmsd_endpoints', '-')}  dE {record.get('energy_gap', '-')}",
            flush=True,
        )

    done = [r for r in records if r["outcome"] == "compared"]
    moved = [r for r in done if (r.get("rmsd_built") or 0) > 1e-4]
    payload = {
        "study": "embedding-vs-ff-on-self-templates",
        "n": len(records),
        "compared": len(done),
        "relaxation_moved_the_build": len(moved),
        "medians": {
            "rmsd_built": _med(done, "rmsd_built"),
            "rmsd_endpoints": _med(done, "rmsd_endpoints"),
            "rmsd_built_where_moved": _med(moved, "rmsd_built"),
            "rmsd_endpoints_where_moved": _med(moved, "rmsd_endpoints"),
            "energy_gap": _med(done, "energy_gap"),
            "energy_gap_where_moved": _med(moved, "energy_gap"),
            "exp_fixed_built": _med(done, "exp_fixed_built"),
            "exp_freed_built": _med(done, "exp_freed_built"),
            "exp_fixed_ff": _med(done, "exp_fixed_ff"),
            "exp_freed_ff": _med(done, "exp_freed_ff"),
        },
        # the verdict: does freeing the slots land nearer the crystal,
        # as built and after the force field? Counted per structure,
        # because a median over 16 hides which way individuals went
        "closer_to_experiment": {
            stage: {
                "freed": sum(
                    1
                    for r in done
                    if r.get(f"exp_freed_{stage}") is not None
                    and r.get(f"exp_fixed_{stage}") is not None
                    and r[f"exp_freed_{stage}"] < r[f"exp_fixed_{stage}"]
                ),
                "fixed": sum(
                    1
                    for r in done
                    if r.get(f"exp_freed_{stage}") is not None
                    and r.get(f"exp_fixed_{stage}") is not None
                    and r[f"exp_freed_{stage}"] > r[f"exp_fixed_{stage}"]
                ),
            }
            for stage in ("built", "ff")
        },
        "records": records,
    }
    Path(args.output).write_text(json.dumps(payload, indent=1, default=str))
    print(f"\ncompared {len(done)}; relaxation moved {len(moved)} builds")
    print(json.dumps(payload["medians"], indent=1))


if __name__ == "__main__":
    main()
