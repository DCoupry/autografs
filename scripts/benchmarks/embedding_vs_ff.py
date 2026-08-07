"""Is the embedding relaxation redundant once a force field runs?

The geometric relaxation frees a blueprint's symmetry-allowed slot
displacements; the force field then moves every atom anyway. If both
paths converge on the same relaxed structure, the geometric step buys
nothing wherever an FF step follows, and its scope shrinks to outputs
that never see one (as-built screening, porosity descriptors, chemistry
with no FF parameters).

So each structure is built **twice** - fixed slots and freed - and both
builds are handed the same full UFF minimization. What matters is the
*endpoint*:

* **energy gap** after FF. Same minimum, or a different one?
* **RMSD between the two FF endpoints**, atom by atom (identical graph
  and atom order, so the correspondence is exact). This is the direct
  test: near zero means redundant.
* FF wall time and the displacement each path required, as secondary
  evidence about cost rather than outcome.

Nets are chosen by *how much freedom they actually have*
(``symmetry.orbit_displacements``): a fully pinned net gives the
relaxation nothing to do, and including some is deliberate - the
"it does nothing here" case belongs in the record too.

Usage:
    python scripts/benchmarks/embedding_vs_ff.py -o embedding-vs-ff.json
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

from autografs import Autografs
from autografs.exceptions import AutografsError


def _pick_builds(mofgen: Autografs, max_nets: int) -> list[dict]:
    """Buildable (net, mapping) specs, spanning slot freedom."""
    from autografs.symmetry import orbit_displacements

    specs: list[dict] = []
    for entry in mofgen.topologies.raw_items():
        if len(specs) >= max_nets:
            break
        name = entry[0] if isinstance(entry, tuple) else entry
        try:
            topology = mofgen.topologies[name]
        except Exception:  # noqa: BLE001 - a bad library entry is not our test
            continue
        # keep builds small enough to force-field twice in reasonable
        # time; layer nets are excluded because their slab padding makes
        # a cell comparison meaningless
        if topology.is_2d or len(topology) > 40:
            continue
        try:
            free = int(orbit_displacements(topology).n_free)
            compatible = mofgen.list_building_units(sieve=name)
        except Exception:  # noqa: BLE001
            continue
        # every slot type must have at least one compatible SBU, or the
        # net is not buildable and tells us nothing either way
        if not compatible or not all(compatible.get(s) for s in topology.mappings):
            continue
        mapping = {slot: compatible[slot][0] for slot in topology.mappings}
        specs.append({"net": name, "n_free": free, "mapping": mapping})
    return specs


def _compare(mofgen: Autografs, spec: dict) -> dict:
    """Build fixed and freed, FF-relax both, compare the endpoints."""
    record: dict = {"net": spec["net"], "n_free": spec["n_free"]}
    topology = mofgen.topologies[spec["net"]]
    built = {}
    for arm, relax in (("fixed", False), ("freed", True)):
        try:
            built[arm] = mofgen.build(
                topology,
                mappings=spec["mapping"],
                max_rmsd=0.5,
                min_distance=None,
                relax_embedding=relax,
            )
        except (AutografsError, ValueError, KeyError) as exc:
            record["outcome"] = f"build_failed_{arm}"
            record["error"] = f"{type(exc).__name__}: {exc}"[:120]
            return record
    record["n_atoms"] = len(built["fixed"].structure)
    # geometry as built, before any force field
    for arm in ("fixed", "freed"):
        record[f"contact_{arm}_built"] = round(float(built[arm].min_contact()), 3)
    coords_built = {a: np.asarray(f.cart_coords, float) for a, f in built.items()}
    record["rmsd_built"] = round(
        float(
            np.sqrt(
                ((coords_built["freed"] - coords_built["fixed"]) ** 2).sum(1).mean()
            )
        ),
        4,
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
        record[f"energy_{arm}"] = (
            round(float(relaxed[arm].energy), 3)
            if relaxed[arm].energy is not None
            else None
        )
    # THE test: do the two force-field endpoints coincide? Identical
    # graph and atom order, so this is a direct atom-by-atom comparison
    after = {a: np.asarray(f.cart_coords, float) for a, f in relaxed.items()}
    record["rmsd_endpoints"] = round(
        float(np.sqrt(((after["freed"] - after["fixed"]) ** 2).sum(1).mean())), 4
    )
    if record["energy_fixed"] is not None and record["energy_freed"] is not None:
        record["energy_gap"] = round(record["energy_freed"] - record["energy_fixed"], 3)
    record["outcome"] = "compared"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", default="embedding-vs-ff.json")
    parser.add_argument("--max-nets", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=1500.0)
    parser.add_argument("--one", default=None, help="internal: one net")
    args = parser.parse_args()

    mofgen = Autografs()
    if args.one:
        specs = _pick_builds(mofgen, args.max_nets)
        spec = next(s for s in specs if s["net"] == args.one)
        print(json.dumps(_compare(mofgen, spec), default=str))
        return

    specs = _pick_builds(mofgen, args.max_nets)
    print(
        f"{len(specs)} buildable nets; free-displacement counts "
        f"{sorted({s['n_free'] for s in specs})}\n",
        flush=True,
    )

    records = []
    for index, spec in enumerate(specs, 1):
        record: dict = {"net": spec["net"], "outcome": "aborted"}
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, our own module
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--one",
                    spec["net"],
                    "--max-nets",
                    str(args.max_nets),
                ],
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            # one slow net must not end the sweep and silently truncate
            # the population - that is how a partial run reads as a
            # complete one
            record["outcome"] = "timeout"
            records.append(record)
            print(f"[{index}/{len(specs)}] {spec['net']:>8} timeout", flush=True)
            continue
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("{"):
                record = json.loads(line)
                break
        records.append(record)
        print(
            f"[{index}/{len(specs)}] {spec['net']:>8} n_free={record.get('n_free', '?')} "
            f"{record['outcome']}  rmsd_endpoints={record.get('rmsd_endpoints', '-')} "
            f"dE={record.get('energy_gap', '-')}",
            flush=True,
        )

    done = [r for r in records if r["outcome"] == "compared"]
    acting = [r for r in done if r["n_free"] >= 2]
    payload = {
        "study": "embedding-relaxation-vs-force-field",
        "n": len(records),
        "compared": len(done),
        "with_freedom": len(acting),
        "medians": {
            "rmsd_built": _med(done, "rmsd_built"),
            "rmsd_endpoints": _med(done, "rmsd_endpoints"),
            "rmsd_endpoints_with_freedom": _med(acting, "rmsd_endpoints"),
            "energy_gap": _med(done, "energy_gap"),
            "energy_gap_with_freedom": _med(acting, "energy_gap"),
        },
        "records": records,
    }
    Path(args.output).write_text(json.dumps(payload, indent=1, default=str))
    print(f"\ncompared {len(done)} ({len(acting)} with n_free>=2) -> {args.output}")
    print(json.dumps(payload["medians"], indent=1))


def _med(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.median(values), 4) if values else None


if __name__ == "__main__":
    main()
