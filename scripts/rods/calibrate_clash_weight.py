"""Corpus calibration of the covalent-contact bound's weight.

``_RodBuild.clash_weight`` charges for non-bonded atoms closer than the
sum of their Cordero radii. A weight chosen from one structure is worth
nothing - the DIRECTION_WEIGHT calibration made exactly that mistake and
picked a value that was optimal on HKUST-1 and inert on the corpus - so
this sweeps a population, and a population of two kinds:

* **treatment**: recombinations (one crystal's rod on another's
  blueprint), which is where overlaps actually occur and where the bound
  is supposed to help;
* **control**: self-templates, each structure rebuilt on its own
  blueprint, which are already faithful. The bound must not move these.
  A weight that rescues the first arm by degrading the second is not a
  good weight, and only reporting the pooled figure would hide it.

Both closure (worst inter-unit bond deviation) and packing (closest
contact) are recorded at every weight, because the bound trades one for
the other by construction and the whole question is the exchange rate.

Usage:
    python scripts/rods/calibrate_clash_weight.py MANIFEST -o calib.json
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from _corpus import collect as _collect  # noqa: E402
from embedding import bond_residuals  # noqa: E402
from swap_rod_units import _rehome_laterals, harvest  # noqa: E402

import autografs.rod_build as rod_build  # noqa: E402
from autografs import Autografs  # noqa: E402
from autografs.exceptions import AutografsError  # noqa: E402
from autografs.rod_build import build_rod_framework  # noqa: E402

WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0)


def _build(
    host: dict, rod, laterals: dict, weight: float, scale_band: float | None
) -> dict | None:
    """One build at this weight.

    ``scale_band`` must match how the arm is built in production or the
    comparison is against a straw man: the self-template path bands the
    transverse scale (the blueprint is at the crystal's own size), while
    a recombination deliberately leaves it free because a swapped unit
    is *not* at that size.
    """
    rod_build._RodBuild.clash_weight = weight
    try:
        framework = build_rod_framework(
            host["topology"],
            rod,
            {slot: copy.deepcopy(f) for slot, f in laterals.items()},
            run=host["run"],
            min_distance=None,
            bond_tolerance=1e9,  # record closure, never gate on it here
            verify_net=False,
            initial_scale=1.0,
            scale_band=scale_band,
        )
    except (AutografsError, KeyError, ValueError):
        return None
    except Exception:  # noqa: BLE001 - a build bug is data, not a crash
        return None
    finally:
        rod_build._RodBuild.clash_weight = 0.0
    residual = bond_residuals(framework)
    return {
        "contact": round(float(framework.min_contact()), 3),
        "worst_bond": round(float(residual["max"]), 3),
        "median_bond": round(float(residual["median"]), 3),
        "volume": round(float(framework.structure.volume), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus")
    parser.add_argument("-o", "--output", default="clash-calibration.json")
    parser.add_argument("--limit", type=int, default=26)
    args = parser.parse_args()

    gen = Autografs()
    stock = harvest(gen, _collect(args.corpus)[: args.limit], verbose=False)
    print(f"harvested {len(stock)} rod structures", flush=True)
    if len(stock) < 2:
        raise SystemExit("need at least two structures")

    # control: every structure on its own blueprint with its own units
    control = [(entry, entry["rod"], entry["laterals"]) for entry in stock]
    # treatment: the first compatible linker swap for each host
    treatment = []
    for host in stock:
        for donor in stock:
            if donor is host:
                continue
            if set(host["lateral_arity"].values()) <= set(
                donor["lateral_arity"].values()
            ):
                try:
                    treatment.append((host, host["rod"], _rehome_laterals(host, donor)))
                except KeyError:
                    continue
                break
    print(f"{len(control)} control, {len(treatment)} treatment builds per weight\n")

    results: dict[str, dict[str, list]] = {
        arm: {str(w): [] for w in WEIGHTS} for arm in ("control", "treatment")
    }
    # the self-template path bands the scale, the swap path does not
    bands = {"control": 0.25, "treatment": None}
    for arm, cases in (("control", control), ("treatment", treatment)):
        for index, (host, rod, laterals) in enumerate(cases, 1):
            for weight in WEIGHTS:
                record = _build(host, rod, laterals, weight, bands[arm])
                if record is not None:
                    record["host"] = host["name"]
                    results[arm][str(weight)].append(record)
            print(f"  {arm} [{index}/{len(cases)}] {host['name'][:34]}", flush=True)

    summary: dict[str, dict[str, dict]] = {}
    for arm in ("control", "treatment"):
        summary[arm] = {}
        for weight in WEIGHTS:
            rows = results[arm][str(weight)]
            if not rows:
                continue
            contacts = sorted(r["contact"] for r in rows)
            summary[arm][str(weight)] = {
                "n": len(rows),
                "contact_median": round(statistics.median(contacts), 3),
                "contact_p10": round(contacts[int(0.1 * len(contacts))], 3),
                "clash_free": sum(1 for c in contacts if c >= 1.5),
                "worst_bond_median": round(
                    statistics.median(r["worst_bond"] for r in rows), 3
                ),
                "volume_median": round(statistics.median(r["volume"] for r in rows), 1),
            }
    payload = {
        "study": "clash-weight-calibration",
        "weights": list(WEIGHTS),
        "summary": summary,
        "records": results,
    }
    Path(args.output).write_text(json.dumps(payload, indent=1, default=str))
    print(f"\n-> {args.output}")
    for arm in ("control", "treatment"):
        print(f"\n{arm}:")
        for weight, row in summary[arm].items():
            print(
                f"  w={weight:>5}  n={row['n']:3d}  contact {row['contact_median']:6.3f} "
                f"(p10 {row['contact_p10']:6.3f}, clash-free {row['clash_free']:3d})  "
                f"worst bond {row['worst_bond_median']:6.3f}  "
                f"vol {row['volume_median']}"
            )


if __name__ == "__main__":
    main()
