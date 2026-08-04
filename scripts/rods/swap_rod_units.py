"""Harvest rods and lateral SBUs from real rod MOFs, then recombine them.

The self-templated round trip (coverage plan, rod arm) established that a
rod structure's own units, on its own blueprint, regenerate it. That is a
*fidelity* result. This driver asks the generative question it licenses:
take the blueprint of one crystal, put a **different** crystal's rod or
linkers on it, and see what builds.

A swap is only offered when the blueprint's own connectivity accepts it:

* a substitute rod must match the run's screw order (a helical run's node
  slots are the screw's orbit, filled one repeat each) and carry the same
  number of arms per repeat as the node slot has connection points;
* a substitute linker must carry as many connection points as the slot it
  fills.

Everything else the builder decides. In particular the transverse scale
is seeded at 1 but **not** banded: a self-template is at its own crystal's
size by construction, and a swapped unit is not, so the cell genuinely has
to move.

Usage:
    python scripts/rods/swap_rod_units.py CORPUS -o swaps/ --limit 60
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from _corpus import collect as _collect  # noqa: E402

from autografs import Autografs  # noqa: E402
from autografs.exceptions import AutografsError  # noqa: E402
from autografs.extract_topology import (  # noqa: E402
    rod_topology_from_deconstruction,
)
from autografs.rod_build import build_rod_framework  # noqa: E402
from autografs.rods import save_rods  # noqa: E402

# below this closest-contact a build is not a candidate structure, whatever
# its bonds do; the library's own default overlap gate is 1.0 A and a
# comfortable framework sits well above it
CONTACT_FLOOR = 1.5


def harvest(mofgen: Autografs, paths: list[Path], verbose: bool = True) -> list[dict]:
    """Deconstruct each structure into a reusable (blueprint, rod, laterals)."""
    stock: list[dict] = []
    for index, path in enumerate(paths, 1):
        try:
            result = mofgen.deconstruct(str(path))
            if len(result.rod_units) != 1 or result.n_periodic_components != 1:
                continue
            topology, run, mapping, rod = rod_topology_from_deconstruction(result)
            laterals = {
                slot: copy.deepcopy(result.fragments[name])
                for slot, name in mapping.items()
            }
        except (AutografsError, ValueError, KeyError, IndexError):
            continue
        except Exception:  # noqa: BLE001 - one bad CIF must not kill a harvest
            continue
        nodes = run.nodes or run.slots
        entry = {
            "name": path.name,
            "topology": topology,
            "run": run,
            "rod": rod,
            "laterals": laterals,
            # what a substitute has to match
            "screw_order": int(rod.repeat.screw_order),
            "arms_per_repeat": len(rod.arms),
            "node_arity": len(topology.slots[nodes[0]].atoms.indices_from_symbol("X")),
            "lateral_arity": {
                slot: len(fragment.atoms.indices_from_symbol("X"))
                for slot, fragment in laterals.items()
            },
            "rod_formula": _rod_formula(rod),
            "formula": result.structure.composition.reduced_formula,
            "repeat": float(rod.repeat.repeat_length),
        }
        stock.append(entry)
        if verbose:
            print(
                f"[{index}/{len(paths)}] {path.name}: rod {entry['rod_formula']} "
                f"screw {entry['screw_order']} arms {entry['arms_per_repeat']}, "
                f"{len(laterals)} lateral slots",
                flush=True,
            )
    return stock


def _rod_formula(rod) -> str:
    counts = Counter(rod.repeat.symbols)
    return "".join(f"{s}{counts[s]}" for s in sorted(counts))


def _swaps(stock: list[dict]) -> list[tuple[dict, dict, str]]:
    """Every (host blueprint, donor, kind) pair the connectivity allows."""
    pairs: list[tuple[dict, dict, str]] = []
    for host in stock:
        for donor in stock:
            if donor is host:
                continue
            # rod swap: the run's screw orbit and the node's connection
            # count are what the blueprint fixes
            if (
                donor["screw_order"] == host["screw_order"]
                and donor["arms_per_repeat"] == host["arms_per_repeat"]
                and donor["rod_formula"] != host["rod_formula"]
            ):
                pairs.append((host, donor, "rod"))
            # linker swap: the donor must be able to supply a unit of the
            # right connectivity for every lateral slot the host has.
            # Slot *indices* are blueprint-local and mean nothing across
            # structures, so this compares the arities themselves.
            if donor["lateral_arity"] and host["lateral_arity"]:
                wanted = set(host["lateral_arity"].values())
                offered = set(donor["lateral_arity"].values())
                donor_names = {f.name for f in donor["laterals"].values()}
                host_names = {f.name for f in host["laterals"].values()}
                if wanted <= offered and donor_names != host_names:
                    pairs.append((host, donor, "linkers"))
    return pairs


def _rehome_laterals(host: dict, donor: dict) -> dict:
    """The donor's fragments on the *host's* lateral slots, by arity.

    Both sides are keyed by their own blueprint's slot indices, which
    mean nothing to each other; what carries across is how many
    connection points a slot needs. Donor fragments are offered in a
    fixed order per arity so a host with several slots of the same
    arity gets a reproducible assignment rather than an arbitrary one.
    """
    by_arity: dict[int, list] = {}
    for _slot, fragment in sorted(donor["laterals"].items()):
        arity = len(fragment.atoms.indices_from_symbol("X"))
        by_arity.setdefault(arity, []).append(fragment)
    filled: dict = {}
    cursor: dict[int, int] = {}
    for slot, arity in sorted(host["lateral_arity"].items()):
        options = by_arity.get(arity)
        if not options:
            raise KeyError(f"donor has no {arity}-connected unit for slot {slot}")
        index = cursor.get(arity, 0)
        filled[slot] = options[index % len(options)]
        cursor[arity] = index + 1
    return filled


def attempt(
    host: dict, donor: dict, kind: str, max_rmsd: float, relax_embedding: bool = False
) -> dict:
    """Build one recombination and report what came out."""
    record: dict = {
        "host": host["name"],
        "donor": donor["name"],
        "kind": kind,
        "outcome": None,
    }
    rod = host["rod"] if kind == "linkers" else donor["rod"]
    # slot indices are blueprint-local, so a donor's mapping cannot be
    # handed to the host as-is - its keys would land on the host's run
    # slots. Fill the HOST's lateral slots, matching each to a donor
    # fragment of the same connectivity.
    laterals = _rehome_laterals(host, donor) if kind == "linkers" else host["laterals"]
    record["rod_formula"] = _rod_formula(rod)
    record["linkers"] = sorted({f.name for f in laterals.values()})
    start = time.perf_counter()
    try:
        framework = build_rod_framework(
            host["topology"],
            rod,
            {slot: copy.deepcopy(f) for slot, f in laterals.items()},
            run=host["run"],
            max_rmsd=max_rmsd,
            min_distance=None,
            bond_tolerance=1.0,
            verify_net=False,
            # seeded at the host's own size, but NOT banded: a swapped
            # unit is not at that size and the cell has to move
            initial_scale=1.0,
            # the host blueprint's *proportions* belong to its own units,
            # and one transverse scale cannot re-proportion it for a
            # different one - which is the same one-scale limit the
            # embedding chapter measures, met here generatively. Freeing
            # the lateral slot centres is the remedy that work proposes.
            relax_embedding=relax_embedding,
        )
    except (AutografsError, KeyError) as exc:
        record["outcome"] = "refused"
        record["error"] = f"{type(exc).__name__}: {exc}"[:160]
        record["seconds"] = time.perf_counter() - start
        return record
    except Exception as exc:  # noqa: BLE001 - a build bug is data
        record["outcome"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"[:160]
        record["traceback"] = traceback.format_exc(limit=3)
        record["seconds"] = time.perf_counter() - start
        return record
    built = framework.structure
    record["outcome"] = "built"
    record["formula"] = built.composition.reduced_formula
    record["n_atoms"] = len(built)
    record["cell"] = [round(x, 3) for x in built.lattice.abc]
    record["min_contact"] = round(framework.min_contact(), 3)
    # closure and packing are separate claims. A build can satisfy every
    # bond target and still put two atoms 0.4 A apart, which is not a
    # candidate structure - it is a starting point for a relaxation. Only
    # the clash-free ones are worth calling generated.
    record["clash_free"] = record["min_contact"] >= CONTACT_FLOOR
    record["novel"] = record["formula"] not in (host["formula"], donor["formula"])
    record["seconds"] = time.perf_counter() - start
    record["_framework"] = framework
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", help="directory, glob, manifest, or single CIF")
    parser.add_argument("-o", "--output", default="rod-swaps")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--max-swaps", type=int, default=40)
    parser.add_argument("--max-rmsd", type=float, default=0.5)
    parser.add_argument(
        "--relax-embedding",
        action="store_true",
        help="free the lateral slot centres (symmetry-allowed displacements)",
    )
    parser.add_argument("--topofile", default=None)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    corpus = _collect(args.corpus)[: args.limit]
    mofgen = Autografs(topofile=args.topofile) if args.topofile else Autografs()

    print(f"harvesting from {len(corpus)} structures\n")
    stock = harvest(mofgen, corpus)
    print(f"\nharvested {len(stock)} usable rod structures")
    if len(stock) < 2:
        raise SystemExit("need at least two to recombine")

    # persist the library itself - this is what was never saved before.
    # Rods are keyed by source structure: deduplication across sources is
    # merge_rod's job (harvest.py) and would hide which crystal each came
    # from, which is exactly the provenance a swap report needs.
    save_rods(
        {Path(entry["name"]).stem: entry["rod"] for entry in stock},
        out / "rods.json",
    )
    print(f"wrote {out / 'rods.json'}")

    pairs = _swaps(stock)
    print(f"{len(pairs)} connectivity-compatible swaps available")
    # spread the attempts over distinct hosts rather than exhausting one
    by_host: dict[str, list] = {}
    for host, donor, kind in pairs:
        by_host.setdefault(host["name"], []).append((host, donor, kind))
    ordered: list = []
    while len(ordered) < args.max_swaps and any(by_host.values()):
        for key in list(by_host):
            if by_host[key]:
                ordered.append(by_host[key].pop(0))
            if len(ordered) >= args.max_swaps:
                break

    records = []
    for index, (host, donor, kind) in enumerate(ordered, 1):
        record = attempt(
            host, donor, kind, args.max_rmsd, relax_embedding=args.relax_embedding
        )
        framework = record.pop("_framework", None)
        if framework is not None:
            stem = f"{index:03d}-{kind}-{record['formula']}"
            framework.write_cif(str(out / f"{stem}.cif"))
            record["cif"] = f"{stem}.cif"
            # CIF drops the bond graph, and relaxation needs it (the
            # UFF4MOF typing and the inter-unit bonds are exactly what a
            # rod build establishes), so persist the framework itself too
            framework.save(str(out / f"{stem}.json"))
            record["framework"] = f"{stem}.json"
        records.append(record)
        tag = record["outcome"]
        if tag == "built":
            tag = (
                f"built {record['formula']} "
                f"({'novel' if record['novel'] else 'known'}, "
                f"contact {record['min_contact']}"
                f"{'' if record['clash_free'] else ' CLASH'})"
            )
        print(
            f"[{index}/{len(ordered)}] {kind}: {host['name'][:24]} "
            f"<- {donor['name'][:24]} : {tag}",
            flush=True,
        )

    payload = {
        "harvested": len(stock),
        "swaps_available": len(pairs),
        "attempted": len(records),
        "outcomes": dict(Counter(r["outcome"] for r in records)),
        "novel_built": sum(1 for r in records if r.get("novel")),
        "novel_clash_free": sum(
            1 for r in records if r.get("novel") and r.get("clash_free")
        ),
        "contact_floor": CONTACT_FLOOR,
        "records": records,
        "stock": [
            {
                k: v
                for k, v in entry.items()
                if k not in ("topology", "run", "rod", "laterals")
            }
            for entry in stock
        ],
    }
    (out / "swaps.json").write_text(json.dumps(payload, indent=1, default=str))
    print(f"\n{payload['outcomes']}, {payload['novel_built']} novel -> {out}")


if __name__ == "__main__":
    main()
