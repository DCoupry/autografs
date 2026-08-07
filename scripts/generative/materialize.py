"""Rebuild selected recombination candidates and persist them.

``recombine.py --write-usable`` only saves builds that are already
clash-free, which is the wrong population to hand a force field: fixing
packing is precisely what a relaxation is *for*, so the candidates worth
relaxing are the ones whose **bonds** close, clashing or not.

Rather than re-running a sweep (the geometric sieve dominates it), this
rebuilds straight from the records: a build is fully determined by its
net, its SBU names, and the blueprint positions they occupy. Older
records predate ``slot_order`` being stored, so the sieve is re-run once
per net to recover it - still far cheaper than a whole sweep.

Selection is deduplicated on the assembly fingerprint: several sampled
mappings can land on the same (net, blocks, fold) assembly, and that is
one hypothetical material.

Usage:
    python scripts/generative/materialize.py a.json b.json \\
        --xyz vocabulary.xyz -o candidates/ --criterion closed
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recombine import coverable_nets, is_genuine_framework  # noqa: E402

from autografs import Autografs  # noqa: E402
from autografs.exceptions import AutografsError  # noqa: E402

_WORKER: dict = {}


def select(reports: list[Path], criterion: str, genuine_only: bool) -> dict[str, dict]:
    """Records worth materializing, deduplicated by assembly."""
    chosen: dict[str, dict] = {}
    for report in reports:
        payload = json.loads(report.read_text(encoding="utf-8"))
        for record in payload["records"]:
            if record["outcome"] != "built" or not record.get("novel"):
                continue
            if not record.get("connected", True):
                continue
            if criterion == "closed" and not record.get("closed"):
                continue
            if criterion == "usable" and not record.get("usable"):
                continue
            if genuine_only and not is_genuine_framework(record):
                continue
            chosen.setdefault(record["fingerprint"], record)
    return chosen


def _init(xyzfile: str, topofile: str | None, subset_prefixes: tuple[str, ...]) -> None:
    _WORKER["mofgen"] = Autografs(xyzfile=xyzfile, topofile=topofile)
    _WORKER["subset"] = sorted(
        n for n in _WORKER["mofgen"].sbu if n.startswith(subset_prefixes)
    )


def _rebuild_net(task: tuple) -> list[dict]:
    """Rebuild every selected record for one net, saving each framework."""
    net, records, out_dir, max_rmsd = task
    mofgen: Autografs = _WORKER["mofgen"]
    out = Path(out_dir)
    results: list[dict] = []
    topology = mofgen.topologies[net]
    ordering = list(topology.mappings)

    order = records[0].get("slot_order")
    empty = records[0].get("empty_slot_types") or []
    if order is None:
        # pre-`slot_order` records: recover the ordering from the sieve,
        # which is deterministic for a given vocabulary
        plans = coverable_nets(mofgen, _WORKER["subset"], only=[net], verbose=False)
        if not plans:
            return [{"net": net, "error": "net no longer coverable"}]
        order, empty = plans[0]["slot_order"], plans[0]["empty_slot_types"]

    for record in records:
        mappings: dict = {
            ordering[position]: name
            for position, name in zip(order, record["sbus"], strict=False)
        }
        for position in empty:
            mappings[ordering[position]] = None
        stem = (
            f"{net}-{hashlib.sha256(record['fingerprint'].encode()).hexdigest()[:10]}"
        )
        try:
            framework = mofgen.build(
                topology,
                {k: copy.deepcopy(v) if v else v for k, v in mappings.items()},
                max_rmsd=max_rmsd,
            )
        except (AutografsError, KeyError, ValueError) as exc:
            results.append(
                {
                    "net": net,
                    "stem": stem,
                    "error": f"{type(exc).__name__}: {exc}"[:140],
                }
            )
            continue
        except Exception as exc:  # noqa: BLE001 - a rebuild bug is data
            results.append(
                {
                    "net": net,
                    "stem": stem,
                    "error": f"{type(exc).__name__}: {exc}"[:140],
                    "traceback": traceback.format_exc(limit=3),
                }
            )
            continue
        framework.save(str(out / f"{stem}.json"))
        results.append(
            {
                "net": net,
                "stem": stem,
                "written": stem,
                "fingerprint": record["fingerprint"],
                "sbus": record["sbus"],
                "n_atoms": record.get("n_atoms"),
                "n_free": record.get("n_free"),
                "min_contact": record.get("min_contact"),
                "usable": record.get("usable"),
                "novel": True,
                "outcome": "built",
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+")
    parser.add_argument(
        "--xyz", required=True, help="the vocabulary they were built from"
    )
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument(
        "--criterion",
        default="closed",
        choices=("closed", "usable"),
        help="closed = bonds right, packing left to the force field",
    )
    parser.add_argument("--all-kinds", action="store_true", help="skip the linker test")
    parser.add_argument("--max-rmsd", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--topofile", default=None)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    chosen = select([Path(p) for p in args.reports], args.criterion, not args.all_kinds)
    records = list(chosen.values())
    if args.limit:
        records = records[: args.limit]
    print(f"{len(records)} distinct assemblies to materialize ({args.criterion})")

    by_net: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_net[record["net"]].append(record)
    print(f"over {len(by_net)} nets")

    tasks = [
        (net, group, str(out), args.max_rmsd) for net, group in sorted(by_net.items())
    ]
    started = time.perf_counter()
    written: list[dict] = []
    if args.n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=args.n_jobs,
            initializer=_init,
            initargs=(args.xyz, args.topofile, ("node_", "linker_")),
        ) as pool:
            for index, group in enumerate(pool.map(_rebuild_net, tasks), 1):
                written.extend(group)
                print(
                    f"[{index}/{len(tasks)}] {group[0]['net']}: "
                    f"{sum(1 for r in group if 'written' in r)}/{len(group)} rebuilt",
                    flush=True,
                )
    else:
        _init(args.xyz, args.topofile, ("node_", "linker_"))
        for index, task in enumerate(tasks, 1):
            group = _rebuild_net(task)
            written.extend(group)
            print(
                f"[{index}/{len(tasks)}] {task[0]}: "
                f"{sum(1 for r in group if 'written' in r)}/{len(group)} rebuilt",
                flush=True,
            )

    ok = [r for r in written if "written" in r]
    manifest = out / "manifest.json"
    # shaped like a recombine report so relax_recombined.py consumes it
    manifest.write_text(
        json.dumps(
            {
                "benchmark": "materialize",
                "criterion": args.criterion,
                "summary": {"attempted": len(records)},
                "records": ok,
            },
            indent=1,
            default=str,
        )
    )
    print(
        f"\n{len(ok)}/{len(records)} rebuilt in "
        f"{time.perf_counter() - started:.0f}s -> {manifest}"
    )
    failed = [r for r in written if "written" not in r]
    if failed:
        print(f"  {len(failed)} could not be rebuilt; first: {failed[0].get('error')}")


if __name__ == "__main__":
    main()
