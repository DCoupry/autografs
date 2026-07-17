"""
Staged multi-fidelity screening driver (the "funnel").

Drives a candidate population of built frameworks through escalating
levels of theory, computing the properties each level unlocks,
selecting the top N, and escalating the survivors:

    asbuilt   no relaxation; geometric descriptors only
    uff4mof   LAMMPS/UFF4MOF relax; energy, elastic moduli, EQeq
              charges, geometry recomputed on the relaxed structure
    gfn-ff    periodic GFN-FF relax (xtb-python) on the survivors
    gfn1      GFN1-xTB relax (tblite)
    dftb      DFTB+ relax (binary + Slater-Koster files)

Every candidate keeps full provenance in the work directory: the
framework JSON after each level (charges included), a properties
record per level, and the selection history in ``funnel.json``.
Finished (candidate, level) pairs are recorded atomically and skipped
on restart, so an interrupted run resumes where it stopped — the same
contract as build_all checkpointing (#121), whose output directories
are directly usable as funnel input.

Usage::

    python scripts/funnel/funnel.py run "ckpt/*.json.gz" --workdir funnel/ \
        --levels asbuilt,uff4mof --keep 20 --rank-by void_fraction
    python scripts/funnel/funnel.py report funnel/

``report`` prints the per-level rankings and the Spearman rank
correlation of every property between adjacent levels — the
rank-stability data that decides each property's safe screening level.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("funnel")


@dataclass(frozen=True)
class Level:
    """One rung of the funnel."""

    name: str
    relax: str | None  # None, "uff4mof"/"uff"/"dreiding" (LAMMPS), or ASE name
    elastic: bool = False
    charges: bool = False


LEVELS = {
    "asbuilt": Level("asbuilt", None),
    "uff4mof": Level("uff4mof", "uff4mof", elastic=True, charges=True),
    "gfn-ff": Level("gfn-ff", "gfn-ff"),
    "gfn1": Level("gfn1", "gfn1"),
    "dftb": Level("dftb", "dftb"),
}

LAMMPS_FORCE_FIELDS = {"uff4mof": "UFF4MOF", "uff": "UFF", "dreiding": "Dreiding"}


def geometric_properties(framework, spacing: float) -> dict:
    """The descriptors available at every level."""
    return {
        "density": float(framework.density),
        "void_fraction": float(framework.void_fraction(spacing=spacing)),
        "largest_cavity_diameter": float(
            framework.largest_cavity_diameter(spacing=spacing)
        ),
        "pore_limiting_diameter": float(
            framework.pore_limiting_diameter(spacing=spacing)
        ),
    }


def run_level(
    framework,
    level: Level,
    spacing: float,
    elastic: bool = True,
    charges: bool = True,
):
    """One candidate through one level: relax + properties.

    Returns (framework, properties). The framework is the relaxed
    (and possibly charged) structure to carry to the next level.
    """
    if level.relax is not None:
        if level.relax in LAMMPS_FORCE_FIELDS:
            framework = framework.relax(force_field=LAMMPS_FORCE_FIELDS[level.relax])
        else:
            framework = framework.relax(calculator=level.relax)
    properties = geometric_properties(framework, spacing)
    if framework.energy is not None:
        properties["energy"] = float(framework.energy)
        properties["energy_per_atom"] = float(framework.energy) / len(framework)
    if level.elastic and elastic:
        elastic_props = framework.elastic_properties()
        properties.update(
            bulk_hill=elastic_props.bulk_hill,
            shear_hill=elastic_props.shear_hill,
            young_hill=elastic_props.young_hill,
            young_min=elastic_props.young_min,
            young_max=elastic_props.young_max,
            born_stable=bool(elastic_props.is_stable),
        )
    if level.charges and charges:
        framework = framework.assign_charges("eqeq")
        assigned = framework.charges
        assert assigned is not None
        properties["charge_min"] = float(assigned.min())
        properties["charge_max"] = float(assigned.max())
    return framework, properties


def _write_json(path: Path, payload: dict) -> None:
    """Atomic JSON write: finished records never appear half-written."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def select_top(
    records: dict[str, dict], key: str, keep: int, ascending: bool = False
) -> list[str]:
    """The candidate names surviving a selection round.

    Candidates whose record lacks the key (or errored) are dropped;
    ties break by name for reproducibility.
    """
    ranked = sorted(
        (name for name, props in records.items() if key in props),
        key=lambda name: (
            records[name][key] if ascending else -records[name][key],
            name,
        ),
    )
    return ranked[:keep]


def run_funnel(
    inputs: list[str],
    workdir: str | Path,
    levels: list[str],
    keep: list[int],
    rank_by: str,
    ascending: bool = False,
    spacing: float = 0.5,
    elastic: bool = True,
    charges: bool = True,
) -> dict:
    """Drive the full funnel; returns (and writes) the summary."""
    from autografs.framework import Framework

    for name in levels:
        if name not in LEVELS:
            raise ValueError(f"Unknown level {name!r}; known: {sorted(LEVELS)}.")
    if len(keep) != max(len(levels) - 1, 0):
        raise ValueError(
            f"{len(levels)} levels need {len(levels) - 1} keep counts "
            f"(one per escalation), got {len(keep)}."
        )
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for pattern in inputs:
        matched = sorted(glob.glob(pattern)) or [pattern]
        for item in matched:
            path = Path(item)
            stem = path.name.removesuffix(".gz").removesuffix(".json")
            paths[stem] = path
    if not paths:
        raise ValueError(f"No candidate frameworks matched {inputs}.")
    logger.info(f"Funnel over {len(paths)} candidates: {' -> '.join(levels)}")

    survivors = sorted(paths)
    history = []
    for rung, level_name in enumerate(levels):
        level = LEVELS[level_name]
        records: dict[str, dict] = {}
        for name in survivors:
            candidate_dir = workdir / name
            candidate_dir.mkdir(exist_ok=True)
            props_path = candidate_dir / f"{level_name}.props.json"
            frame_path = candidate_dir / f"{level_name}.json.gz"
            if props_path.exists():
                records[name] = json.loads(props_path.read_text(encoding="utf-8"))
                continue
            try:
                if rung == 0:
                    framework = Framework.load(paths[name])
                else:
                    framework = Framework.load(
                        workdir / name / f"{levels[rung - 1]}.json.gz"
                    )
                framework, properties = run_level(
                    framework, level, spacing, elastic=elastic, charges=charges
                )
                framework.save(frame_path)
            except Exception as exc:
                logger.warning(f"{name} failed at {level_name}: {exc}")
                properties = {"error": f"{type(exc).__name__}: {exc}"}
            _write_json(props_path, properties)
            records[name] = properties
        entry: dict = {"level": level_name, "evaluated": sorted(records)}
        if rung < len(levels) - 1:
            survivors = select_top(records, rank_by, keep[rung], ascending)
            entry["selected"] = survivors
            logger.info(
                f"{level_name}: kept {len(survivors)}/{len(records)} by {rank_by}"
            )
        history.append(entry)

    summary = {
        "levels": levels,
        "rank_by": rank_by,
        "ascending": ascending,
        "keep": keep,
        "history": history,
    }
    _write_json(workdir / "funnel.json", summary)
    return summary


def rank_stability(workdir: str | Path) -> dict[str, dict[str, float]]:
    """Spearman rank correlation of each property between level pairs.

    For every property present at two adjacent levels of a finished
    run, correlates the values over the candidates evaluated at both.
    High correlation means the cheaper level ranks candidates the same
    way — it is a safe screening level for that property.
    """
    from scipy.stats import spearmanr

    workdir = Path(workdir)
    summary = json.loads((workdir / "funnel.json").read_text(encoding="utf-8"))
    levels = summary["levels"]
    table: dict[str, dict[str, float]] = {}
    for low, high in zip(levels, levels[1:], strict=False):
        pair = f"{low}->{high}"
        merged: dict[str, dict[str, tuple]] = {}
        for candidate in workdir.iterdir():
            low_path = candidate / f"{low}.props.json"
            high_path = candidate / f"{high}.props.json"
            if not (low_path.exists() and high_path.exists()):
                continue
            low_props = json.loads(low_path.read_text(encoding="utf-8"))
            high_props = json.loads(high_path.read_text(encoding="utf-8"))
            for key in set(low_props) & set(high_props):
                if isinstance(low_props[key], (int, float)) and not isinstance(
                    low_props[key], bool
                ):
                    merged.setdefault(key, {})[candidate.name] = (
                        low_props[key],
                        high_props[key],
                    )
        for key, values in merged.items():
            if len(values) < 3:
                continue
            lows, highs = zip(*values.values(), strict=True)
            rho = spearmanr(lows, highs).statistic
            table.setdefault(key, {})[pair] = float(rho)
    return table


def _main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    runner = sub.add_parser("run", help="drive the funnel")
    runner.add_argument("inputs", nargs="+", help="framework JSONs or globs")
    runner.add_argument("--workdir", required=True)
    runner.add_argument(
        "--levels",
        default="asbuilt,uff4mof",
        help=f"comma-separated, from {sorted(LEVELS)}",
    )
    runner.add_argument(
        "--keep",
        default=None,
        help="comma-separated survivor counts, one per escalation",
    )
    runner.add_argument("--rank-by", default="void_fraction")
    runner.add_argument("--ascending", action="store_true")
    runner.add_argument("--spacing", type=float, default=0.5)
    runner.add_argument("--no-elastic", action="store_true")
    runner.add_argument("--no-charges", action="store_true")

    reporter = sub.add_parser("report", help="rank-stability report")
    reporter.add_argument("workdir")

    args = parser.parse_args(argv)
    if args.command == "run":
        levels = [name.strip() for name in args.levels.split(",") if name.strip()]
        if args.keep is None:
            keep = [10] * max(len(levels) - 1, 0)
        else:
            keep = [int(k) for k in args.keep.split(",") if k.strip()]
        run_funnel(
            args.inputs,
            args.workdir,
            levels,
            keep,
            rank_by=args.rank_by,
            ascending=args.ascending,
            spacing=args.spacing,
            elastic=not args.no_elastic,
            charges=not args.no_charges,
        )
        return 0
    table = rank_stability(args.workdir)
    if not table:
        print("No overlapping property records found.")
        return 1
    for key in sorted(table):
        for pair, rho in table[key].items():
            print(f"{key:28s} {pair:22s} rho = {rho:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
