"""Harvested units on OTHER nets: the finite generative recombination arm.

``scripts/rods/swap_rod_units.py`` asks this question for rods - put one
crystal's rod on another's blueprint. This driver asks it for the finite
pipeline and against the *cataloged* nets: harvest building units from
real crystals, then place them on library blueprints they were never
seen in, alone or mixed with the shipped SBU library.

Three arms, chosen with ``--arm``:

``harvested``
    Every slot filled from the harvested vocabulary. The generative
    question in its pure form - can real chemistry, re-cataloged, be
    re-assembled somewhere else?
``mixed``
    Slots filled from harvested units *and* the shipped library, with at
    least one of each. The cross-vocabulary case, and the one a user
    actually runs: a harvested node with a designed linker.
``library``
    Shipped SBUs only. The control: whatever failure rate this arm shows
    is the pipeline's own, not something harvesting introduced.

Attribution is the point, so gates are measured rather than stacked.
Only the per-slot shape gate (``max_rmsd``) can refuse a build - it is
what decides whether a unit *fits* a slot at all - and everything past
it (bond closure, closest contact) is recorded as a number on a build
that happened. A funnel that stops at the first gate cannot tell a
badly-shaped unit from a well-shaped one in a cell that will not close.

Novelty is a fingerprint lookup, not a formula comparison:
``fingerprint.AssemblyFingerprint`` over (net, block multiset, fold), in
the harvest's own vocabulary, against every source structure the harvest
saw. A build that reproduces a source's assembly is a round trip, not a
generation, and is reported separately.

Usage:
    python scripts/generative/recombine.py CORPUS -o recombine.json \\
        --harvest-limit 60 --nets 300 --per-net 3 --n-jobs 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import numpy as np  # noqa: E402
from _corpus import collect as _collect  # noqa: E402
from embedding import bond_residuals  # noqa: E402

from autografs import Autografs  # noqa: E402
from autografs.deconstruct import merge_fragment  # noqa: E402
from autografs.exceptions import AlignmentError, AutografsError  # noqa: E402
from autografs.fingerprint import AssemblyFingerprint  # noqa: E402

# below this closest non-bonded contact a build is not a candidate
# structure whatever its bonds do - it is a starting point for a
# relaxation. The same floor swap_rod_units.py uses.
CONTACT_FLOOR = 1.5
# a build whose worst inter-unit bond deviates more than this from its
# covalent target has not closed, however well it packs. Reported, never
# gated: build_framework's own bond_tolerance is off by default for
# exactly this reason (a corpus-wide default would change which builds
# the library produces).
CLOSURE_TOLERANCE = 0.5
# the shape gate's own message carries the number we want to attribute
_WORST_RMSD = re.compile(r"worst: ([0-9.]+)")

_WORKER: dict = {}


# --------------------------------------------------------------------
# the vocabulary
# --------------------------------------------------------------------


def _harvest_one(path_str: str) -> dict:
    """Deconstruct one structure into a picklable harvest payload.

    Returns the pieces the parent needs to merge, rather than the
    Deconstruction itself: the per-source fragments, one block name per
    non-cap unit occurrence, and the net candidates. The parent then
    re-expresses the blocks in the MERGED vocabulary, which is the only
    place a cross-structure fingerprint can be computed.
    """
    label = Path(path_str).stem
    mofgen = _WORKER["mofgen"]
    try:
        result = mofgen.deconstruct(path_str)
    except (AutografsError, ValueError, KeyError, IndexError) as exc:
        return {"label": label, "error": f"{type(exc).__name__}: {exc}"[:120]}
    except Exception as exc:  # noqa: BLE001 - one bad CIF must not kill it
        return {"label": label, "error": f"{type(exc).__name__}: {exc}"[:120]}
    if result.rod_units:
        # rods have no finite fragment and their recombination is
        # swap_rod_units.py's subject, not this driver's
        return {
            "label": label,
            "error": "rod structure (see scripts/rods/swap_rod_units.py)",
        }
    return {
        "label": label,
        "fragments": dict(result.fragments),
        "blocks": [unit.name for unit in result.units if unit.kind != "cap"],
        "nets": list(result.net_candidates),
        "fold": int(result.n_periodic_components),
    }


def build_stock(
    mofgen: Autografs,
    paths: list[Path],
    verbose: bool = True,
    n_jobs: int = 1,
    topofile: str | None = None,
) -> dict:
    """Harvest one deduplicated vocabulary, keeping source assemblies.

    ``harvest.harvest`` does the merging already, but discards the
    per-source Deconstruction - and the novelty question needs each
    source's assembly fingerprint expressed in the *merged* vocabulary,
    which only exists once merging is done. So the merge runs here, and
    each source's blocks are re-expressed once the vocabulary is final.

    Deconstruction dominates the wall clock (~10 s per structure) and is
    embarrassingly parallel, so ``n_jobs`` farms it out; the merge stays
    in the parent because it is inherently sequential (each fragment is
    compared against the vocabulary built so far).
    """
    from autografs.fingerprint import _reduced

    fragments: dict = {}
    kinds: dict[str, str] = {}
    provenance: dict[str, list[str]] = {}
    failures: dict[str, str] = {}
    payloads: list[dict] = []

    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_worker,
            initargs=(None, topofile, 0.5, False, None),
        ) as pool:
            stream = pool.map(_harvest_one, [str(p) for p in paths], chunksize=1)
            for index, payload in enumerate(stream, 1):
                payloads.append(payload)
                if verbose and index % 25 == 0:
                    print(f"  [{index}/{len(paths)}] deconstructed", flush=True)
    else:
        _WORKER.setdefault("mofgen", mofgen)
        for index, path in enumerate(paths, 1):
            payloads.append(_harvest_one(str(path)))
            if verbose and index % 25 == 0:
                print(f"  [{index}/{len(paths)}] deconstructed", flush=True)

    kept: list[dict] = []
    for payload in payloads:
        label = payload["label"]
        if "error" in payload:
            failures[label] = payload["error"]
            continue
        local_to_merged: dict[str, str] = {}
        for name, fragment in payload["fragments"].items():
            base = re.sub(r"(X)_\d+$", r"\1", name)
            merged = merge_fragment(fragments, fragment, base)
            local_to_merged[name] = merged
            kinds[merged] = name.split("_", 1)[0]
            provenance.setdefault(merged, [])
            if label not in provenance[merged]:
                provenance[merged].append(label)
        payload["local_to_merged"] = local_to_merged
        kept.append(payload)

    # now that the vocabulary is final, express every source in it. Uses
    # fingerprint's own reducer so these strings are directly comparable
    # to the ones from_framework produces for a build.
    realized: dict[str, AssemblyFingerprint] = {}
    for payload in kept:
        counts: Counter[str] = Counter()
        for name in payload["blocks"]:
            counts[payload["local_to_merged"].get(name, f"unmatched:{name}")] += 1
        realized[payload["label"]] = AssemblyFingerprint(
            nets=tuple(sorted(payload["nets"])),
            blocks=_reduced(counts),
            fold=payload["fold"],
        )
    return {
        "fragments": fragments,
        "kinds": kinds,
        "provenance": provenance,
        "realized": realized,
        "failures": failures,
        "n_sources": len(kept),
    }


def building_units(stock: dict) -> dict:
    """Nodes and linkers - caps are bound solvent, not building blocks."""
    return {
        name: fragment
        for name, fragment in stock["fragments"].items()
        if stock["kinds"].get(name) in ("node", "linker")
    }


# --------------------------------------------------------------------
# what the blueprints will accept
# --------------------------------------------------------------------


def _index_of(ordering: list, slot_type) -> int:
    """Position of a slot type in the blueprint's own ordering.

    By identity: Fragment does not define ``__eq__``, so ``.index`` would
    fall back to identity anyway, but saying so keeps the intent legible
    if Fragment ever grows one.
    """
    for index, candidate in enumerate(ordering):
        if candidate is slot_type:
            return index
    raise KeyError(f"slot type {slot_type!r} is not one of the blueprint's")


def _slot_arity(slot_type) -> int:
    return len(slot_type.atoms.indices_from_symbol("X"))


def _fingerprint_path(xyz_path: Path) -> Path:
    """Sidecar holding the source assemblies a vocabulary came from."""
    return xyz_path.with_suffix(".fingerprints.json")


def net_freedom(topology, cache: dict | None = None) -> int | None:
    """How many symmetry-allowed slot displacements the blueprint has.

    The stratifier this investigation turns on. ``n_free == 0`` means
    every slot centre is pinned by site symmetry and the cell parameters
    are the only freedom - the regime pcu/dia/srs/fcu/sql/hcb live in,
    and (per the embedding work) exactly why those nets build correctly.
    A large ``n_free`` means the blueprint's *proportions* are not
    determined by its symmetry, so the idealized embedding carries
    arbitrary ones that a single scale cannot fix.
    """
    from autografs.symmetry import orbit_displacements

    if cache is not None and topology.name in cache:
        return cache[topology.name]
    try:
        value: int | None = int(orbit_displacements(topology).n_free)
    except Exception:  # noqa: BLE001 - a descriptor must never kill a sweep
        value = None
    if cache is not None:
        cache[topology.name] = value
    return value


def coverable_nets(
    mofgen: Autografs,
    subset: list[str],
    limit: int | None = None,
    verbose: bool = True,
    rng: np.random.Generator | None = None,
    empty_2c: bool = False,
    only: list[str] | None = None,
    max_free: int | None = None,
    freedom_cache: dict | None = None,
) -> list[dict]:
    """Nets whose every slot type has at least one compatible unit.

    A cheap arity prefilter runs first, straight off the raw JSON
    payloads: a net needing a 7-connected slot cannot be covered by a
    vocabulary that offers none, and reading that from ``raw_items``
    costs no Topology reconstruction. Only the survivors pay the
    geometric sieve.

    With ``empty_2c`` the blueprint's 2-connected slot types are left
    **empty** (#179) instead of filled: their neighbours bond directly.
    That is what an edge-decorated net actually needs - HKUST-1 is tbo
    with its 96 two-connected slots empty - and filling them instead
    inserts a whole extra molecule into every edge.

    ``max_free`` keeps only blueprints with at most that many
    symmetry-allowed slot displacements. Low-freedom nets are RARE (of
    200 randomly sampled coverable nets, 3 were pinned and 9 more had
    n_free == 1), so the filter runs BEFORE the geometric sieve: a
    freedom computation costs ~0.3 s against ~2.5 s for a sieve pass
    over a 130-unit vocabulary, and reversing the order means paying the
    expensive test on the ~95% of nets that are about to be discarded.
    """
    offered = {len(mofgen.sbu[name].atoms.indices_from_symbol("X")) for name in subset}
    prefiltered = []
    for name, raw in mofgen.topologies.raw_items():
        wanted = {slot["species"].count("X") for slot in raw["slots"]}
        if empty_2c:
            wanted.discard(2)
        if wanted <= offered:
            prefiltered.append(name)
    if only is not None:
        wanted_names = set(only)
        prefiltered = [name for name in prefiltered if name in wanted_names]
        missing = wanted_names - set(prefiltered)
        if missing and verbose:
            print(f"  [!] not in library or prefiltered out: {sorted(missing)}")
    if verbose:
        print(f"{len(prefiltered)} nets pass the arity prefilter", flush=True)
    if rng is not None and only is None:
        # the library is stored alphabetically and ``limit`` stops at the
        # first N coverable nets, so an unshuffled sweep is a sweep of
        # the a-names: 40 nets meant aab..asc-a, all of them obscure
        # zeolite derivatives, which is a sample of the alphabet rather
        # than of the library
        prefiltered = [prefiltered[i] for i in rng.permutation(len(prefiltered))]
    plans: list[dict] = []
    n_scanned = 0
    for index, name in enumerate(prefiltered, 1):
        topology = mofgen.topologies[name]
        freedom = net_freedom(topology, cache=freedom_cache)
        if max_free is not None:
            n_scanned += 1
            if freedom is None or freedom > max_free:
                continue
        ordering = list(topology.mappings)
        emptied = (
            [i for i, slot_type in enumerate(ordering) if _slot_arity(slot_type) == 2]
            if empty_2c
            else []
        )
        if empty_2c and len(emptied) == len(ordering):
            # nothing would be placed at all
            continue
        available = mofgen.list_building_units(sieve=name, subset=subset)
        needed = [
            slot_type for i, slot_type in enumerate(ordering) if i not in set(emptied)
        ]
        # list_building_units keys its result in SBU-iteration order,
        # which is NOT topology.mappings order, so the slot types a
        # choice belongs to have to travel WITH it as positions in the
        # blueprint's own ordering. Zipping a choice against
        # mappings.keys() instead silently permutes the assignment, and
        # the builder then reports a connectivity mismatch that looks
        # like a chemistry result and is not one.
        order, options = [], []
        for slot_type in needed:
            names = available.get(slot_type)
            if not names:
                break
            order.append(_index_of(ordering, slot_type))
            options.append(sorted(names))
        if len(order) != len(needed):
            continue
        plans.append(
            {
                "net": name,
                "n_slot_types": len(order),
                "n_slots": len(topology),
                "n_free": freedom,
                "n_slot_types_total": len(ordering),
                "empty_slot_types": emptied,
                "slot_order": order,
                "options": options,
                "combinations": math.prod(len(o) for o in options),
            }
        )
        if verbose and len(plans) % 10 == 0:
            print(
                f"  ... {index}/{len(prefiltered)} scanned, {len(plans)} coverable",
                flush=True,
            )
        if limit is not None and len(plans) >= limit:
            break
    if verbose and max_free is not None:
        print(
            f"  freedom filter: {len(plans)} kept of {n_scanned} scanned "
            f"(n_free <= {max_free})",
            flush=True,
        )
    return plans


def balanced_combinations(
    mofgen: Autografs, plan: dict, per_net: int, targets: int = 81
) -> list[tuple[str, ...]]:
    """Candidates whose units are proportioned to suit the net.

    The sieve matches arm DIRECTIONS and never length - for a
    2-connected slot it is vacuous - so sampling from it uniformly is a
    coin flip on size, and size is what decides whether bonded units
    interpenetrate. ``Autografs.suggest_mappings`` returns the single
    best-balanced choice; a generative sweep wants several, so this
    walks the same target-ratio family and keeps the ``per_net``
    distinct choices with the lowest predicted edge-scale spread.
    """
    from autografs.validation import edge_scales, scale_spread, slot_arm_length

    topology = mofgen.topologies[plan["net"]]
    ordering = list(topology.mappings)
    per_position: list[list[tuple[float, str]]] = []
    for position, names in zip(plan["slot_order"], plan["options"], strict=True):
        slot_type = ordering[position]
        arm = slot_arm_length(slot_type)
        options: list[tuple[float, str]] = []
        for name in names:
            fragment = mofgen.sbu[name]
            dummy_idx = [
                i for i, s in enumerate(fragment.atoms) if s.specie.symbol == "X"
            ]
            if not dummy_idx:
                continue
            dummies = np.asarray(fragment.atoms.cart_coords)[dummy_idx]
            length = float(
                np.linalg.norm(dummies - dummies.mean(axis=0), axis=1).mean()
            )
            options.append((length / arm if arm > 1e-9 else 0.0, name))
        if not options:
            return []
        per_position.append(options)

    every_ratio = [ratio for options in per_position for ratio, _ in options]
    grid = np.linspace(min(every_ratio), max(every_ratio), max(targets, 2))
    scored: dict[tuple[str, ...], float] = {}
    for target in grid:
        choice = tuple(
            min(options, key=lambda pair: abs(pair[0] - target))[1]
            for options in per_position
        )
        if choice in scored:
            continue
        resolved: dict = {}
        for position, name in zip(plan["slot_order"], choice, strict=True):
            for index in topology.mappings[ordering[position]]:
                resolved[index] = mofgen.sbu[name]
        spread = scale_spread(edge_scales(topology, resolved))
        scored[choice] = float("inf") if spread is None else spread

    # Ranked on spread alone. NOTE the known blind spot: spread is
    # scale-INVARIANT (a ratio of ratios), so uniformly *tiny* units
    # score a perfect 1.0 and give a trivially open structure - 27 of
    # this arm's 47 usable builds are bare metal nets like
    # node_Cu_2X + node_Fe_6X -> "FeCu3". Filtering those is the
    # caller's job (`is_genuine_framework`), NOT a tie-break here:
    # preferring bulkier units among equally-balanced choices was tried
    # and made everything worse (genuine 20 -> 16, clash-free 51 -> 38),
    # because it drags in high-spread picks the rounding hid.
    ranked = sorted(scored.items(), key=lambda item: item[1])
    return [choice for choice, _ in ranked[:per_net]]


def sample_combinations(
    plan: dict, per_net: int, rng: np.random.Generator, require: dict | None = None
) -> list[tuple[str, ...]]:
    """Distinct SBU choices for one net, sampled rather than enumerated.

    The full product is astronomical on multinodal nets (one 49-slot-type
    net offers 2e44 combinations), so a seeded sample is the only honest
    enumeration. ``require`` optionally demands that the choice draw from
    each of several named pools at least once - how the mixed arm gets
    both vocabularies into one build instead of trusting chance.
    """
    options = plan["options"]
    seen: set[tuple[str, ...]] = set()
    picks: list[tuple[str, ...]] = []
    for _ in range(per_net * 40):
        if len(picks) >= per_net:
            break
        choice = tuple(o[rng.integers(len(o))] for o in options)
        if choice in seen:
            continue
        if require is not None and not all(
            any(name in pool for name in choice) for pool in require.values()
        ):
            continue
        seen.add(choice)
        picks.append(choice)
    return picks


# --------------------------------------------------------------------
# one build
# --------------------------------------------------------------------


def attempt(
    mofgen: Autografs,
    net: str,
    choice: tuple[str, ...],
    slot_order: list[int],
    max_rmsd: float,
    empty_slot_types: list[int] | None = None,
    realized: set[str] | None = None,
    relax_embedding: bool = False,
    write_usable: Path | None = None,
) -> dict:
    """Build one (net, mapping) and report every measurable thing about it."""
    record: dict = {
        "net": net,
        "sbus": list(choice),
        # the blueprint positions `sbus` belongs to: without it a record
        # cannot be rebuilt without re-running the (expensive) sieve
        "slot_order": list(slot_order),
        "empty_slot_types": list(empty_slot_types or []),
        "outcome": None,
        "seconds": 0.0,
    }
    start = time.perf_counter()
    topology = mofgen.topologies[net]
    ordering = list(topology.mappings)
    mappings: dict = {
        ordering[position]: name
        for position, name in zip(slot_order, choice, strict=True)
    }
    for position in empty_slot_types or []:
        mappings[ordering[position]] = None
    try:
        framework = mofgen.build(
            topology, mappings, max_rmsd=max_rmsd, relax_embedding=relax_embedding
        )
    except AlignmentError as exc:
        record["outcome"] = "shape_gate"
        record["error"] = f"{type(exc).__name__}: {exc}"[:200]
        match = _WORST_RMSD.search(str(exc))
        if match:
            record["worst_rmsd"] = float(match.group(1))
        record["seconds"] = time.perf_counter() - start
        return record
    except AutografsError as exc:
        record["outcome"] = "refused"
        record["error"] = f"{type(exc).__name__}: {exc}"[:200]
        record["seconds"] = time.perf_counter() - start
        return record
    except Exception as exc:  # noqa: BLE001 - a build bug is data
        record["outcome"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"[:200]
        record["traceback"] = traceback.format_exc(limit=4)
        record["seconds"] = time.perf_counter() - start
        return record
    built = framework.structure
    residual = bond_residuals(framework)
    record["outcome"] = "built"
    record["formula"] = built.composition.reduced_formula
    record["n_atoms"] = len(built)
    record["cell"] = [round(x, 3) for x in built.lattice.abc]
    record["volume_per_atom"] = round(built.volume / len(built), 3)
    record["min_contact"] = round(framework.min_contact(), 3)
    record["bond_residual"] = residual
    # bond_residuals returns {} when no inter-unit bond is measurable -
    # every bond internal to one slot, or every element absent from the
    # Cordero table. Emptying the 2-connected slots makes that reachable
    # on real nets, so closure is UNKNOWN rather than satisfied, and the
    # build must not be counted as closed on a measurement that does not
    # exist.
    worst_bond = residual.get("max")
    record["closure_measured"] = worst_bond is not None
    record["closed"] = worst_bond is not None and worst_bond <= CLOSURE_TOLERANCE
    if relax_embedding:
        # which candidate the closure guard kept, and the EFFECTIVE
        # freedom after empty-slot pinning - without it a build where
        # relaxation never engaged is indistinguishable from one where
        # the guard rejected it, which is exactly the misattribution
        # the corpus arm made
        record["relaxation"] = framework.graph.graph.get("relaxation")
    record["clash_free"] = record["min_contact"] >= CONTACT_FLOOR
    components, free_molecules = _connectivity(framework)
    record["n_components"] = components
    record["n_free_molecules"] = free_molecules
    record["connected"] = free_molecules == 0
    record["usable"] = bool(
        record["closed"] and record["clash_free"] and record["connected"]
    )
    # ALWAYS fingerprint: it reads the framework's own per-slot
    # provenance and costs nothing, and the novelty lookup happens in
    # the parent (workers do not carry the harvest's source
    # assemblies). Computing it only when `realized` was passed meant
    # workers - which are never passed it - produced no fingerprint at
    # all, so a freshly-harvested sweep still reported novelty as
    # unmeasured.
    from autografs.fingerprint import from_framework

    fingerprint = from_framework(framework, nets=(net,))
    record["fingerprint"] = str(fingerprint)
    if realized is not None:
        record["novel"] = record["fingerprint"] not in realized
    if write_usable is not None and record["usable"]:
        # a count of hypothetical materials is only worth as much as the
        # structures behind it. CIF for inspection, framework JSON
        # because CIF loses the bond graph and UFF4MOF types that any
        # downstream relaxation needs.
        # hashlib, not hash(): str hashing is salted per process, so
        # workers would disagree and reruns would not reproduce names
        digest = hashlib.sha256(record["fingerprint"].encode()).hexdigest()[:10]
        stem = f"{net}-{digest}"
        framework.write_cif(str(write_usable / f"{stem}.cif"))
        framework.save(str(write_usable / f"{stem}.json"))
        record["written"] = stem
    record["seconds"] = time.perf_counter() - start
    return record


# --------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------


def _init_worker(
    xyzfile: str,
    topofile: str | None,
    max_rmsd: float,
    relax_embedding: bool = False,
    write_usable: str | None = None,
) -> None:
    _WORKER["mofgen"] = Autografs(xyzfile=xyzfile, topofile=topofile)
    _WORKER["max_rmsd"] = max_rmsd
    _WORKER["relax_embedding"] = relax_embedding
    _WORKER["write_usable"] = Path(write_usable) if write_usable else None


def _run_worker(task: tuple) -> dict:
    net, choice, slot_order, empty_slot_types, descriptors, arm = task
    try:
        record = attempt(
            _WORKER["mofgen"],
            net,
            tuple(choice),
            list(slot_order),
            _WORKER["max_rmsd"],
            empty_slot_types=list(empty_slot_types),
            relax_embedding=_WORKER.get("relax_embedding", False),
            write_usable=_WORKER.get("write_usable"),
        )
    except Exception as error:  # noqa: BLE001 - one bad net must not kill a sweep
        record = {
            "net": net,
            "sbus": list(choice),
            "outcome": "driver_error",
            "error": f"{type(error).__name__}: {error}"[:200],
            "traceback": traceback.format_exc(limit=3),
        }
    record.update(descriptors)
    record["arm"] = arm
    return record


def summarize(records: list[dict]) -> dict:
    """The funnel, plus the distributions that explain it."""
    built = [r for r in records if r["outcome"] == "built"]

    def _stats(values: list[float]) -> dict | None:
        if not values:
            return None
        values = sorted(values)
        return {
            "n": len(values),
            "median": round(statistics.median(values), 4),
            "p10": round(values[max(0, int(0.10 * len(values)) - 1)], 4),
            "p90": round(values[min(len(values) - 1, int(0.90 * len(values)))], 4),
        }

    return {
        "attempted": len(records),
        "outcomes": dict(Counter(r["outcome"] for r in records)),
        "built": len(built),
        "closed": sum(1 for r in built if r.get("closed")),
        "clash_free": sum(1 for r in built if r.get("clash_free")),
        "with_free_molecules": sum(1 for r in built if r.get("n_free_molecules")),
        "usable": sum(1 for r in built if r.get("usable")),
        # novelty needs the harvest's source fingerprints, which only
        # exist when this run did the harvesting; reusing a vocabulary
        # with --xyz leaves it UNMEASURED, and reporting that as 0 would
        # read as "nothing new was made"
        "novel_usable": (
            sum(1 for r in built if r.get("usable") and r.get("novel"))
            if any("novel" in r for r in built)
            else None
        ),
        "worst_bond": _stats(_worst_bonds(built)),
        "unmeasurable_closure": sum(
            1 for r in built if not r.get("closure_measured", True)
        ),
        "min_contact": _stats([r["min_contact"] for r in built]),
        "shape_gate_rmsd": _stats(
            [r["worst_rmsd"] for r in records if "worst_rmsd" in r]
        ),
        "seconds": _stats([r.get("seconds", 0.0) for r in records]),
        "by_freedom": _by_freedom(built, _stats),
        "novelty": _novelty(built),
    }


def _connectivity(framework) -> tuple[int, int]:
    """(bond-graph components, of which are free molecules).

    A component with no boundary-crossing bond is 0-periodic: a
    molecule floating in the cell, not part of the framework. Several
    *periodic* components are legitimate - that is interpenetration -
    so only the 0-periodic ones disqualify a build. This is the same
    convention deconstruct.py uses when it strips guests.

    Emptying 2-connected slots can strand a unit this way: two gra
    builds passed closure and contact while carrying a free Ni2 dimer
    and free CO2 fragments, and lammps-interface refused them as "free
    molecules" - the force field caught what the geometric gates did not.
    """
    import networkx as nx

    graph = framework.graph
    cell = np.asarray(graph.graph["cell"], dtype=float)
    inverse = np.linalg.inv(cell)
    components = list(nx.connected_components(graph))
    free = 0
    for component in components:
        crossing = False
        for node_a, node_b in graph.subgraph(component).edges():
            delta = np.asarray(graph.nodes[node_a]["coord"], float) - np.asarray(
                graph.nodes[node_b]["coord"], float
            )
            if np.any(np.round(delta @ inverse)):
                crossing = True
                break
        if not crossing:
            free += 1
    return len(components), free


def is_genuine_framework(record: dict) -> bool:
    """Did this build actually place an organic linker on a node?

    The guard against a degenerate "success". Emptying every
    2-connected slot on a net whose other slots are metal nodes leaves a
    BARE METAL net - a handful of atoms, nothing non-bonded within the
    contact cutoff, so it scores as closed and clash-free and enters a
    usable count as a hypothetical MOF. Measured: of 58 usable novel
    builds under --empty-2c, 54 were exactly this. A framework worth
    counting places at least one linker and more than one distinct unit.
    """
    sbus = record.get("sbus") or []
    has_linker = any(str(s).startswith("linker_") for s in sbus)
    return has_linker and len(set(sbus)) >= 2


def _novelty(built: list[dict]) -> dict | None:
    """Usable builds split by whether the corpus already realizes them.

    A build that reproduces a source structure's assembly is a round
    trip, not a generation, so the two must never be pooled into one
    "generated" count. ``distinct_usable_novel`` deduplicates on the
    fingerprint itself: several sampled mappings can land on the same
    (net, blocks) assembly, and that is one hypothetical material.

    ``genuine_usable_novel`` applies :func:`is_genuine_framework` on
    top, and is the number to quote: it is the one that does not count
    bare metal nets as generated materials.
    """
    scored = [r for r in built if "novel" in r]
    if not scored:
        return None
    usable_novel = [r for r in scored if r.get("usable") and r["novel"]]
    genuine = [r for r in usable_novel if is_genuine_framework(r)]
    return {
        "scored": len(scored),
        "novel": sum(1 for r in scored if r["novel"]),
        "usable_novel": len(usable_novel),
        "usable_known": sum(1 for r in scored if r.get("usable") and not r["novel"]),
        "distinct_usable_novel": len(
            {r["fingerprint"] for r in usable_novel if "fingerprint" in r}
        ),
        "distinct_nets_usable_novel": len(
            {r["net"] for r in usable_novel if "net" in r}
        ),
        "genuine_usable_novel": len(genuine),
        "degenerate_usable_novel": len(usable_novel) - len(genuine),
        "distinct_genuine_usable_novel": len(
            {r["fingerprint"] for r in genuine if "fingerprint" in r}
        ),
        "distinct_nets_genuine": len({r["net"] for r in genuine if "net" in r}),
    }


def _worst_bonds(built: list[dict]) -> list[float]:
    """Worst-bond deviations, skipping builds where none was measurable."""
    return [
        r["bond_residual"]["max"]
        for r in built
        if "max" in (r.get("bond_residual") or {})
    ]


def _freedom_bin(n_free: int | None) -> str:
    """The strata the embedding work already established.

    Not equal-width: the interesting boundary is 0 (fully pinned by site
    symmetry - the regime the shipped nets that build correctly live in)
    against anything else, and the corpus measurement of embedding
    relaxation found 1 behaving differently from 2+.
    """
    if n_free is None:
        return "unknown"
    if n_free == 0:
        return "0 (pinned)"
    if n_free == 1:
        return "1"
    if n_free <= 5:
        return "2-5"
    return ">5"


def _by_freedom(built: list[dict], _stats) -> dict:
    """Build quality stratified by the blueprint's own symmetry freedom."""
    groups: dict[str, list[dict]] = {}
    for record in built:
        groups.setdefault(_freedom_bin(record.get("n_free")), []).append(record)
    out = {}
    for key in sorted(groups):
        group = groups[key]
        out[key] = {
            "built": len(group),
            "closed": sum(1 for r in group if r.get("closed")),
            "clash_free": sum(1 for r in group if r.get("clash_free")),
            "worst_bond": _stats(_worst_bonds(group)),
            "min_contact": _stats([r["min_contact"] for r in group]),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", help="directory, glob, manifest, or single CIF")
    parser.add_argument("-o", "--output", default="recombine.json")
    parser.add_argument("--harvest-limit", type=int, default=60)
    parser.add_argument("--nets", type=int, default=200, help="coverable nets to use")
    parser.add_argument("--per-net", type=int, default=3)
    parser.add_argument("--max-rmsd", type=float, default=0.5)
    parser.add_argument(
        "--arm",
        default="harvested",
        choices=("harvested", "mixed", "library"),
        help="which vocabulary fills the slots",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--xyz", default=None, help="reuse a harvested vocabulary")
    parser.add_argument("--topofile", default=None)
    parser.add_argument(
        "--empty-2c",
        action="store_true",
        help="leave 2-connected slots empty (#179) instead of filling them",
    )
    parser.add_argument(
        "--relax-embedding",
        action="store_true",
        help="free the symmetry-allowed slot displacements (#174)",
    )
    parser.add_argument(
        "--freedom-cache",
        default=None,
        help="JSON file caching n_free per net; n_free depends only on the "
        "topology, so a sweep should never recompute it",
    )
    parser.add_argument(
        "--selection",
        default="random",
        choices=("random", "balanced"),
        help="how the SBUs are chosen: seeded sampling, or proportioned to the net",
    )
    parser.add_argument(
        "--write-usable",
        default=None,
        help="directory to write CIF + framework JSON for every usable build",
    )
    parser.add_argument(
        "--max-free",
        type=int,
        default=None,
        help="keep only blueprints with at most this many free displacements",
    )
    parser.add_argument(
        "--net-list",
        default=None,
        help="comma-separated net names, or a file of them, instead of a sample",
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    usable_dir = Path(args.write_usable) if args.write_usable else None
    if usable_dir:
        usable_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- the vocabulary -------------------------------------------
    xyz_path = out.with_suffix(".xyz") if args.xyz is None else Path(args.xyz)
    realized: set[str] = set()
    stock_report: dict = {}
    if args.xyz is None:
        corpus = _collect(args.corpus)[: args.harvest_limit]
        print(f"harvesting from {len(corpus)} structures")
        mofgen = Autografs(topofile=args.topofile) if args.topofile else Autografs()
        stock = build_stock(mofgen, corpus, n_jobs=args.n_jobs, topofile=args.topofile)
        units = building_units(stock)
        from autografs.deconstruct import write_fragments_xyz

        write_fragments_xyz(units, xyz_path)
        stock_report = {
            "n_sources_attempted": len(corpus),
            "n_sources_used": stock["n_sources"],
            "n_units": len(units),
            "arities": dict(
                sorted(
                    Counter(
                        len(f.atoms.indices_from_symbol("X")) for f in units.values()
                    ).items()
                )
            ),
            "failures": len(stock["failures"]),
        }
        # the source assemblies travel WITH the vocabulary: novelty is a
        # lookup against them, and a reused .xyz without them can only
        # report novelty as unmeasured. They are strings because that is
        # what the comparison uses (workers cannot return the frozen
        # dataclass without the same vocabulary loaded).
        realized = {str(f) for f in stock["realized"].values()}
        _fingerprint_path(xyz_path).write_text(json.dumps(sorted(realized), indent=1))
        print(
            f"harvested {len(units)} building units from "
            f"{stock['n_sources']}/{len(corpus)} structures -> {xyz_path}"
        )
    else:
        print(f"reusing vocabulary {xyz_path}")
        sidecar = _fingerprint_path(xyz_path)
        if sidecar.exists():
            realized = set(json.loads(sidecar.read_text(encoding="utf-8")))
            print(f"  {len(realized)} source assemblies for the novelty lookup")
        else:
            print(f"  [!] no {sidecar.name}: novelty will be reported as unmeasured")

    # ---- the acceptance surface -----------------------------------
    mofgen = Autografs(xyzfile=str(xyz_path), topofile=args.topofile)
    harvested = sorted(
        name for name in mofgen.sbu if name.startswith(("node_", "linker_"))
    )
    library = sorted(set(mofgen.sbu) - set(harvested))
    subset = {
        "harvested": harvested,
        "library": library,
        "mixed": sorted(mofgen.sbu),
    }[args.arm]
    print(f"arm {args.arm!r}: {len(subset)} SBUs in play")
    only = None
    if args.net_list:
        candidate = Path(args.net_list)
        text = (
            candidate.read_text(encoding="utf-8")
            if candidate.exists()
            else args.net_list
        )
        only = [
            piece.strip()
            for piece in text.replace("\n", ",").split(",")
            if piece.strip()
        ]
    freedom: dict = {}
    freedom_path = Path(args.freedom_cache) if args.freedom_cache else None
    if freedom_path and freedom_path.exists():
        freedom = json.loads(freedom_path.read_text(encoding="utf-8"))
        print(f"freedom cache: {len(freedom)} nets")
    plans = coverable_nets(
        mofgen,
        subset,
        limit=None if only else args.nets,
        rng=rng,
        empty_2c=args.empty_2c,
        only=only,
        max_free=args.max_free,
        freedom_cache=freedom,
    )
    if freedom_path:
        freedom_path.write_text(json.dumps(freedom, indent=0, sort_keys=True))
    print(f"{len(plans)} fully coverable nets")
    if not plans:
        raise SystemExit("no net is coverable by this vocabulary")

    require = None
    if args.arm == "mixed":
        require = {"harvested": set(harvested), "library": set(library)}
    tasks = []
    for plan in plans:
        descriptors = {
            key: plan[key]
            for key in ("n_free", "n_slot_types", "n_slots", "n_slot_types_total")
        }
        descriptors["n_empty_slot_types"] = len(plan["empty_slot_types"])
        if args.selection == "balanced":
            choices = balanced_combinations(mofgen, plan, args.per_net)
        else:
            choices = sample_combinations(plan, args.per_net, rng, require=require)
        for choice in choices:
            tasks.append(
                (
                    plan["net"],
                    choice,
                    plan["slot_order"],
                    plan["empty_slot_types"],
                    descriptors,
                    args.arm,
                )
            )
    print(f"{len(tasks)} builds queued")

    # ---- build -----------------------------------------------------
    records: list[dict] = []
    started = time.perf_counter()
    if args.n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=args.n_jobs,
            initializer=_init_worker,
            initargs=(
                str(xyz_path),
                args.topofile,
                args.max_rmsd,
                args.relax_embedding,
                str(usable_dir) if usable_dir else None,
            ),
        ) as pool:
            for index, record in enumerate(
                pool.map(_run_worker, tasks, chunksize=2), 1
            ):
                records.append(record)
                _report(index, len(tasks), record)
    else:
        _init_worker(
            str(xyz_path),
            args.topofile,
            args.max_rmsd,
            args.relax_embedding,
            str(usable_dir) if usable_dir else None,
        )
        for index, task in enumerate(tasks, 1):
            record = _run_worker(task)
            records.append(record)
            _report(index, len(tasks), record)

    # novelty needs the corpus fingerprints, which only the parent has
    if realized:
        for record in records:
            if "fingerprint" in record:
                record["novel"] = record["fingerprint"] not in realized

    payload = {
        "benchmark": "recombine",
        "arm": args.arm,
        "max_rmsd": args.max_rmsd,
        "relax_embedding": args.relax_embedding,
        "max_free": args.max_free,
        "selection": args.selection,
        "empty_2c": args.empty_2c,
        "closure_tolerance": CLOSURE_TOLERANCE,
        "contact_floor": CONTACT_FLOOR,
        "seed": args.seed,
        "stock": stock_report,
        "n_coverable_nets": len(plans),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "summary": summarize(records),
        "records": records,
    }
    out.write_text(json.dumps(payload, indent=1, default=str))
    print(f"\n-> {out}")
    for key, value in payload["summary"].items():
        print(f"  {key:<18} {value}")


def _report(index: int, total: int, record: dict) -> None:
    tag = record["outcome"]
    if tag == "built":
        worst = (record.get("bond_residual") or {}).get("max")
        bond = f"{worst:.2f}" if worst is not None else "n/a"
        tag = (
            f"built {record['formula']} contact {record['min_contact']} "
            f"worst-bond {bond}"
            f"{' USABLE' if record.get('usable') else ''}"
        )
    print(f"[{index}/{total}] {record['net']}: {tag}", flush=True)


if __name__ == "__main__":
    main()
