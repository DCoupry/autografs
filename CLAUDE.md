# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AuToGraFS generates Metal-Organic Frameworks (MOFs), COFs and other periodic crystalline structures by mapping molecular building blocks (SBUs) onto topological blueprints, and runs the pipeline in reverse (deconstruction). Python >= 3.11. Built on pymatgen, ASE, networkx, scipy. Docs: `README.md` is the overview + FAQ; feature guides live in `docs/*.md` (building, cofs-and-stacking, editing, deconstruction, cli, extending, internals, coverage). Keep them accurate to the current API — when changing behaviour, update the relevant `docs/` page (and the README FAQ/Why bullets if the change is user-facing).

## Commands

```bash
pip install -e ".[dev]"          # dev install

pytest                            # run tests (slow tests skipped by default)
pytest tests/test_builder.py      # single file
pytest tests/test_builder.py -k test_name   # single test
pytest -m slow                    # run only the slow tests

ruff check src tests scripts     # lint
ruff format src tests scripts    # format
mypy src/autografs               # type check
```

`conftest.py` auto-skips `@pytest.mark.slow` tests unless `-m` is passed explicitly.

Console entry points: `autografs` (interactive wizard, `cli.py`) and `autografs-topologies` (CGD → JSON topology converter, `cgd.py`).

## Architecture

Modules in `src/autografs/`, in pipeline order:

- **`fragment.py`** — `Fragment`: a pymatgen `Molecule` plus dummy-atom (`"X"`) connection points. Represents both SBUs and topology slots. Compatibility (`has_compatible_symmetry`) is geometric: directional RMSD between unit arm vectors (threshold `COMPATIBILITY_MAX_RMSD = 0.35`), not point-group symbols. Also pre-build editing: `rotate`, `flip`, `functionalize`.
- **`topology.py`** — `Topology`: a pymatgen `Lattice` plus an array of `Fragment` slots. `mappings` groups slots by crystallographic orbit (`equivalence_class`); one SBU choice covers all slots of an orbit. `is_2d` marks layer nets (c is frozen slab padding).
- **`topology_io.py`** — versioned JSON (de)serialization of topology libraries; `LazyTopologyLibrary` materializes entries on first access (don't `dict.update()` it — that materializes everything). Lookup-only aliases via `attach_aliases` (the builder attaches `data/iza_aliases.json`, 55 IZA zeolite codes → lowercase RCSR names; aliases resolve in `[]`/`in` but never appear in iteration). Legacy dill pickles still load with a warning.
- **`alignment.py`** — the numpy build core. Directional matching (Hungarian + Kabsch, proper rotations only — chirality preserved) and the cell objective (RMS deviation of anchor-pair distances from Cordero covalent bond lengths, over the crystal system's free parameters only). Everything precomputed into a `BuildPlan`; no pymatgen objects inside the optimization loop.
- **`builder.py`** — `Autografs`: entry point. Loads SBUs (`data/defaults.xyz` + `data/pormake.xyz` + user XYZ) and topologies (`data/topologies.json.gz`). `build(topology, mappings, max_rmsd=..., min_distance=...)` → `Framework`; gates raise `AlignmentError`/`OverlapError` instead of returning bad structures. `build_all()` enumerates combinations (seeded sampling above `max_per_topology`, `n_jobs` worker processes). Module-level `build_framework()` is the library-independent core used by workers.
- **`framework.py`** — `Framework`: the result. The networkx bond graph is the source of truth (symbols, coords, bond orders, UFF4MOF types, tags, per-atom `slot`/`sbu` provenance); views/exports: `structure`, `to_ase`, `write_cif`, `to_gulp`, `min_contact`, `stack` (2D layer → COF crystal, AA/AB/serrated/staggered), `relax` (in-process LAMMPS/UFF4MOF, optional `[relax]` extra). Post-build editing methods (`slots`, `supercell`, `defects`, `rotate`, `flip`, `functionalize`) are thin delegates into `editing.py`.
- **`framework_io.py`** — versioned JSON (de)serialization of Framework objects (`Framework.save`/`load`): persists what CIF loses — the bond graph with orders, UFF4MOF types, anchor tags, and slot/SBU provenance — so a framework saved in one session stays editable in another.
- **`porosity.py`** — geometric porosity descriptors behind `Framework.void_fraction`/`largest_cavity_diameter`/`pore_limiting_diameter`: a periodic vdW-surface distance grid; PLD by binary search over probe radius with a wrap-detecting union-find (a percolating channel = an open component connected to its own periodic image).
- **`net.py`** — labeled quotient graphs (one node per slot/unit, edges carry periodic-image voltages). Verification (`verify_net`: exact edge-multiset comparison, possible because built atoms record slot provenance) and identification (`identify_net`: coordination-sequence signatures, caps pruned / 2-c vertices contracted, matched two-tier — uncontracted first, so a net beats its edge-decorated derivatives; a degree-multiset prefilter reads `raw_items()` and scans the full library without materializing it).
- **`deconstruct.py`** — the inverse pipeline. CIF/Structure → EconNN bonds capped at `BOND_LENGTH_SLACK` × UFF radius sums (EconNN is adaptive and bonds isolated guests to anything) → 0-periodic components removed as guests → clustering → dummy-capped `Fragment`s (X at cut-bond midpoints) → quotient graph (built directly from the cut list: one voltage-labeled edge per cut bond, so parallel bonds through different periodic images all survive; gauge shared with `autografs.net`) → `identify_net`. Guest removal iterates until the bond graph is guest-free, so the fold and the subframework split come from one final graph. Clustering dispatches on metal presence: **metal-oxo** for MOFs (metals + C-free metal-bound atoms + carboxylate/sulfonate/phosphonate groups — cuts at carboxylate-C↔backbone-C and metal↔donor bonds), **branch-point** for metal-free COFs (`_organic_units`: ring systems + non-ring atoms collapse to super-vertices via `_ring_bonds` — a bounded zero-voltage cycle search on the periodic graph, no global unwrap — and external-degree ≥3/2/1 → node/linker/cap; single-node convention). Rod / 1-periodic units raise `DeconstructionError`. Interpenetrated structures split into per-periodic-component quotient graphs, each identified independently (`subframework_nets`); `net_candidates` is their consensus, `n_periodic_components` the fold. Entry points: `Autografs.deconstruct(source)`, module-level `deconstruct(source, topologies=...)`.
- **`harvest.py`** — batch SBU harvesting: runs `deconstruct` over many structures (directory/glob/iterable) and merges building units into one cross-structure-deduplicated `Fragment` library (`merge_fragment` from `deconstruct.py`), keeping per-fragment provenance, per-source net candidates, and a non-aborting failure report. `HarvestResult.building_units` excludes monotopic `cap` fragments (bound solvent). Entry: `Autografs.harvest(sources)`.
- **`editing.py`** — post-build editing on the bond graph: exact supercells (per-edge periodic image shift recovered by min-image rounding of unwrapped coords), statistical defects (seeded SBU removal + capping of dangling anchors), placed-SBU rotation/flip (isometries that fix the anchor atoms), framework functionalization (terminal-atom replacement by X-tagged groups). All ops return new Frameworks.
- **`relax.py`** — LAMMPS relaxation backend behind `Framework.relax`.
- **`plane_groups.py`** — the 17 plane groups, used for 2D layer nets.
- **`cgd.py`** — CGD parser and `autografs-topologies` CLI (RCSR download, spacegroup expansion, orbit extraction).
- **`cli.py`** — `autografs` interactive wizard (questionary + rich). Deliberately has no non-interactive build flags; scripted use is the Python API.
- **`utils.py`** — XYZ parsing, UFF4MOF atom typing (constants in `data/uff4mof.py`), Fragment→graph conversions, GULP export.
- **`exceptions.py`** — `AutografsError` hierarchy: `AlignmentError`, `OverlapError`, `StackingError`, `RelaxationError`, `TopologyExtractionError`, `NetMismatchError`, `DeconstructionError`.

Pipeline: libraries → geometric compatibility sieve (`list_building_units(sieve=...)` / `list_topologies(sieve=...)`) → slot-type-to-SBU mappings → per-slot alignment + cell optimization → quality gates → `Framework`.

## Gotchas

- **Everything ships in the wheel**: `data/topologies.json.gz` (2686 RCSR nets) and both SBU files are bundled; no generation step is needed. Regeneration (only when updating the library): `autografs-topologies --use_rcsr -o topologies.json.gz`. `scripts/cgd2pkl.py` is a deprecated shim for that command.
- Dummy atoms use pymatgen's `"X"` species convention throughout. Slot/SBU compatibility is decided by dummy count + arm-direction match, NOT point group (symbols are metadata only).
- In `cgd.py`, `# EDGE_CENTER` lines in RCSR files look commented out but are deliberately uncommented by the parser (they create the 2-connected slots). Preserve that behavior when refactoring.
- `Autografs._validate_mappings` deep-copies fragments: alignment mutates them in place, and aliasing would corrupt the SBU library. Keep that invariant.
- Framework node ids must stay contiguous 0..n-1: `min_contact` and relax index pymatgen Structure sites by sorted node id. Any graph-editing operation must relabel before returning (see `editing._relabel`). Graph node coords are unwrapped cartesian; a bond's periodic image is implicit and recovered by min-image rounding — editing relies on it.
- Formatting/linting is ruff (not black/isort/flake8); mypy targets 3.12 even though the runtime floor is 3.11 (numpy stubs use PEP 695).
