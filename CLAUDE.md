# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AuToGraFS generates Metal-Organic Frameworks (MOFs), COFs and other periodic crystalline structures by mapping molecular building blocks (SBUs) onto topological blueprints. Python >= 3.11. Built on pymatgen, ASE, networkx, scipy. README.md is the single documentation source and describes the current API accurately.

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
- **`topology_io.py`** — versioned JSON (de)serialization of topology libraries; `LazyTopologyLibrary` materializes entries on first access (don't `dict.update()` it — that materializes everything). Legacy dill pickles still load with a warning.
- **`alignment.py`** — the numpy build core. Directional matching (Hungarian + Kabsch, proper rotations only — chirality preserved) and the cell objective (RMS deviation of anchor-pair distances from Cordero covalent bond lengths, over the crystal system's free parameters only). Everything precomputed into a `BuildPlan`; no pymatgen objects inside the optimization loop.
- **`builder.py`** — `Autografs`: entry point. Loads SBUs (`data/defaults.xyz` + `data/pormake.xyz` + user XYZ) and topologies (`data/topologies.json.gz`). `build(topology, mappings, max_rmsd=..., min_distance=...)` → `Framework`; gates raise `AlignmentError`/`OverlapError` instead of returning bad structures. `build_all()` enumerates combinations (seeded sampling above `max_per_topology`, `n_jobs` worker processes). Module-level `build_framework()` is the library-independent core used by workers.
- **`framework.py`** — `Framework`: the result. The networkx bond graph is the source of truth (symbols, coords, bond orders, UFF4MOF types, tags); views/exports: `structure`, `to_ase`, `write_cif`, `to_gulp`, `min_contact`, `stack` (2D layer → COF crystal, AA/AB/serrated/staggered), `relax` (in-process LAMMPS/UFF4MOF, optional `[relax]` extra).
- **`relax.py`** — LAMMPS relaxation backend behind `Framework.relax`.
- **`plane_groups.py`** — the 17 plane groups, used for 2D layer nets.
- **`cgd.py`** — CGD parser and `autografs-topologies` CLI (RCSR download, spacegroup expansion, orbit extraction).
- **`cli.py`** — `autografs` interactive wizard (questionary + rich). Deliberately has no non-interactive build flags; scripted use is the Python API.
- **`utils.py`** — XYZ parsing, UFF4MOF atom typing (constants in `data/uff4mof.py`), Fragment→graph conversions, GULP export.
- **`exceptions.py`** — `AutografsError` hierarchy: `AlignmentError`, `OverlapError`, `StackingError`, `RelaxationError`, `TopologyExtractionError`.

Pipeline: libraries → geometric compatibility sieve (`list_building_units(sieve=...)` / `list_topologies(sieve=...)`) → slot-type-to-SBU mappings → per-slot alignment + cell optimization → quality gates → `Framework`.

## Gotchas

- **Everything ships in the wheel**: `data/topologies.json.gz` (2686 RCSR nets) and both SBU files are bundled; no generation step is needed. Regeneration (only when updating the library): `autografs-topologies --use_rcsr -o topologies.json.gz`. `scripts/cgd2pkl.py` is a deprecated shim for that command.
- Dummy atoms use pymatgen's `"X"` species convention throughout. Slot/SBU compatibility is decided by dummy count + arm-direction match, NOT point group (symbols are metadata only).
- In `cgd.py`, `# EDGE_CENTER` lines in RCSR files look commented out but are deliberately uncommented by the parser (they create the 2-connected slots). Preserve that behavior when refactoring.
- `Autografs._validate_mappings` deep-copies fragments: alignment mutates them in place, and aliasing would corrupt the SBU library. Keep that invariant.
- Formatting/linting is ruff (not black/isort/flake8); mypy targets 3.12 even though the runtime floor is 3.11 (numpy stubs use PEP 695).
