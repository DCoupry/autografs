# v3 Plan — Progress Tracker

Working through [v3_plan.md](v3_plan.md) on branch `periodic-review`.
Environment: `conda activate autografs` (miniforge3, Python 3.11, pymatgen 2026.5.4).
Run tests: `python -m pytest tests/ -q` (pytest config in pyproject handles `src` path).

## Status legend
- [ ] not started  /  [~] in progress  /  [x] done (committed)

## Step 1 — Bug-fix batch (v3_plan §1 + §2)
- [~] Baseline: 3 stale test failures found (old `structure` module import, `X0+`
      dummy-species string, `flip()` now implemented); 19 skips from missing
      `topologies.pkl`. Fixing stale tests first so the suite is green.
- [ ] 1.1 Dead `is_subgroup` check in `Fragment.has_compatible_symmetry`
      (confirmed empirically: pymatgen 2026.5.4 `PointGroupAnalyzer` has no
      `is_subgroup`; AttributeError swallowed by blanket except)
- [ ] 1.2 `find_mmtypes` misaligned return list corrupting UFF types
- [ ] 1.3 Missing `.abs()` on angle_diff in `find_mmtypes`
- [ ] 1.4 `list_topologies(sieve, subset)` ValueError crash
- [ ] 1.5 Int-key aliasing corrupts SBU library; in-place mutation of caller's
      topology; `_align_slot` drops site_properties
- [ ] 1.6 `raise ("NoSymm")` + assert-as-control-flow in scripts/cgd2pkl.py
- [ ] 2.1 Determinism: seeded/removed `perturb()` in `_align_slot`
- [ ] 2.2 RMSD acceptance gate (`max_rmsd`, typed `AlignmentError`)

## Step 2 — Data format migration (v3_plan §3.2, §5.1)
- [ ] JSON topology format replacing dill pickle (name, cell, slots, tags,
      equivalence_class, crystal_system)
- [ ] Record equivalence classes from spacegroup orbits in cgd2pkl
- [ ] Commit tiny 3-net test fixture (pcu, srs, hcb) → unskips 19 tests

## Step 3 — Numpy alignment core (v3_plan §3.1 + §4.1)
- [ ] Geometric matching (linear_sum_assignment + Kabsch on arrays)
- [ ] Chirality handling (proper rotations only)
- [ ] Remove deepcopy hot loop from opt_fun

## Step 4 — Output layer (v3_plan §3.5)
- [ ] Framework result class (pymatgen Structure + bonds + mmtypes)
- [ ] write_cif / to_ase / to_gulp; README rewrite

## Step 5 — Symmetry-constrained cell optimization (v3_plan §3.3)
- [ ] Optimize only crystal-system free parameters

## Step 6 — Tooling sweep (v3_plan §5.3)
- [x] requires-python widened to >=3.11 (done by Damien, committed with groundwork)
- [ ] ruff replacing black+isort+flake8; pre-commit
- [ ] mypy in CI; CI matrix 3.11-3.13
- [ ] Declare tqdm dependency; console entry points

## Step 7 — Scale features (v3_plan §3.6, §4.4)
- [ ] Parallel + sampling build_all
- [ ] SBU library cache
- [ ] 2D stacking for COFs; IZA import; RCSR refresh

## Session log
- **2026-07-02**: Reviewed codebase, wrote v3_plan.md + CLAUDE.md. Confirmed
  bug 1.1 empirically in the conda env. Baseline test run: 3 stale failures,
  19 data-dependent skips. Started Step 1.
