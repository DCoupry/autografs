# v3 Plan — Progress Tracker

Working through [v3_plan.md](v3_plan.md) on branch `periodic-review`.
Environment: `conda activate autografs` (miniforge3, Python 3.11, pymatgen 2026.5.4).
Run tests: `python -m pytest tests/ -q` (pytest config in pyproject handles `src` path).

## Status legend
- [ ] not started  /  [~] in progress  /  [x] done (committed)

## Step 1 — Bug-fix batch (v3_plan §1 + §2) — COMPLETE
- [x] Baseline: fixed 3 stale tests (old `structure` module import, `X0+`
      dummy-species string, `flip()` now implemented). Suite green.
- [x] 1.1 Dead `is_subgroup` check removed; symbol-equality made explicit,
      exception swallowing removed (commit bd438c6)
- [x] 1.2+1.3 `find_mmtypes` rewritten pure, index-aligned, with `.abs()` on
      angle matching + arccos domain clip (commit 8d2993f); pandas global
      option removal + slice-copy fix (56474bf)
- [x] 1.4 `list_topologies` fixed — including discovery that the sieve was a
      complete NO-OP (truthy dict); subset crash + subset mutation also fixed
      (commit 325fdad)
- [x] 1.5 All aliasing/mutation fixed: per-slot fragment copies, build() no
      longer mutates topology/library/input dict, site_properties preserved.
      Bonus: pure-int mappings were always rejected as "unfilled"; coverage
      is now computed over slot indices (commit 31fe344)
- [x] 1.6 cgd2pkl: TopologyExtractionError replaces string-raise and assert;
      dropped nets now logged by name+reason (commit bfa5c07)
- [x] 2.1 Determinism: fixed-seed degeneracy breaking replaces unseeded
      perturb; hash-randomization-stable slot grouping + RMSE accumulation
      (commit 04e60fc)
- [x] 2.2 RMSD gate: `build(max_rmsd=...)` raises typed `AlignmentError`;
      new `autografs.exceptions` module (commit 04e60fc)

### Notes for next session
- `synthetic_mofgen` fixture in tests/test_builder.py builds an Autografs
  via `__new__` with synthetic libraries — full build() path now tested
  without topologies.pkl. Reuse it for new builder tests.
- pymatgen deprecation to handle in tooling pass:
  `MoleculeGraph.with_local_env_strategy` -> `from_local_env_strategy`
  (removal was slated for 2025-03; still present in 2026.5.4).
- Smoke-tested CGD parsing with an embedded pcu string (works end-to-end);
  reuse that pattern for parser tests and for generating the 3-net fixture.

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
- **2026-07-02 (cont.)**: Step 1 complete — all §1 bugs + §2 determinism/gate
  fixed, each with regression tests, across 8 commits (817e48b..04e60fc).
  Full suite green (123 tests, 19 data-dependent skips). Next: Step 2,
  the pkl→JSON topology format with orbit-derived equivalence classes,
  which unskips the remaining tests.
