# COF / 2D layer nets — design plan

*The next major feature after the v3 overhaul (see `v3_plan.md`,
`progress.md`). Target: build 2D COFs (hcb, sql, kgm, hxl, ...) from the
~200 plane-group nets currently skipped, with stacking control. To be
done on a fresh branch after the `periodic-review` PR merges.*

## Why this is the next step

The conversion pipeline currently drops 461 nets for untranslatable
GROUP symbols. Measured breakdown on RCSRnets-2019-06-01.cgd:

- **200 are 2D plane groups** — the entire layer-net family, i.e. the
  COF workhorses: p6mm (46), c2mm (28), p4mm (18), p2gg (18), p2mg
  (16), p4gm (15), p2mm (15), p31m (10), p2 (9), p6 (7), pg (6), cm
  (5), p3m1 (4), p4 (1), pm (1), p1 (1).
- 66 are nonstandard monoclinic settings — Cmca (22, renamed Cmce in
  2002), I12/a1 (20), P121/n1 (15), I12/m1 (8), A12/n1 (1) — a bonus
  rescue in the same code area.

The CGD parser already half-supports 2D entries: a 3-parameter CELL is
padded to a 10 Å slab (`is_2d` branch in
`autografs.cgd.topology_from_string`) and node coordinates get z=0.
Only the GROUP symbol translation is missing.

## Part 1 — plane-group symbol support in `autografs.cgd`

Two families, different risk levels:

### Easy: hexagonal/square groups (101 nets)

Their rotation axis is perpendicular to the layer, which is the c-axis
in the standard 3D setting, so the extruded space group is a direct
symbol map with **no setting trap**:

| plane group | space group | number |
|---|---|---|
| p4    | P4    | 75  |
| p4mm  | P4mm  | 99  |
| p4gm  | P4bm  | 100 | <!-- the in-plane g extrudes to a b glide -->

| p3m1  | P3m1  | 156 |
| p31m  | P31m  | 157 |
| p6    | P6    | 168 |
| p6mm  | P6mm  | 183 |

### Careful: oblique/rectangular groups (99 nets)

p1, p2, pm, pg, cm, p2mm, p2mg, p2gg, c2mm. Their standard 3D settings
put the unique axis along **b**, but the layer's symmetry elements are
along/perpendicular to **c**. This is exactly the origin/setting trap
class that produced wrong dia nets (see the Fd-3m:2 story in
`progress.md`). Two options:

1. Map to c-unique setting symbols (P112, P11m, ...) where pymatgen's
   `SpaceGroup` supports them — verify support per group first.
2. **Recommended:** bypass `Structure.from_spacegroup` for 2D entries
   and expand orbits with explicit plane-group operators. There are
   only 17 plane groups, each with a handful of generators; a
   hand-written operator table is small, testable in isolation, and
   immune to setting conventions. Apply operators to (x, y), keep z.

Whichever path: **verify each group against a known net** (hcb must
give honeycomb, sql a square grid) before trusting the batch.

### Also in this area

- The `'# EDGE_CENTER'` uncommenting quirk applies to 2D entries too —
  keep it.
- Monoclinic bonus: try `SpaceGroup("Cmce")` for Cmca, and the
  alternate-setting symbols (P21/n etc.) pymatgen may support; else
  standard-setting transformation matrices. 66 nets.

## Part 2 — 2D awareness in Topology and the cell optimizer

- `Topology.is_2d: bool` (constructor param, default False), stored in
  the JSON format as an optional key — `.get()` on load keeps format
  version 1 backward compatible.
- `CellParametrization` gains a layer mode when `is_2d`: **c is frozen**
  during optimization. No dummy pairs cross the slab direction, so the
  pair-coincidence objective is flat in c and Nelder-Mead would drift
  (already noted in `progress.md`). Free parameters become: hexagonal/
  square 1 (a), rectangular 2 (a, b), oblique 3 (a, b, gamma).
- Orbit equivalence classes: spglib on the padded slab works as-is
  (the slab's extra z-mirror can only refine orbits, never wrongly
  merge them).

## Part 3 — stacking

A single optimized layer is not a COF crystal; the interlayer geometry
is a *user* variable (dispersion-driven, not topology-driven).

API: post-hoc on the result object, so it works for any layered
framework:

```python
layer = mofgen.build(hcb, mappings)                  # c = pad value
cof   = layer.stack(mode="AA", interlayer=3.35)      # -> new Framework
cof   = layer.stack(mode="AB", interlayer=3.35)      # 2-layer cell,
                                                     # offset (1/3, 2/3)
cof   = layer.stack(mode="serrated", offset=(0.5, 0), interlayer=3.35)
```

Implementation notes:

- `stack()` returns a **new** Framework: replace c with `interlayer`
  (AA) or `2 * interlayer` with a second, offset copy of the layer
  (AB/serrated/staggered); intra-layer bonds duplicate per layer, no
  inter-layer bonds (COF layers are van-der-Waals stacked).
- Validate the input is actually layered: all atoms within a z-window,
  and no tag pairs crossing the c boundary. Raise a typed error
  otherwise.
- Default `interlayer=3.35` Å (graphite-like; typical COF range
  3.3–3.6).

## Part 4 — regenerate + golden tests

- Regenerate `topologies.json.gz`: expect ~2464 → ~2650+ usable nets.
- Extend `scripts/make_test_fixture.py` with hcb (p6mm) and sql (p4mm)
  verbatim RCSR entries.
- Golden test, the COF-1 prototype: **hcb + Boroxine_triangle +
  Benzene_linear** (both already in `defaults.xyz`). Assert: hexagonal
  cell (a = b, gamma = 120 to machine precision — the layer-mode
  parametrization guarantees it), correct atom count, layer planarity
  (all z within tolerance), and a stacked AA variant with c =
  interlayer.
- Stacking unit tests: atom count doubles for AB, offset applied,
  cell heights correct, non-layered input rejected.

## Part 5 — docs

README section "2D COFs" with the hcb quickstart; note the stacking
modes and the interlayer default.

## Verification checklist (from the v3 sessions' hard lessons)

- [x] hcb renders a honeycomb (visual + coordination check), not a
      setting-mangled net — the dia origin bug says *check, don't
      assume*. (tests/test_plane_groups.py + COF-1 golden build)
- [x] Every oblique/rectangular plane group verified against at least
      one net with known geometry. (scripts/verify_plane_groups.py:
      all 200 RCSR 2D nets, 16 groups, 0 endpoint misses)
- [x] Fixture regeneration byte-identical when rerun (determinism).
- [x] Full-library conversion stats recorded in `progress.md`
      (usable count, remaining failures by class). 2464 → 2686.
- [x] c stays exactly frozen through refine_cell=True.
      (golden test asserts lattice.c == pad exactly)
