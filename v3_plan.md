# AuToGraFS v3 — Review & Improvement Plan

*Full-codebase review performed 2026-07-02 (branch `periodic-review`). Covers correctness,
solid-state science, performance, and software engineering. File/line references are
against the tree at the time of review.*

**Goal:** generate topological materials (MOFs, COFs, zeolites, ...) at scale from the
RCSR topology database plus user-supplied building-unit databases.

**TL;DR:** The core design (dummy atoms + symmetry-gated slot matching + Hungarian
alignment) is sound and compact. But there are **five confirmed correctness bugs** (one
silently disables the subgroup-compatibility logic entirely), a **reproducibility
problem** (builds are stochastic with no seed), a **mutation/aliasing hazard** that can
corrupt the SBU library between builds, and a hot loop that deep-copies the entire
topology per optimizer iteration. On the science side, the biggest wins are: replacing
symbolic point-group matching with connection-vector RMSD, constraining cell optimization
to the net's crystal system, storing slot equivalence classes from the space-group orbits
instead of re-deriving them by fuzzy equality, and adding a real structure output layer
(CIF/pymatgen/ASE) instead of GULP-only.

---

## 1. Confirmed bugs

### 1.1 The subgroup compatibility check is dead code — `AttributeError` swallowed
`src/autografs/fragment.py:177` calls `self.symmetry.is_subgroup(other.symmetry)`, but
pymatgen's `PointGroupAnalyzer` has **no** `is_subgroup` method (verified against current
pymatgen docs — it lives on `pymatgen.symmetry.groups.PointGroup`, a different class).
Every call raises `AttributeError`, which the blanket `except Exception: return False`
at `fragment.py:178` silently eats. Net effect: for fragments with >3 dummies,
compatibility is **exact Schoenflies-symbol match only** — the documented "square SBU
fits rectangular slot" behavior never happens. This is also the strongest argument for
replacing broad `except Exception` with narrow, typed handling everywhere.

### 1.2 `find_mmtypes` returns a misaligned list, corrupting UFF atom types
In `src/autografs/utils.py:287-321`, `mmtypes.append(...)` only executes in the
fall-through branch — atoms resolved early (single candidate type, or unique coordination
match) `continue` without appending. The returned list is shorter than the atom count and
its indices don't correspond to atom indices. Then `utils.py:350-352` does
`for i, mmtype in enumerate(mmtypes): mg.molecule[i].properties["ufftype"] = mmtype`,
overwriting the (correct) types set in-place inside `find_mmtypes` with types belonging
to *different atoms*. Any structure with a mix of easy and ambiguous atoms gets scrambled
UFF types, which flows straight into GULP inputs.

### 1.3 Angle matching picks the *most negative* difference, not the closest
`src/autografs/utils.py:317-318`: `angle_diff = coordinations_compat.angle - angle` then
`sort_values` ascending — missing `.abs()`. For a measured angle of 120° with candidate
types at 90° and 120°, the diffs are −30 and 0; the sort picks −30, i.e. the *wrong* type
wins whenever any candidate angle is smaller than the measured one.

### 1.4 `list_topologies(sieve=..., subset=...)` crashes
`src/autografs/builder.py:382-395`: when `subset` is given, `full_list = subset`, but the
sieve loop iterates over **all** `self.topologies` and calls
`full_list.remove(topology.name)` — the first incompatible topology that isn't in the
subset raises `ValueError`.

### 1.5 Integer-keyed mappings alias and corrupt the SBU library
`src/autografs/builder.py:293-300`: `_validate_mappings` deep-copies fragments for
`Fragment` keys but stores the value **by reference** for `int` keys. Since `_align_slot`
mutates `fragment.atoms` in place (`builder.py:540`), a user who passes
`{0: "Zn_mof5_octahedral"}` gets the library copy in `self.sbu` rotated/translated —
every subsequent build using that SBU starts from corrupted coordinates. Same family:

- the final `_align_all_mappings` call scales the *caller's* topology in place
  (`builder.py:243`, via `scale_slots`);
- `_align_slot` rebuilds the `Molecule` without `site_properties` (`builder.py:540`),
  silently dropping anything a user attached.

### 1.6 `raise ("NoSymm")` in the topology pipeline
`scripts/cgd2pkl.py:175`: raising a string is a `TypeError` at runtime — it "works" only
because `read_cgd_data` catches blanket `Exception`. Together with
`assert len(fragment_sites) <= 12` used as flow control (asserts vanish under
`python -O`), the CGD pipeline's error handling is accidental rather than designed.

---

## 2. Reproducibility and validity

The two most important *behavioral* issues:

### 2.1 Builds are non-deterministic
`_align_slot` calls `m0.perturb(distance=0.01)` and `m1.perturb(...)`
(`builder.py:533-536`) with no seed, presumably to break Hungarian-assignment degeneracy
on symmetric dummy arrangements. Consequences:

1. The same inputs give different structures on every run — bad for science, bad for
   testing.
2. The Nelder-Mead objective (`opt_fun`) is **stochastic**, and Nelder-Mead's convergence
   criteria (`xatol`, `fatol`) behave poorly on noisy objectives — it can terminate
   spuriously or wander.

**Fix:** seed an `np.random.Generator` passed down from `build(seed=...)`, or better,
remove the perturbation entirely and break degeneracy deterministically (e.g.,
lexicographic tie-break on coordinates in the assignment).

### 2.2 No alignment quality gate
`build()` never compares the final RMSD to a threshold. A geometrically incompatible SBU
that passes the (weakened, see 1.1) symbol check produces a distorted structure returned
as if valid. `build_all` then only filters on `AssertionError`.

**Fix:** add a configurable `max_rmsd` (normalized per dummy, per fragment size) and
raise a typed `AlignmentError`. This is also exactly the mechanism that allows *loosening*
the symmetry pre-filter safely (see 3.1).

---

## 3. Solid-state science

### 3.1 Replace symbolic point-group compatibility with geometric matching
Current pipeline: compare dummy counts → compare Schoenflies symbols → (dead) subgroup
check → Hungarian alignment. The string-based symmetry gate is both too strict (a
slightly distorted D4h SBU labeled C2v gets rejected from a D4h slot it would fit fine)
and too loose (`this_size <= 3: return True` at `fragment.py:173` accepts
trigonal-*pyramidal* SBUs into trigonal-*planar* slots). What modern generators
(pormake, ToBaCCo-style) do — and the recommendation here:

1. Prefilter on connection count only (cheap, exact).
2. Normalize both dummy sets to unit vectors from the centroid.
3. Solve the optimal rotation (Kabsch/quaternion) under Hungarian assignment on pure
   numpy arrays.
4. Accept/rank by RMSD threshold.

Symmetry becomes an *emergent* property of the geometry rather than a symbol comparison;
handles near-symmetric and distorted SBUs gracefully; fixes 1.1 by deleting the code it
lives in; dramatically faster (see §4). Keep the point-group analysis for
metadata/reporting — genuinely useful for cataloguing — but not as the gate.

**Chirality must be handled explicitly:** `HungarianOrderMatcher` (and a naive Kabsch)
can return an improper rotation (det = −1), silently inverting chiral SBUs. Constrain to
proper rotations, and optionally offer the mirrored SBU as a separately-flagged solution.

### 3.2 Slot equivalence should come from crystallography, not fuzzy `Fragment.__eq__`
`Topology.mappings` groups "equivalent" slots via `Fragment.__eq__`/`__hash__` =
(Schoenflies symbol, atom count) (`fragment.py:93-108`). Two problems:

- **Scientific:** crystallographically *inequivalent* vertices with the same coordination
  number and local point group get merged (common in binodal/trinodal nets). Users can't
  put different SBUs on them, and `list_building_units` reports them as one slot type.
- **CS:** dict keys whose hash depends on mutable state (`len(self.atoms)` changes under
  `functionalize`) is a latent lookup-corruption bug.

The information is already available at generation time: in `scripts/cgd2pkl.py`,
`Structure.from_spacegroup` builds the crystal from unique Wyckoff sites — each generated
site's orbit membership **is** the equivalence class. Record an `equivalence_class`
integer per slot when building the topology database, and make `Topology.mappings` a
plain `dict[int, list[int]]`. Frozen, hashable, correct — and removes the need for
`Fragment` to be hashable at all.

### 3.3 Constrain cell optimization to the net's crystal system
`build()` optimizes (a, b, c) independently with angles frozen (`builder.py:226-238`).
For a cubic net like **pcu** this is 3 DOF where the symmetry allows 1 — the optimizer
can (and with a noisy objective, will) converge to a ≠ b ≠ c, *breaking the cubic
symmetry of the output*. Store the crystal system (known for free in cgd2pkl from the
space group) and optimize only the free parameters: cubic → 1 DOF,
tetragonal/hexagonal → 2, orthorhombic → 3, monoclinic → 4 (include β), triclinic → 6.
Fewer DOF = faster convergence *and* symmetry-correct outputs. For nets with one edge
type, seed with an analytic guess (sum of the two half-fragment dummy radii along an
edge) that's already near-optimal.

### 3.4 The 12-neighbor cap silently excludes important nets
`assert len(fragment_sites) <= 12` in `scripts/cgd2pkl.py:163` drops every net with
>12-coordinated vertices — that excludes **rht** (24-c), important Zr-cluster nets, and
others central to MOF chemistry. Make it a CLI parameter, and log *which* nets were
dropped and why (currently they vanish into the blanket `except`).

### 3.5 Output layer: the field speaks CIF, not networkx
`build()` returns a networkx graph, and the only export is GULP (`utils.py:447`) — a
GULP round-trip is needed just to get a CIF. Wrap the result in a `Framework` class
holding a pymatgen `Structure` (fractional coords, proper PBC wrapping) plus the bond
graph and mmtypes, with `.write_cif()`, `.to_ase()`, `.to_gulp()`, `.view()`. This also
restores the ergonomics the README still advertises from v2. Two science details to get
right: wrap atoms into the cell, and store periodic-image vectors on bonds (currently
inter-fragment bonds are implied by tag matching with raw cartesian coords, which is
ambiguous for atoms bonded across a boundary).

### 3.6 Scope features for the COF/zeolite ambition
- **2D nets** are extruded to c = 10 Å in `scripts/cgd2pkl.py:99`. For COFs, interlayer
  stacking (AA, AB, staggered, inclined) is a first-class scientific variable — a
  `stacking=` option on build for 2D nets would be a distinctive feature.
- **Zeolites:** RCSR covers some, but the authoritative source is the IZA database
  (framework types as CIF/systre files). An IZA importer alongside the CGD one would
  directly serve that goal.
- **RCSR currency:** the download URL is pinned to `RCSRnets-2019-06-01.cgd`; RCSR has
  grown since. Also `codecs.escape_decode` is a private CPython API — replace with an
  explicit decode.
- **Post-build sanity checks:** minimum-interatomic-distance screening (catch
  interpenetrating/overlapping garbage), optional UFF relaxation via ASE instead of
  external GULP.

---

## 4. Performance

Ordered by expected payoff:

1. **The optimizer hot loop deep-copies the world.** Every Nelder-Mead evaluation calls
   `topology.copy()` + `copy.deepcopy(mappings)` (`builder.py:217-218`), then
   `scale_slots` deep-copies every slot again, then `_align_slot` constructs perturbed
   pymatgen `Molecule`s per slot. For ~100 iterations × N slots this is thousands of deep
   copies of pymatgen objects to compute one scalar. The entire objective can be numpy:
   precompute each slot's dummy coordinates *fractionally* once; scaling is then one
   matrix multiply; alignment is `scipy.optimize.linear_sum_assignment` + Kabsch on
   (≤12 × 3) arrays. Expected 1–2 orders of magnitude speedup — this is what makes
   `build_all` at RCSR scale feasible.
2. **Startup cost:** `Autografs()` re-runs `PointGroupAnalyzer` on every SBU in
   `defaults.xyz` on every instantiation. Cache the parsed library keyed by file hash
   (and drop the analyzer from the critical path entirely if 3.1 lands).
3. **`fragments_to_networkx` is O(N²)** in atoms for the inter-fragment tag matching
   (`utils.py:416-418`) — a `dict[tag, nodes]` makes it linear. Same for the per-slot
   tag-transfer double loop in `_align_slot` → one `scipy.spatial.distance.cdist`.
4. **`build_all` parallelism:** embarrassingly parallel (the commented-out `cpu_count`
   line shows the intent). With the mutation issues from 1.5 fixed (pure functions, no
   shared state), `concurrent.futures.ProcessPoolExecutor` is straightforward. Also:
   `itertools.product` over all slot×SBU combinations is materialized into a list —
   stream it, and add a sampling strategy (`max_per_topology=`, seedable random sampling)
   because the product explodes combinatorially at scale.

---

## 5. Software engineering / maintainability

### 5.1 Data distribution — replace dill pickles
`topologies.pkl` is (a) not in the repo, so a fresh clone can't run the real pipeline or
its tests; (b) a pickle, so loading it executes arbitrary code — a real concern if users
ever share topology packs; (c) coupled to pymatgen's internal object layout, so a
pymatgen upgrade can orphan every existing file. A topology is just
`{name, cell, per-slot: species/coords/tags/equivalence_class}` — serialize to compressed
JSON (or msgpack) and reconstruct objects on load. Diffable, versionable, safe, and
enables **committing a tiny 3-net fixture (pcu, srs, hcb)** so CI finally exercises the
real build path end-to-end.

### 5.2 API/typing hygiene
- `list_building_units` is annotated `-> dict[str, Fragment]` but returns
  `dict[Fragment, list[str]]` keyed by slot fragments (`builder.py:400-405`); the `sieve`
  parameter is rebound from `str` to `Fragment`/`Topology` mid-function in both list
  methods. Mypy would flag all of this — it's configured but not run in CI.
- Module import side effect: `utils.py:59` sets
  `pandas.options.mode.chained_assignment = None` **globally for the user's whole
  process**. The warning it suppresses is real: `radii.symbol = radii.symbol.str[:2]` at
  `utils.py:255` writes to a slice view; fix the copy instead of muting pandas.
- `tqdm` and (for the script) `requests` are used but undeclared — they arrive
  transitively via pymatgen today, which is luck, not a dependency spec.
- Replace `autografs.data.__path__[0]` with `importlib.resources.files("autografs.data")`
  (zip-safe, modern idiom).
- Custom exceptions (`IncompatibleSlotsError`, `AlignmentError`, `TopologyParseError`)
  instead of assert-as-control-flow (`builder.py:163` catches `AssertionError` from deep
  inside) and blanket `except Exception`.
- Single-source the version via `importlib.metadata` — currently duplicated in
  `pyproject.toml` and `__init__.py`.
- Delete the ~50-line commented-out scratchpad in `__main__` at `builder.py:560-610` —
  that's what git history is for.

### 5.3 Tooling modernization
- **ruff** replaces black + isort + flake8 in one tool (its rules would have caught
  several items above, e.g. f-strings in logging calls). Add **pre-commit**.
- Run **mypy in CI** (configured and dep-declared, but CI only runs flake8 with
  `--exit-zero`, i.e. advisory-only).
- **uv** for lockfile-based dev installs and much faster CI.
- `requires-python >= 3.13` is unusually aggressive for a scientific package — most of
  the ecosystem (HPC clusters especially) sits on 3.10–3.12. Nothing in the code needs
  3.13; consider ≥3.11 and a CI matrix.
- Promote `cgd2pkl.py` into the package as a console entry point
  (`autografs-db build --use-rcsr`), plus an `autografs build ...` CLI — regenerating the
  topology DB currently requires knowing about a loose script.
- README still documents the v2 API (`make()`, `functionalize()` on the framework,
  `mmanalysis`) — either port those features or rewrite the examples; it's the first
  thing users hit.

### 5.4 Testing
104 tests with hypothesis is a genuinely good base. Gaps:

- No golden-structure regression test (build pcu + known SBUs → assert cell params, RMSD,
  atom count against stored reference). Determinism (§2) is a prerequisite.
- No CGD-parser tests (embed small CGD strings — the parser is the most fragile code in
  the repo).
- Nothing exercises real topologies in CI until the fixture-data problem (5.1) is fixed.

---

## 6. Suggested order of attack

1. **Bug-fix release** — §1 items plus seeding/determinism and the RMSD gate (§2). Small
   diffs, immediate correctness value, unblocks golden tests.
2. **Data format migration** (pkl → JSON with equivalence classes and crystal system
   stored; committed test fixtures). Unblocks CI, 3.2, and 3.3.
3. **Numpy alignment core** (3.1 + 4.1 together — they're the same rewrite): geometric
   matching, chirality handling, deterministic, fast.
4. **Output layer** (`Framework` class, CIF/ASE) + README rewrite against the real API.
5. **Symmetry-constrained cell optimization** (3.3).
6. **Tooling sweep** (ruff, pre-commit, mypy-in-CI, uv, entry points, version widening) —
   mechanical, can happen anytime.
7. **Scale features**: parallel + sampling `build_all`, SBU cache, 2D stacking, IZA
   import, RCSR refresh.

Items 1 and 2 are each roughly an afternoon of work; item 3 is the big one but pays for
itself the first time `build_all` runs over the full RCSR set.
