# Rod MOFs: 1-periodic building units (MOF-74 & friends)

Most framework generators — AuToGraFS 2.x included — assume every
building block is a *finite* molecule: a node or a linker with a handful
of connection points. A large and important class of materials breaks
that assumption. In **rod MOFs** the inorganic building unit is an
*infinite chain* — a metal-oxide rod running through the crystal, with
organic linkers hung off it. The MOF-74 / CPO-27 family (M₂(dobdc)), many
metal carboxylates, and metal formates are all rod MOFs. Their rods are
often **helical**: the metal-oxo chain spirals with a screw symmetry
(MOF-74 is a 3₁ screw), and helicity is chiral.

A rod has no finite fragment, so it cannot be a `Fragment` and cannot
round-trip through the finite builder. AuToGraFS 3.x handles rods as a
first-class case, both directions:

- **deconstruct** a rod MOF into its rod(s) + linkers and identify the net;
- **harvest** rod families across many structures with a canonical,
  screw-aware identity;
- **build** a rod framework forward from a harvested rod + a linker — for
  **straight and helical rods alike**, including general (non-180°)
  screws;
- **verify** that a built rod framework realizes its blueprint net.

Forward building of helical rods is, to our knowledge, not available in
other open framework assemblers.

---

## Deconstructing a rod MOF

`deconstruct` detects a 1-periodic building unit automatically and
reports it in `rod_units` instead of `fragments`:

```python
from autografs import Autografs

mofgen = Autografs()
result = mofgen.deconstruct("MOF-74-Zn.cif")

result.net_candidates      # ['etb']   (MOF-74's net)
result.rod_units           # [RodUnit(axis=..., repeat_length=..., ...)]
result.fragments           # the organic linker(s) only
rod = result.rod_units[0]
rod.axis, rod.repeat_length, rod.generator     # the chain's direction + period
rod.n_connections                              # cut bonds per crystallographic repeat
```

In the quotient graph the rod is replaced by its **points of extension**
(the atoms that bond out to the rest of the framework), joined along the
axis — the O'Keeffe PoE convention, under which MOF-74 identifies as
`etb`. Because that expansion carries no blueprint edge centers, rod nets
usually match on the *contracted* tier — check
`result.subframework_nets[i].tier`. A bare 1-periodic polymer with no
framework connections still raises `DeconstructionError` (it is not a
framework), as do 2-periodic (layer) building units.

## Canonical rod identity (screw-aware)

A rod's identity is its *chemical* repeat, modulo everything the crystal
embedding chooses freely — rotation about the axis, axial phase,
translation, and the proper flip. `autografs.rods.canonical_rod` reduces
a detected rod to one chemical repeat in a cylindrical frame about its
axis:

```python
from autografs.rods import canonical_rod

repeat = canonical_rod(result.structure, rod)
repeat.formula          # 'O2Zn2'  (one chemical repeat)
repeat.repeat_length    # the chemical repeat along the axis (Å)
repeat.screw_order      # chemical repeats per crystallographic repeat (1 = straight)
repeat.screw_angle      # signed rotation per chemical repeat, degrees (0 = straight)
```

The screw is detected by self-matching the atoms under an (axial shift,
rotation): a MOF-74-style rod whose crystallographic repeat is three
chemical repeats around a 120° screw reduces to `screw_order = 3`,
`screw_angle = ±120`. The **sign** matters — it is the handedness, and it
is frame-invariant once the axial step is normalized positive, so
`RodRepeat.matches` treats enantiomeric screws (+120° vs −120°) as
**different** building units. No proper isometry relates them; merging
them would be a chemistry error. A 2× supercell of the same rod, by
contrast, dedupes back to the 1× chemical repeat.

## Harvesting rod families

`harvest` collects rods across a batch just like it collects finite SBUs,
deduplicating them by canonical identity with per-source provenance:

```python
result = mofgen.harvest("mof74_family/")

result.rods                 # {'rod_O2Zn2': RodFragment(...), ...}
result.rod_provenance       # {'rod_O2Zn2': ['MOF-74-Zn', 'CPO-27-Zn', ...]}
result.write_rods("rods.json")     # a buildable rod library (JSON sidecar)
```

Rods can't join the XYZ SBU format (no finite molecule), so they get
their own versioned JSON sidecar — `save_rods` / `load_rods`, or
`HarvestResult.write_rods`. A `RodFragment` carries the identity
(`repeat`), the local-frame atom template (`positions`), the connection
arms (`arms`, from the cut-bond midpoints), and the internal bond graph
(`bonds`, including the screw-image continuation) — everything the
forward builder needs.

## Building a rod framework forward

A harvested rod goes back through the builder with `build_rod`:

```python
harvest = mofgen.harvest("mof74_family/")
rod = harvest.rods["rod_O2Zn2"]

mof = mofgen.build_rod(mofgen.topologies["pcu"], rod, "Benzene_linear")
mof.write_cif("built.cif")
```

The rod is placed on one of the blueprint's **slot runs** and finite SBUs
fill every other slot. Two structural facts fall out of a rod being
one-periodic, and both differ from the finite pipeline:

- **The rod pins a cell parameter.** The run-axis length is fixed to
  `n_repeats × chemical repeat`, never a free parameter. The remaining
  freedom is one in-plane scale, optimized together with the rod's own
  axis rotation and axial phase against covalent bond-length targets.
- **Inter-unit bonds are explicit graph edges,** not the tag pairs finite
  SBUs use — a rod anchor (the metal) carries several connections on one
  atom. Every connection point in the build, rod arm tip and SBU dummy
  alike, is paired against every other by shortest periodic separation. A
  relief pass then rotates each straight ditopic linker about its own arm
  axis (which leaves the anchors, hence the bonds, fixed) to clear steric
  clashes.

Closure, alignment RMSD, and `min_distance` are hard gates;
out-of-scope inputs raise `AlignmentError` with the reason.

### Mixed rod / finite mappings

The third argument takes a **mapping**, exactly like `build`: one
Fragment (or SBU name) for every lateral slot, or a `{slot type: SBU}` /
`{slot index: SBU}` dict when the blueprint's non-run slots are not all
alike. Lateral SBUs may be of any connectivity — and a polytopic one
bonds to its lateral *neighbours* as well as to the rod, which is what
real rod MOFs need: MOF-74's DOBDC is a 4-connected bridge, not a ditopic
linker, and no library net with a rod run has polytopic laterals that
touch only run nodes.

```python
etbe = mofgen.topologies["etb-e"]        # etb's edge net: 2-c and 4-c laterals
mof = mofgen.build_rod(etbe, rod, {ditopic_slot_type: "Benzene_linear",
                                   tetratopic_slot_type: dobdc})
```

How many lateral arms a rod needs is read off the run itself — a node
slot's degree minus the connections the run's own continuation consumes —
so a blueprint node of any local geometry is handled.

Which slots may bond is the blueprint's business, not geometry's: the
optimizer spends one bond budget per bonded slot pair of the blueprint,
and geometry only picks which individual ports (and periodic images)
realize them. Pairing purely by proximity looked simpler but made the
objective multi-modal — its *global* minimum could be a spurious pairing
at a badly inflated cell, where unrelated connection points happen to
meet.

### Leaving a slot empty

A 2-connected slot mapped to `None` is left **empty**: nothing is placed
and its two neighbours bond to each other directly.

```python
mof = mofgen.build_rod(etbe, rod, {decoration_slot_type: None,
                                   tetratopic_slot_type: dobdc})
```

Real rod MOFs need this. A MOF-74 metal-oxo rod binds *straight* onto its
4-connected DOBDC, but the blueprint decorates every edge with a
2-connected slot the structure has no unit for — which is also why such
structures identify on the contracted tier when deconstructed. Net
verification knows about it: the emptied slots are contracted out of the
blueprint before the comparison, exactly as a run's own axial edge
centers are.

### Straight vs helical runs

Which run a rod occupies is chosen automatically from its screw:

- A **straight** rod builds on a straight axial run
  (`autografs.net.axial_runs`) — a `pcu`-family chain with one
  point-of-extension slot per period, supercelled along the axis.
- A **helical** rod builds on a helical run
  (`autografs.net.helical_runs`) — a spiralling channel whose
  `screw_order` node slots are filled 1:1 by the rod's chemical repeats,
  laid down by the screw operation about the run's axis line. The run
  must agree with the rod on node count *and signed angle*, so a
  left-handed net never hosts a right-handed rod.

```python
from autografs.net import axial_runs, helical_runs

axial_runs(mofgen.topologies["pcu"])     # three straight runs (a, b, c)
helical_runs(mofgen.topologies["etb"])   # MOF-74's 3₁ screw (order 3, ±120°)
helical_runs(mofgen.topologies["srs"])   # the 4₁ chiral net (order 4, +90°)
```

A 2₁ (180°) screw is a special case: a ditopic linker's arm sign-flip is
a no-op, so it still builds on a straight `pcu` run. A general screw
(3₁, 4₁, …) rotates the linkers to genuinely new directions and needs a
spiralling blueprint. For example, `unc` is a 4₁ single-helix net; a
four-repeat −Zn–O– helix builds on it, and a round-trip recovers it:

```python
mof = mofgen.build_rod(unc_topology, rod_4_1, linker, verify_net=True)
mof.min_contact()                        # a clean, un-clashed structure
mofgen.deconstruct(mof.structure).net_candidates   # ['unc'] — the same net
```

### Cross-linked multi-rod nets (etb / MOF-74 proper)

MOF-74's own net, `etb`, is not a single helix but **six interleaved 3₁
helices** per cell, joined by ditopic linkers that bridge one helix to
another. `build_rod` handles this directly: it finds every helical run,
places one rod on each helix, and lets the linkers cross-link them. Two
things make it work from a *single* harvested rod:

- **Both handednesses.** `etb` is centrosymmetric — three helices spiral
  each way. The opposite-hand helices are filled with the rod's
  **enantiomer** (a reflection of the template), which is a proper copy
  for an achiral metal-oxo rod, so one harvested rod fills all six.
- **The linkers bridge rods, not repeats.** Every non-run slot connects a
  node on one helix to a node on another; the optimizer's global port
  pairing wires them across rods.

A −Zn–O− rod built on `etb` gives six cross-linked helices that
`verify_net(etb)` accepts and that identify as `etb`. (The idealized net
is small, so a synthetic rod crowds it — `relax()` cleans the packing;
the topology is exact.)

`etb-e`, the same six helices with 4-connected nodes and a tetratopic
lateral linker, is the mixed-mapping version of the same structure — the
MOF-74 arrangement as it actually bonds, rod → ditopic → tetratopic
bridge → ditopic → rod. It builds with two lateral species, clears the
default contact gate unrelaxed, and identifies as `etb-e`.

### Several nodes per period

A straight run usually carries one point-of-extension node per period,
and the rod is supercelled along it. 66 library nets instead **chain
several** — `cds` alternates two 4-connected nodes per period — and each
of those nodes holds one of the rod's chemical repeats, so the blueprint
period already contains *k* of them and no supercell may be needed at
all.

Two things have to line up, and both are checked:

- the nodes must be **evenly spaced** along the axis, because a rod's
  chemical repeats are;
- consecutive nodes must be related by a **constant rotation** about the
  axis, and the rod's own screw must match it. This is subtler than it
  looks: a run can be perfectly straight — its slot *centres* collinear —
  while the slots' *orientations* turn. `cds` turns 90° per node, so it
  wants a 4₁ rod even though nothing about the run's geometry is helical.

```python
mof = mofgen.build_rod(mofgen.topologies["cds"], rod_4_1, linker)
mof.verify_net(mofgen.topologies["cds"])
```

Where a rod does not fit a multi-node run, `build_rod` offers it the
blueprint's other runs rather than refusing outright — `cds` also carries
two one-node runs, which a screwless rod fills happily.

### Woven (multi-axis) packings

Some nets spiral along several cell axes at once. `bmn` is the clean
case: six 4₁ helices per cubic cell, **two along each axis**, together
covering every one of its 24 node slots without sharing one. `build_rod`
places a rod on all six.

```python
mof = mofgen.build_rod(mofgen.topologies["bmn"], rod_4_1, linker)
mof.structure.lattice.abc     # cubic — all three axes pinned by the rod
```

Each axis carries its own placement freedoms (a rotation about it and an
axial phase), and each is **pinned** to `n_repeats × chemical repeat`.
A fully woven packing therefore determines the whole cell: there is
nothing left for the in-plane scale of a single-axis build to vary. That
also sets the limit — one rod pins every axis it runs along to the *same*
length, so runs of different periods cannot be woven together. Where a
net offers several (`bbe` spirals 2₁ along `a` and `b` with different
periods), the busiest period is built and the other runs' slots stay
lateral, needing linkers like any other slot.

### Channels that close on several cells

A helix need not come back onto itself after one cell. `twt`'s channel
advances `c`/3 and turns 60° per node, so translating it by `c` lands on
a *half-turn-rotated* copy of itself; it only closes after **two** cells,
and its run direction is ⟨002⟩ rather than ⟨001⟩. `fnt`, `uom`, `uoo`,
`fne`, `src` and `twt-e` do the same; `mdf`-family nets close on three.

Nothing about the cell parametrization changes — the channel still runs
along a single cell axis — but the pinned parameter is the run period
**divided by** that multiple:

```python
mof = mofgen.build_rod(mofgen.topologies["twt"], rod_6_1, linker)
mof.structure.lattice.abc[2]   # 6 × chemical repeat ÷ 2, not × 6
```

The picture behind the division: one cell holds *all six* of the run's
node slots, filled by **two interleaved passes** of the same rod half a
turn apart, rather than one pass of a rod six repeats long. Because the
two passes are a lattice translation apart in height but not in azimuth,
neither is the other's periodic image, and the six repeats stay distinct.

Some nets describe one channel *both* ways — `nlr` reports a ⟨001⟩ and a
⟨002⟩ walk over the same node slots — and run selection deduplicates by
node set, keeping the shorter description, so those build exactly as
before. Where the two are genuinely different channels (`mdf`'s ⟨003⟩
helices do not share nodes with its ⟨001⟩ ones), the usual policy
applies: the busiest set of runs is built, ties going to the shorter
period, and `run=` forces the other.

Runs along a *diagonal* (⟨011⟩ and friends, ~110 nets) are still out of
scope: their period is a **combination** of cell parameters, so it cannot
be substituted into a cell row and has to enter the optimizer as a
constraint on the free parameters instead.

### Verifying the built net

Passing `verify_net=True` (or calling `mof.verify_net(topology)` after
the fact) checks that the build actually realizes the blueprint's net.
Rod frameworks are verified against the blueprint's **points-of-extension
form** — the forward mirror of the deconstruction expansion — rather than
by the exact slot-multiset comparison the finite pipeline uses, because a
rod's slots are its repeats and linker placements, not blueprint slots. A
mis-wired build (a dropped continuation, a linker on the wrong unit)
raises `NetMismatchError`.

## Editing rod frameworks

`Framework.is_rod` flags a rod build. Because rods carry explicit
inter-unit bonds and no anchor tags — and because "removing one SBU" from
an infinite rod would leave dangling chain ends with no cap chemistry —
the tag/anchor-based edits (`defects`, `rotate`, `flip`,
`functionalize`) refuse rod frameworks with a clear error. `supercell`
still works (it extends the rod and stays marked as a rod). To decorate a
rod MOF, edit the linkers of the source structure and rebuild. See
[Editing](editing.md).

## Scope & limitations

Rod building covers runs along a **cell axis**, with one rod species per
net: straight (single-node and multi-node), single-helix, cross-linked
multi-rod (`etb`), multi-axis woven (`bmn`), and — since #173 — channels
that close only after several cells along their axis (`twt`). Lateral
slots take a per-slot mapping of finite SBUs of any connectivity, or
`None` to leave a 2-connected slot empty.

What is left out:

- runs along a lattice **diagonal** (⟨011⟩ and friends), whose period is
  a combination of cell parameters rather than one of them;
- a *straight* run closing on several cells — none exists in the library
  (every axis-multiple run is helical), and the straight path supercells
  whole blueprint periods, so it is refused rather than mis-built;
- more than one rod species per net.

Detection is also deliberately conservative about 2D layer nets (an
in-plane zig-zag is not a 3D rod channel and is skipped).

For the reverse direction and the rest of the inverse pipeline, see
[Deconstruction](deconstruction.md).
