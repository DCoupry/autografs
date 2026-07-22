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

Rod building covers single-axis runs — straight, single-helix, and
cross-linked multi-rod (`etb`) — with one rod species per net. Mixed
rod/finite mappings (a rod net that also needs finite SBUs on some
slots) and multi-axis "woven" rod packings are future work. Detection is
also deliberately conservative about 2D layer nets (an in-plane zig-zag is not
a 3D rod channel and is skipped).

For the reverse direction and the rest of the inverse pipeline, see
[Deconstruction](deconstruction.md).
