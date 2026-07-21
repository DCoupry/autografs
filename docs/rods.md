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

The rod is placed on one of the blueprint's **slot runs** and a ditopic
linker fills every other slot. Two structural facts fall out of a rod
being one-periodic, and both differ from the finite pipeline:

- **The rod pins a cell parameter.** The run-axis length is fixed to
  `n_repeats × chemical repeat`, never a free parameter. The remaining
  freedom is one in-plane scale, optimized together with the rod's own
  axis rotation and axial phase against covalent bond-length targets.
- **Inter-unit bonds are explicit graph edges,** not the tag pairs finite
  SBUs use — a rod anchor (the metal) carries several connections on one
  atom. A relief pass then rotates each ditopic linker about its own arm
  axis (which leaves the anchors, hence the bonds, fixed) to clear steric
  clashes.

Closure, alignment RMSD, and `min_distance` are hard gates;
out-of-scope inputs raise `AlignmentError` with the reason.

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

Rod building currently covers single-axis runs with one rod species per
run. **Cross-linked multi-rod nets** — where several interleaved helices
share the same linkers, `etb` (MOF-74) itself being the canonical example
— and mixed rod/finite mappings are future work: detection
(`helical_runs`) and single-helix building (validated on `unc`) are in
place, but the general multi-rod placement is not yet. Detection is also
deliberately conservative about 2D layer nets (an in-plane zig-zag is not
a 3D rod channel and is skipped).

For the reverse direction and the rest of the inverse pipeline, see
[Deconstruction](deconstruction.md).
