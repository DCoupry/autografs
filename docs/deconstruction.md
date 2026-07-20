# Deconstruction: from a crystal structure back to SBUs + net

*Part of the [AuToGraFS documentation](../README.md#documentation).*

The inverse pipeline. `deconstruct` takes an experimental structure (a
CIF file or a pymatgen `Structure`), detects bonds with the same
strategy the builder uses, removes free guests, clusters atoms into
building units, places a dummy at every cut bond, and matches the
resulting quotient graph against the topology library by
coordination-sequence signature. Clustering follows the *metal-oxo*
convention for MOFs (metal clusters keep their inorganic coordination
sphere and their carboxylate / phosphonate / sulfonate binding groups)
and a *branch-point* convention for metal-free frameworks (COFs) —
rigid ring systems and non-ring atoms collapse to super-vertices, and
a super-vertex's external connection count sets its role (≥3 a node, 2
a linker, 1 a cap):

```python
result = mofgen.deconstruct("IRMOF-1.cif")

result.net_candidates       # ['pcu']
result.fragments            # {'node_C6O13Zn4_6X': ..., 'linker_C6H4_2X': ...}
result.units                # every placed unit: kind (node/linker/cap),
                            # atom indices, connection count
result.guest_formulas       # compositions of removed free solvent
result.write_xyz("harvested_sbus.xyz")   # library-ready SBU file
```

The extracted fragments are ordinary `Fragment` objects with `X`
dummies at the cut-bond midpoints, so they feed straight back into the
build pipeline — `Autografs(xyzfile="harvested_sbus.xyz")` or passing
them directly in `build` mappings both work, which makes a full
deconstruct → rebuild → `verify_net` round trip possible.

Net identification is signature-based (the multiset of per-vertex
coordination sequences, computed on the quotient graph with capping
ligands pruned): an almost-unique invariant matched in two tiers,
first with ditopic linkers counted as vertices (separating a net from
its edge-decorated derivatives), then against the underlying
2-coordination-suppressed net. The returned list carries the tier on
its `.tier` attribute (`"exact"`, `"contracted"`, or `None`) — worth
checking before trusting a result, since a contracted-tier match is
blind to edge decoration; the per-subframework entries in
`subframework_nets` carry it too. Multiple candidates are returned as a
list rather than silently picking one.

Interpenetrated (catenated) structures are handled: each periodic
subframework is a separate connected component, identified on its own,
so `n_periodic_components` gives the fold and `subframework_nets` the
per-net results. `net_candidates` is their consensus, and
`is_catenated` is a convenience flag:

```python
result = mofgen.deconstruct("IRMOF-9.cif")   # 2-fold interpenetrated pcu
result.n_periodic_components   # 2
result.subframework_nets       # [['pcu'], ['pcu']]
result.net_candidates          # ['pcu']   (consensus)
result.is_catenated            # True
```

Both MOFs and COFs are handled; the COF path uses the single-node
convention, so a node bundled differently in the original SBU library
(e.g. a triphenylamine core cut at its central atom) may come back more
finely divided, but the recovered net is the same.

Rod MOFs (1-periodic building units, e.g. the MOF-74 family) are
detected and reported rather than rejected: a rod has no finite
fragment, so it appears in `rod_units` (axis, crystallographic repeat,
points of extension, cut count) instead of `fragments`, and in the
quotient graph the rod is replaced by its points of extension joined
along the axis — the O'Keeffe PoE convention — before net
identification. Because that expansion carries no blueprint edge
centers, rod nets typically match on the contracted tier (check
`subframework_nets[i].tier`). A bare 1-periodic polymer with no
framework connections still raises `DeconstructionError`, as do
2-periodic (layer) building units.

Rods also get a canonical identity (`autografs.rods`): `canonical_rod`
reduces a detected rod to one *chemical* repeat in a cylindrical frame
about its axis — detecting screw rods whose crystallographic repeat is
a multiple of the chemical one (the MOF-74 helix) and recording the
screw order and signed screw angle. `RodRepeat.matches` compares rods
modulo everything the crystal embedding chooses freely (rotation about
the axis, axial phase, and the proper flip), so a 2× supercell
deconstruction dedupes with a 1× one, while enantiomeric screws stay
distinct — helicity is chiral, and no proper isometry relates them.

## Harvesting a library from many structures

`harvest` runs `deconstruct` over a batch — a directory of CIFs, a
glob, or an iterable of paths/Structures — and merges the building
units into one deduplicated, library-ready fragment set. The same
paddlewheel appearing in fifty MOFs becomes one fragment tagged with
every source it came from; deconstruction failures are recorded rather
than aborting the run, so a real success rate falls out:

```python
result = mofgen.harvest("core_mof_subset/")

result.report()          # 'harvested 34 fragments (...) from 47/50 structures, 3 failed'
result.building_units    # nodes + linkers (bound-solvent caps excluded)
result.provenance        # {'node_C4O8Zn2_4X': ['HKUST-1', 'MOF-505', ...], ...}
result.nets              # {'HKUST-1': ['tbo'], ...}  per-source net candidates
result.failures          # {'disordered_entry': 'DeconstructionError: ...', ...}
result.rods              # {'rod_OZn': RodRepeat(...)} rod families (see above)
result.rod_provenance    # {'rod_OZn': ['MOF-74-Zn', ...]}

result.write_xyz("harvested_sbus.xyz")          # nodes + linkers by default
mofgen2 = Autografs(xyzfile="harvested_sbus.xyz")   # build with the harvest
```

Monotopic organic units (a single connection point — bound solvent,
modulators, capping residues) are classified `cap` and excluded from
the default `write_xyz` output and the `building_units` view.

## Assembly fingerprints: is this combination realized?

`autografs.fingerprint` labels an assembly as the hashable triple
(nets, building-block multiset, interpenetration fold), with block
identity expressed in one shared vocabulary — a harvest's deduplicated
fragment library. A framework *built from* that library fingerprints
equal to the experimental structure it came from, so screening an
enumeration against a corpus is a set lookup:

```python
from autografs import fingerprint

harvest = mofgen.harvest("corpus/")
realized = {
    fingerprint.from_deconstruction(mofgen.deconstruct(cif),
                                    library=harvest.fragments)
    for cif in corpus_cifs
}
for framework in mofgen.build_all(...):        # built from the harvest
    if fingerprint.from_framework(framework) not in realized:
        ...                                     # an unrealized combination
```

Block counts are gcd-reduced (supercells fingerprint identically), caps
are excluded (same convention as harvesting), and blocks with no match
in the vocabulary carry an `unmatched:` marker that never collides with
a buildable combination.
