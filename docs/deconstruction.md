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
2-coordination-suppressed net. Multiple candidates are returned as a
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
finely divided, but the recovered net is the same. Scope: frameworks
with molecular building units. Rod MOFs and other 1-periodic (chain)
building units raise `DeconstructionError`.

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

result.write_xyz("harvested_sbus.xyz")          # nodes + linkers by default
mofgen2 = Autografs(xyzfile="harvested_sbus.xyz")   # build with the harvest
```

Monotopic organic units (a single connection point — bound solvent,
modulators, capping residues) are classified `cap` and excluded from
the default `write_xyz` output and the `building_units` view.
