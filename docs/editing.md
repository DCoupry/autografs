# Editing building blocks and frameworks

*Part of the [AuToGraFS documentation](../README.md#documentation).*

## Editing building blocks (pre-build)

`Fragment` carries pre-build editing methods — modify a copy of a library SBU,
then build with the modified object:

```python
linker = mofgen.sbu["Benzene_linear"].copy()

linker.rotate(3.14159 / 2)        # around the dummy-dummy axis (2-connected)
linker.flip()                     # apply a mirror operation, if one exists
linker.functionalize(index=3, functional_group="amine")  # H -> NH2 etc.
# available groups: pymatgen.core.structure.FunctionalGroups

mof = mofgen.build(topology, mappings={edge_type: linker, node_type: "Zn_mof5_octahedral"})
```

## Post-build editing

A built `Framework` records which placed SBU every atom belongs to
(`mof.slots` lists them), so a structure can keep being edited after the
build. All editing methods return a new `Framework` and leave the input
untouched, so they chain:

```python
mof = mofgen.build(topology, mappings)

# supercells: bonds crossing the cell boundary are remapped onto the
# correct periodic image, so the supercell bond graph is exact
big = mof.supercell(2)                    # 2x2x2
big = mof.supercell((2, 2, 1))

# statistical defects: remove whole SBUs, cap the dangling connection
# points with hydrogen (missing-linker / missing-node defects)
defective = big.defects(fraction=0.1, sbu="Benzene_linear", seed=42)
defective = big.defects(slots=[5, 13])    # explicit choice instead
defective = big.defects(fraction=0.1, cap=None)   # open metal sites

# re-orient one placed unit (see mof.slots for the indices):
# rotate a 2-connected linker around its bond axis, or mirror a unit
# through a plane that keeps its connection points fixed
mof = mof.rotate(slot=1, theta=3.14159 / 2)
mof = mof.flip(slot=1)

# graft functional groups onto the framework itself - site by site,
# including sites made inequivalent by a supercell or a defect
sites = mof.functionalizable_sites()               # terminal H atoms
tagged = mof.functionalize(sites[0], "amine")      # one site
tagged = mof.functionalize(sites[:4], "methyl")    # several at once
```

`functionalize` accepts every group in
`pymatgen.core.structure.FunctionalGroups` or any pymatgen `Molecule`
with exactly one `X` dummy marking the attachment point. Defect capping
and group placement both use tabulated covalent bond lengths, and new
atoms get UFF4MOF types from their local environment — edited
frameworks stay valid inputs for `relax()`, `min_contact()` and every
export. Since defects and grafts distort nothing else, a final
`relax()` is the recommended clean-up for production structures.

### Rod frameworks

Frameworks built with `build_rod` (`Framework.is_rod`) hold their
inter-unit bonds as explicit edges and carry no anchor tags, so the
tag/anchor-based edits — `defects`, `rotate`, `flip`, `functionalize`
— refuse them with a clear error (removing "one SBU" from an infinite
rod would leave dangling rod ends with no cap chemistry). `supercell`
still works: it extends the rod, and the result stays marked as a rod
framework. To decorate a rod MOF, edit the linkers of the source
structure and rebuild. See [Rod MOFs](rods.md) for building rods in the
first place.
