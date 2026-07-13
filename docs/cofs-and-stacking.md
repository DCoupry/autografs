# 2D COFs and stacking

*Part of the [AuToGraFS documentation](../README.md#documentation).*

Layer nets (hcb, sql, kgm, hxl, ...) are stored as 2D plane-group topologies.
A build on one produces a single flat layer in a padded slab: the in-plane
cell is optimized while c stays frozen, since the interlayer spacing is
dispersion-driven chemistry, not topology. The COF-1 prototype:

```python
from autografs import Autografs

mofgen = Autografs()
hcb = mofgen.topologies["hcb"]

mappings = {}
for slot_type in hcb.mappings:
    n_connections = len(slot_type.atoms.indices_from_symbol("X"))
    mappings[slot_type] = (
        "Boroxine_triangle" if n_connections == 3 else "Benzene_linear"
    )

layer = mofgen.build(hcb, mappings=mappings)
print(layer)   # hexagonal layer, a = b = 14.7, gamma = 120

# turn the layer into a crystal by choosing the stacking
cof = layer.stack(mode="AA", interlayer=3.35)   # eclipsed
cof = layer.stack(mode="AB")                    # two-layer cell, offset (1/3, 2/3)
cof = layer.stack(mode="serrated", offset=(0.5, 0))
cof.write_cif("cof1.cif")
```

`stack` returns a new `Framework`: AA keeps one layer per cell with
`c = interlayer`; AB / serrated / staggered build a two-layer cell with an
in-plane-offset copy. Layers are van-der-Waals stacked (no inter-layer bonds).
The default `interlayer=3.35` Å is graphite-like; typical COFs fall in
3.3–3.6. Stacking a non-layered framework raises
`autografs.exceptions.StackingError`.

COFs are metal-free, but AuToGraFS can also deconstruct them back into building
units and identify their net — see [Deconstruction](deconstruction.md).
