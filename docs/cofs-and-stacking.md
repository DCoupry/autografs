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
`autografs.exceptions.StackingError`. `mode="random"` builds an n-layer
turbostratic disorder model, and an explicit `sequence` of offsets covers
ABC and beyond.

## Twisted (moiré) bilayers

An arbitrary twist angle between two layers breaks periodicity; only
coincidence-site-lattice (CSL) angles give a periodic moiré supercell, and
the supercell grows roughly as 1/θ². `mode="twisted"` therefore *snaps* the
requested angle to the nearest commensurate angle of the layer's lattice:

```python
moire = layer.stack(mode="twisted", angle=22.0)     # snaps to 21.787° (Σ7)
moire.graph.graph["twist_angle"]                    # the exact angle built
moire.graph.graph["twist_strain"]                   # 0 for exact CSL angles
```

The search (`autografs.twist.commensurate_twists`) is plane-group-general:
it enumerates coincidence vector pairs rather than hardcoding the hexagonal
`cos θ = (m² + 4mn + n²) / (2(m² + mn + n²))` formula — which it reproduces
exactly on hexagonal layers, along with the square-lattice `2·atan(n/m)`
family. Knobs:

- `angle_tolerance` (default 1.0°) bounds the snap distance; if no
  commensurate angle is close enough, the error lists the nearest ones.
- `max_index` (default 8) bounds the coincidence search; raise it to reach
  smaller angles (at the price of larger supercells).
- `max_strain` (default 0) enables the strained-commensurate mode standard
  in twistronics tooling: approximately commensurate angles are admitted and
  the mismatch is absorbed as an in-plane strain of the top layer (reported
  in `twist_strain`). Rectangular and oblique layers usually need this —
  generic lattices have no exact CSL rotations.
- `max_atoms` (default 20000) is the runaway guardrail: small angles
  genuinely need huge cells, so exceeding the cap raises instead of quietly
  building a million-atom Framework.

For angle scans (moiré-porosity studies), call the search directly:

```python
from autografs.twist import commensurate_twists

for cand in commensurate_twists(layer.cell, max_index=10):
    print(f"{cand.angle:8.3f}°  {cand.n_cells} cells/layer  strain {cand.strain:.1e}")
```

COFs are metal-free, but AuToGraFS can also deconstruct them back into building
units and identify their net — see [Deconstruction](deconstruction.md).
