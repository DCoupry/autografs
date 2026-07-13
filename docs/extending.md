# Extending the libraries

*Part of the [AuToGraFS documentation](../README.md#documentation).*

## Custom building blocks (SBUs)

SBUs are defined in (multi-)XYZ files. Connection points are dummy atoms with
the symbol `X`; the comment line carries the name:

```text
5
name=My_Tetrahedral pbc="F F F"
Si         0.0000        0.0000        0.0000
X          1.0000        1.0000        1.0000
X          1.0000       -1.0000       -1.0000
X         -1.0000        1.0000       -1.0000
X         -1.0000       -1.0000        1.0000
```

```python
mofgen = Autografs(xyzfile="my_sbus.xyz")    # or: autografs --xyz my_sbus.xyz
```

Rules of thumb:

- **One dummy per bond to a neighboring SBU.** Place each `X` roughly where
  the neighboring block's anchor atom will sit, i.e. along the outgoing bond
  direction from the atom that carries the connection (the *anchor*). During
  the build, the dummy is removed and a bond is created between the two anchor
  atoms it paired.
- **Directions matter, distances don't.** Compatibility and alignment use only
  the unit vectors from the dummy centroid to each dummy; the cell is sized
  from covalent radii, not from your dummy distances.
- Several blocks can live in one file (standard multi-XYZ concatenation); each
  needs a `name=...` in its comment line.
- Custom SBUs with the same name as bundled ones override them; otherwise the
  two libraries merge. The sieve, wizard, and builder treat custom SBUs
  exactly like bundled ones.
- A block is usable on any slot with the same number of connections and a
  matching connection-vector shape — point-group symmetry is diagnostic
  metadata, not a requirement.

A ready-made way to obtain custom SBUs is to
[deconstruct or harvest](deconstruction.md) them from existing structures;
`write_xyz` emits exactly this format.

## Custom topologies

The topology library is a **versioned JSON format** (diffable, safe to share —
unlike pickles, loading it cannot execute code, and it survives pymatgen
upgrades). The bundled library covers the
[RCSR](http://rcsr.anu.edu.au/) database; to regenerate it or convert your own
nets:

```bash
autografs-topologies --use_rcsr -o topologies.json.gz
autografs-topologies -i my_nets.cgd -o my_topologies.json.gz
```

```python
mofgen = Autografs(topofile="my_topologies.json.gz")
```

Input is the [CGD format](http://rcsr.anu.edu.au/help/cgd) used by RCSR and
Systre: a crystal record with a space group (or plane group, for layer nets)
and the asymmetric unit's vertices and edge centers. The converter expands the
symmetry, extracts one slot per vertex/edge with dummy atoms marking the
connections, and groups slots into crystallographic orbits — two slots with
the same local shape but different orbits remain independently mappable.

Programmatic (de)serialization lives in `autografs.topology_io`:

```python
from autografs.topology_io import load_topologies, save_topologies

library = load_topologies("topologies.json.gz")   # lazy: entries materialize
save_topologies(dict(library), "copy.json.gz")    # on first access
```

Legacy dill pickles (`.pkl`) still load, with a warning — convert them once
with `save_topologies` and forget about them.
