# Building frameworks (Python API)

*Part of the [AuToGraFS documentation](../README.md#documentation).*

For the mechanics behind these calls — the sieve, alignment, and cell
optimization — see [How it works](internals.md).

## Exploring the libraries

```python
from autografs import Autografs

mofgen = Autografs()

mofgen.list_topologies()                        # all RCSR symbols
mofgen.list_topologies(sieve="Benzene_linear")  # nets this SBU fits

# building units compatible with a net, grouped by slot type;
# slot types with no compatible SBU are absent from the dict
mofgen.list_building_units(sieve="srs")

# raw access
sbu = mofgen.sbu["Zn_mof5_octahedral"]          # a Fragment
topology = mofgen.topologies["tbo"]             # a Topology (lazy library)

print(len(topology))                  # number of slots
print(topology.cell.abc)              # blueprint cell
print(topology.spacegroup_number)     # 225
print(topology.is_2d)                 # False
for slot_type, indices in topology.mappings.items():
    print(slot_type, "fills slots", indices)
```

## Building

`build` takes a topology and a mapping from slot types (or explicit slot
indices) to SBUs, given as library names or `Fragment` objects:

```python
mof = mofgen.build(
    topology,
    mappings={node_type: "Zn_mof5_octahedral", edge_type: "Benzene_linear"},
    refine_cell=True,   # optimize cell parameters (default)
    max_rmsd=0.3,       # reject builds with bad shape matches
    min_distance=1.0,   # reject builds with overlapping atoms
)
```

- **`max_rmsd`** gates the *directional* mismatch between an SBU's connection
  vectors and its slot's (dimensionless; 0 is a perfect shape match).
  Incompatible geometry raises `autografs.AlignmentError` instead of returning
  a distorted structure.
- **`min_distance`** screens the built structure: if any two non-bonded atoms
  (all periodic images included) are closer than this many Å,
  `autografs.OverlapError` is raised instead of returning overlapping or
  interpenetrating output. The same check is available on any result as
  `Framework.min_contact()`.
- **Slot indices** (integers) may be used as mapping keys to place a specific
  SBU on a specific slot, overriding the slot-type choice:

```python
mappings = {node_type: "sbu_A", edge_type: "linker_1", 7: "linker_2"}
```

## Batch enumeration

`build_all` attempts every compatible SBU combination on every (or a subset
of) topology:

```python
frameworks = mofgen.build_all(
    topology_subset=["pcu", "dia", "srs"],
    sbu_subset=None,          # default: whole SBU library
    max_rmsd=0.3,
    min_distance=1.0,
    max_per_topology=50,      # cap the combinatorial explosion...
    seed=42,                  # ...with a reproducible random sample
    n_jobs=4,                 # parallel builds (near-linear speedup)
)
```

Failed builds are counted and skipped, not raised. Multinodal nets have
combinatorially many SBU choices; when the full product exceeds
`max_per_topology`, a seeded sample of distinct combinations is built instead.
Passing `max_per_topology=-1` (or `None`, the default) disables the cap and
enumerates exhaustively. Either way the total combination count over all
buildable topologies is logged before the first build starts, so an
exhaustive run can be cost-estimated — and aborted — while still cheap.

Long runs can checkpoint: with `checkpoint_dir="ckpt/"` every finished
combination is recorded on disk atomically (built frameworks as `.json.gz`,
gate rejections as `.failed` markers), and rerunning with the same arguments
and the same directory skips them, reloading their outcomes instead of
rebuilding — an interrupted enumeration resumes where it stopped. The ledger
records outcomes, not settings, so rerun with the same gates and seed.

## Working with the result

`build` returns a `Framework`:

```python
mof.structure          # pymatgen Structure (wrapped; site props: tags, ufftype)
mof.graph              # networkx bond graph: symbols, coords, UFF4MOF
                       # atom types, bond orders, tags (source of truth)
mof.formula            # 'Zn4 H12 C24 O13'
mof.lattice            # pymatgen Lattice
mof.bonds              # [(i, j, bond_order), ...]
mof.mmtypes            # UFF4MOF atom types, node order
mof.min_contact()      # closest non-bonded contact (all images)
mof.slots              # {slot index: SBU name} for every placed unit

mof.density                     # g/cm3 (MOF-5: ~0.6)
mof.void_fraction()             # geometric; probe_radius=1.2 for a He probe
mof.largest_cavity_diameter()   # LCD, Angstrom (MOF-5: ~15)
mof.pore_limiting_diameter()    # PLD, Angstrom (MOF-5: ~8) - the channel bottleneck

mof.write_cif("out.cif", symprec=None)   # symprec symmetrizes if set
atoms = mof.to_ase()                     # periodic ase.Atoms
mof.view()                               # ASE viewer
gulp_input = mof.to_gulp()               # UFF4MOF optimization input for GULP
```

CIF export is for downstream tools; it loses the bond graph and the
per-atom provenance that post-build editing needs. To keep a framework
editable across sessions, save it to the versioned JSON format instead:

```python
mof.save("mof5.json.gz")                 # bond graph, tags, provenance, energy
mof = Framework.load("mof5.json.gz")     # editable exactly like the original
defective = mof.supercell(2).defects(fraction=0.1, seed=42)
```

See [Editing](editing.md) for supercells, defects, and functionalization.

## UFF4MOF relaxation

`relax` optimizes the geometry and cell with the UFF4MOF force field through
LAMMPS, in-process, and returns a new `Framework` with the same bond graph:

```bash
pip install "autografs[relax]"
```

```python
relaxed = mof.relax()          # UFF4MOF, alternating cell + FIRE
relaxed.energy                 # kcal/mol per unit cell
relaxed.write_cif("relaxed.cif")
```

Cells smaller than the non-bonded cutoff (12.5 Å by default) are relaxed as an
internal supercell and folded back transparently. `"UFF"` and `"Dreiding"` are
also accepted as `force_field`.

## Error handling

All library exceptions derive from `autografs.AutografsError`:

```python
from autografs import AlignmentError, OverlapError
from autografs.exceptions import StackingError, RelaxationError

try:
    mof = mofgen.build(topology, mappings, max_rmsd=0.3, min_distance=1.0)
except AlignmentError:   # shape mismatch beyond the gate
    ...
except OverlapError:     # non-bonded contact below the gate
    ...
```
