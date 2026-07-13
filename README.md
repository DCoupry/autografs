# AuToGraFS

[![PyPI](https://img.shields.io/pypi/v/AuToGraFS.svg)](https://pypi.org/project/AuToGraFS/)
[![Python](https://img.shields.io/pypi/pyversions/AuToGraFS.svg)](https://pypi.org/project/AuToGraFS/)
[![CI](https://github.com/DCoupry/autografs/actions/workflows/ci.yml/badge.svg)](https://github.com/DCoupry/autografs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/DCoupry/autografs/branch/master/graph/badge.svg)](https://codecov.io/gh/DCoupry/autografs)
[![License](https://img.shields.io/github/license/DCoupry/autografs.svg)](https://github.com/DCoupry/autografs/blob/master/LICENSE.txt)

**AuToGraFS** — the *Automatic Topological Generator for Framework Structures* —
generates Metal-Organic Frameworks (MOFs), Covalent Organic Frameworks (COFs)
and other periodic framework materials by mapping molecular building blocks
(SBUs) onto topological blueprints (nets). It also runs the pipeline in
reverse: deconstruct an experimental structure back into building blocks and
identify its net.

```python
from autografs import Autografs

mofgen = Autografs()
pcu = mofgen.topologies["pcu"]
mof = mofgen.build(pcu, mappings={
    slot: "Zn_mof5_octahedral" if len(slot.atoms.indices_from_symbol("X")) == 6
    else "Benzene_linear"
    for slot in pcu.mappings
})
mof.write_cif("mof5.cif")   # cubic, a = 12.89 A (experiment: 12.9)
```

Original publication: [*"Automatic Topological Generator for Framework
Structures"*](http://pubs.acs.org/doi/abs/10.1021/jp507643v),
Addicoat, Coupry & Heine, *J. Phys. Chem. A* 2014, 118 (40), 9607.

## Why AuToGraFS?

Several excellent framework assemblers exist — notably
[PORMAKE](https://github.com/Sangwon91/PORMAKE) (Lee *et al.*, whose MIT-licensed
building-block library AuToGraFS gratefully bundles) and
[ToBaCCo](https://github.com/tobacco-mofs/tobacco_3.0) (Colón / Gómez-Gualdrón
groups). AuToGraFS was one of the first tools in this space (2014) and version 3
is a ground-up rewrite around a small set of design choices:

- **Works out of the box.** `pip install AuToGraFS` ships **2686 RCSR
  topologies** and **930 building blocks** (63 curated SBUs + the 867-block
  PORMAKE library, 2- to 24-connected). No database generation step, no
  external binaries for building. **96.5 % of the shipped topologies are
  buildable immediately** (see [library coverage](docs/coverage.md)).
- **2D nets are first-class.** The 200 RCSR layer nets (hcb, sql, kgm, ...)
  that COF chemistry builds on are stored as plane-group topologies; a build
  produces a flat layer, and `Framework.stack()` turns it into a bulk crystal
  with AA / AB / serrated / staggered stacking at a chosen interlayer spacing.
- **Geometry, not symmetry tables.** SBUs are matched to slots by optimally
  rotating their connection vectors onto the slot's (Hungarian assignment +
  Kabsch, *proper rotations only* — chiral building blocks are never silently
  mirrored). Point-group labels are metadata, not gates, so low-symmetry (C1)
  vertices stay usable.
- **Physically meaningful cells.** The cell is optimized so that every
  inter-SBU bond sits at its covalent (Cordero) bond length, with the crystal
  system's constraints enforced (a cubic net optimizes a single length).
  MOF-5 comes out cubic at 12.89 Å against the experimental 12.9 — *before*
  any force-field relaxation.
- **Fails loudly, never silently.** Optional hard gates (`max_rmsd`,
  `min_distance`) raise typed exceptions instead of returning distorted or
  interpenetrating structures. Identical inputs give identical outputs.
- **Runs in reverse.** Deconstruct a CIF (MOF or COF) into library-ready SBUs
  and identify its net; harvest a whole SBU library from a folder of
  structures.
- **Post-processing built in.** UFF4MOF assignment on every output, GULP input
  generation, and one-call in-process LAMMPS relaxation
  (`pip install "autografs[relax]"`).
- **Safe, versioned data formats.** Topology libraries are plain JSON
  (diffable, shareable, survives pymatgen upgrades) — not pickles.
- **A guided CLI.** The `autografs` wizard walks through
  topology → SBUs → build → stack → export without writing a script.
- **MIT licensed**, pure-Python installation on Linux / macOS / Windows.

If you need features AuToGraFS doesn't have yet (see [Roadmap](#roadmap)),
PORMAKE and ToBaCCo are actively maintained and may fit better — comparisons
age quickly, so evaluate against their current versions.

## Installation

```bash
pip install AuToGraFS
```

Requires Python ≥ 3.11. Core dependencies: pymatgen, ASE, numpy, scipy,
networkx.

Optional extras:

```bash
pip install "autografs[relax]"   # UFF4MOF relaxation via LAMMPS
```

On Windows, the LAMMPS wheel additionally needs the Microsoft MPI runtime
(`winget install Microsoft.MSMPI`).

Development install:

```bash
git clone https://github.com/DCoupry/autografs.git
cd autografs
pip install -e ".[dev]"
```

## Quickstart: MOF-5

```python
from autografs import Autografs

mofgen = Autografs()

# what fits the pcu net?
available = mofgen.list_building_units(sieve="pcu")
for slot_type, sbu_names in available.items():
    print(slot_type, len(sbu_names), "candidates")
# Oh 6 : ... candidates      (the octahedral node)
# D*h 2 : ... candidates     (the linear edge)

# pick one SBU per slot type
topology = mofgen.topologies["pcu"]
mappings = {}
for slot_type in topology.mappings:
    n_connections = len(slot_type.atoms.indices_from_symbol("X"))
    if n_connections == 6:
        mappings[slot_type] = "Zn_mof5_octahedral"
    else:
        mappings[slot_type] = "Benzene_linear"

mof = mofgen.build(topology, mappings=mappings)
print(mof)
# Framework('pcu', 'Zn4 H12 C24 O13', abc=(12.89, 12.89, 12.89))

mof.write_cif("mof5.cif")
```

Then keep going: [build options and batch enumeration](docs/building.md),
[2D COFs](docs/cofs-and-stacking.md), [editing](docs/editing.md), or
[deconstruction](docs/deconstruction.md).

## Command line

No script needed — the `autografs` wizard covers build, deconstruct, stack, and
export interactively, and `autografs-topologies` (re)builds a topology library:

```bash
autografs                                    # bundled libraries; guided session
autografs --xyz my_sbus.xyz                  # add custom building blocks
autografs-topologies --use_rcsr -o topologies.json.gz   # regenerate the library
```

Full walkthrough of both commands: [Command line](docs/cli.md).

## Documentation

The README is the overview; the depth lives in `docs/`:

- **[Building frameworks](docs/building.md)** — the Python API: exploring the
  libraries, `build`, `build_all`, working with the `Framework` result
  (porosity, save/load, exports), UFF4MOF relaxation, error handling.
- **[2D COFs and stacking](docs/cofs-and-stacking.md)** — layer nets and
  turning a layer into a bulk crystal (AA / AB / serrated / staggered).
- **[Editing](docs/editing.md)** — editing SBUs before a build, and post-build
  supercells, statistical defects, and functionalization.
- **[Deconstruction](docs/deconstruction.md)** — CIF → SBUs + net, net
  identification, interpenetration, COFs, and batch harvesting.
- **[Command line](docs/cli.md)** — the `autografs` wizard and
  `autografs-topologies` in full.
- **[Extending the libraries](docs/extending.md)** — custom SBUs (XYZ) and
  custom topologies (CGD → JSON).
- **[How it works & architecture](docs/internals.md)** — the 2014 idea, the
  build pipeline, and a module map of the codebase.
- **[Library coverage](docs/coverage.md)** — what fraction of RCSR is buildable
  today, and the nets that aren't.

## FAQ

**Does it build COFs?** Yes — 2D layer nets are first-class. Build a flat layer
on a layer net (hcb, sql, ...), then `Framework.stack()` turns it into a bulk
crystal. See [2D COFs and stacking](docs/cofs-and-stacking.md).

**Do I need LAMMPS?** Only for `Framework.relax()`. Building, export,
deconstruction, and everything else are pure-Python. Install the relaxation
backend with `pip install "autografs[relax]"`.

**My SBU is rejected / I get `AlignmentError` — why?** The build gate is
geometric: the SBU's connection-vector *shape* doesn't match the slot's within
`max_rmsd`. Raise `max_rmsd`, pick a net whose vertex figure fits, or edit the
SBU. Point-group symmetry is diagnostic metadata, not the gate.

**Is my net in the library?** 2686 RCSR nets ship; list them with
`mofgen.list_topologies()`. 96.5 % are buildable out of the box — the rest, and
how to enable them, are in [library coverage](docs/coverage.md).

**`deconstruct()` raised `DeconstructionError` — why?** It refuses rod /
1-periodic (chain) building units, disordered structures, and molecular
crystals (no periodic framework to analyze). Metal-free COFs *are* supported.
See [Deconstruction](docs/deconstruction.md).

**CIF or JSON — which should I save?** CIF for downstream tools, but it loses
the bond graph and per-atom provenance that post-build editing needs. Use
`Framework.save()` / `Framework.load()` (versioned JSON) to keep a framework
editable across sessions.

**Can I still load my old `.pkl` topology library?** Yes, with a warning.
Convert it once with `autografs.topology_io.save_topologies` to the JSON format
and keep that.

## Development

```bash
pip install -e ".[dev]"

pytest                    # slow tests auto-skipped
pytest -m slow            # only the slow tests
ruff check src tests      # lint
ruff format src tests     # format
mypy src/autografs        # type check
```

Tests live in `tests/`; `scripts/` holds the coverage audit
(`sbu_coverage.py`), the PORMAKE import (`import_pormake_bbs.py`), and fixture
generators. Architecture and module map: [How it works](docs/internals.md).

## Roadmap

The 3.x line has reached feature parity with 2.x and added the inverse pipeline
(deconstruction, net identification, and SBU harvesting for MOFs and COFs).
Remaining directions — rod-MOF (1-periodic SBU) support, IZA zeolite import,
and curated high-connectivity SBU packs for the last uncovered nets — are
tracked in the [issue tracker](https://github.com/DCoupry/autografs/issues) and
in `v3_plan.md` / `progress.md`.

## Citing

If you use AuToGraFS, please cite:

```bibtex
@article{autografs2014,
  title   = {AuToGraFS: Automatic Topological Generator for Framework Structures},
  author  = {Addicoat, Matthew A. and Coupry, Damien E. and Heine, Thomas},
  journal = {The Journal of Physical Chemistry A},
  volume  = {118},
  number  = {40},
  pages   = {9607--9614},
  year    = {2014},
  doi     = {10.1021/jp507643v}
}
```

Related work you may also want to cite:

- **UFF4MOF** (the force field used for typing and relaxation):
  Addicoat, Vankova, Akter & Heine, *J. Chem. Theory Comput.* 2014, 10, 880;
  Coupry, Addicoat & Heine, *J. Chem. Theory Comput.* 2016, 12, 5215.
- **PORMAKE** (if you use the bundled `pormake.xyz` building blocks):
  S. Lee *et al.*, *ACS Appl. Mater. Interfaces* 2021, 13, 23647.
- **RCSR** (the source of the bundled topologies):
  O'Keeffe, Peskov, Ramsden & Yaghi, *Acc. Chem. Res.* 2008, 41, 1782.

## License

MIT License — see [LICENSE.txt](LICENSE.txt).

The bundled building-block library `pormake.xyz` is converted from
[PORMAKE](https://github.com/Sangwon91/PORMAKE) (MIT License, Copyright (c)
2022 Sangwon Lee; see
[PORMAKE_LICENSE.md](src/autografs/data/PORMAKE_LICENSE.md)).
