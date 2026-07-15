# Command line

*Part of the [AuToGraFS documentation](../README.md#documentation).*

Two console commands are installed with the package.

## `autografs` — interactive wizard

```bash
autografs                       # bundled libraries
autografs --xyz my_sbus.xyz     # add custom building blocks
autografs --topofile my_topologies.json.gz
```

A guided session covers the whole workflow without writing a script:

- **Build a structure** — filter the nets (3D / 2D, by slot connectivity),
  type-to-search a topology, inspect its cell / symmetry / slot types, pick a
  compatible SBU per slot type (only compatible candidates are offered), set
  the build options, and export. Failed alignments drop into a recovery loop
  (relax the gate, or swap SBUs) instead of dying.
- **Deconstruct a structure** — read a CIF (or anything pymatgen parses),
  recover its building units and net, and either write the harvested SBUs to an
  XYZ file or add them straight into the session library so they are selectable
  in the next build. Expected refusals (rod, disordered, non-framework)
  are reported, not crashes.
- **Browse topologies** — summary table per net plus every compatible SBU per
  slot type.
- **Browse building units** — composition, connectivity, dummy point group,
  and how many nets the SBU fits.
- **Batch build** — a front-end for `build_all`: pick a topology subset, cap
  the combinations per topology (-1 disables the cap), and write every
  resulting CIF to a directory.
- 2D layer builds offer **COF stacking** (AA / AB / serrated / staggered,
  chosen interlayer spacing) before export.
- The final menu is an **edit/export loop**: make a supercell, add statistical
  defects, functionalize sites, rotate a placed linker, interpenetrate
  (catenate), or relax with UFF4MOF — each edit feeds the next — then export.
- Export formats: CIF, GULP input (UFF4MOF optimization), or straight into the
  ASE viewer.

Non-interactive `--topology`/`--sbu` flags are deliberately not provided:
addressing slot types from a flag is ambiguous on nets with several
same-connectivity orbits, and scripted use is what the Python API and
`build_all` are for.

## `autografs-topologies` — topology library generator

```bash
# regenerate the full RCSR library
autografs-topologies --use_rcsr -o topologies.json.gz

# convert your own CGD nets (optionally merged with RCSR)
autografs-topologies -i my_nets.cgd -o my_topologies.json.gz
autografs-topologies -i my_nets.cgd --use_rcsr -o combined.json.gz

# admit vertices above the default 24-connected cap
autografs-topologies --use_rcsr --max-connectivity 32 -o big.json.gz
```

See [Extending the libraries](extending.md#custom-topologies) for the input
format.
