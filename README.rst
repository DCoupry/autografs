AuToGraFS
=========

|PyPI| |Python| |CI| |codecov| |License| |Black|

.. |PyPI| image:: https://img.shields.io/pypi/v/AuToGraFS.svg
   :target: https://pypi.org/project/AuToGraFS/
   :alt: PyPI version

.. |Python| image:: https://img.shields.io/pypi/pyversions/AuToGraFS.svg
   :target: https://pypi.org/project/AuToGraFS/
   :alt: Python versions

.. |CI| image:: https://github.com/DCoupry/autografs/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/DCoupry/autografs/actions/workflows/ci.yml
   :alt: CI Status

.. |codecov| image:: https://codecov.io/gh/DCoupry/autografs/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/DCoupry/autografs
   :alt: Code Coverage

.. |License| image:: https://img.shields.io/github/license/DCoupry/autografs.svg
   :target: https://github.com/DCoupry/autografs/blob/master/LICENSE.txt
   :alt: License

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

AuToGraFS generates Metal-Organic Frameworks (MOFs), Covalent Organic
Frameworks (COFs) and other periodic framework materials by mapping
molecular building blocks (SBUs) onto topological blueprints.

Original publication: `"Automatic Topological Generator for Framework
Structures" <http://pubs.acs.org/doi/abs/10.1021/jp507643v>`_,
Addicoat, Coupry & Heine, *J. Phys. Chem. A* 2014, 118 (40), 9607.

Highlights of version 3:

- **2686 RCSR topologies** ship with the package - it works out of the
  box, no database generation step. This includes the 200 2D layer
  nets (hcb, sql, kgm, ...) that COF chemistry builds on.
- **930 building blocks**: 63 curated SBUs plus the 867-block PORMAKE
  library (MIT, Sangwon Lee et al.; curated from ToBaCCo and CoRE MOF),
  covering 2- to 24-connected nodes - including the high-connectivity
  metal clusters (7-, 9-, 10-, 12-, 24-c) that nets like rht need.
- **Geometric matching**: SBUs are matched to topology slots by
  optimally rotating their connection vectors (proper rotations only,
  so chiral building blocks are never silently mirrored). Point-group
  labels are metadata, not gates, which makes low-symmetry vertices
  usable.
- **Physically meaningful cells**: the cell is optimized so that
  every inter-SBU bond sits at its covalent bond length. The MOF-5
  prototype (pcu + Zn4O + benzenedicarboxylate) comes out cubic at
  12.89 Angstrom against the experimental 12.9.
- **Deterministic**: identical inputs give identical structures.

Installation
------------

.. code-block:: bash

    pip install AuToGraFS

For a development install:

.. code-block:: bash

    git clone https://github.com/DCoupry/autografs.git
    cd autografs
    pip install -e ".[dev]"

Quickstart: MOF-5
-----------------

.. code-block:: python

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

Exploring the libraries
-----------------------

.. code-block:: python

    # all topology names (RCSR symbols)
    mofgen.list_topologies()

    # topologies compatible with a given building unit
    mofgen.list_topologies(sieve="Benzene_linear")

    # building units compatible with a given topology, grouped by
    # slot type. Slot types with no compatible SBU are absent.
    mofgen.list_building_units(sieve="srs")

    # topology details
    topology = mofgen.topologies["tbo"]
    print(len(topology))                  # number of slots
    print(topology.cell.abc)              # blueprint cell
    print(topology.spacegroup_number)     # 225
    for slot_type, indices in topology.mappings.items():
        print(slot_type, "fills slots", indices)

Building
--------

``build`` takes a topology and a mapping from slot types (or explicit
slot indices) to SBUs, given as library names or ``Fragment`` objects:

.. code-block:: python

    mof = mofgen.build(
        topology,
        mappings={slot_type: "Zn_mof5_octahedral", edge_type: "Benzene_linear"},
        refine_cell=True,   # optimize cell parameters (default)
        max_rmsd=0.3,       # reject builds with bad shape matches
        min_distance=1.0,   # reject builds with overlapping atoms
    )

- ``max_rmsd`` gates the *directional* mismatch between an SBU's
  connection vectors and its slot's (dimensionless; 0 is a perfect
  shape match). Incompatible geometry raises
  ``autografs.AlignmentError`` instead of returning a distorted
  structure.
- ``min_distance`` screens the built structure: if any two non-bonded
  atoms (all periodic images included) are closer than this many
  Angstroms, ``autografs.OverlapError`` is raised instead of returning
  overlapping or interpenetrating output. The same check is available
  on any result as ``Framework.min_contact()``.
- Slot indices (integers) may be used as mapping keys to place a
  specific SBU on a specific slot, overriding the slot-type choice.

To enumerate every compatible combination:

.. code-block:: python

    frameworks = mofgen.build_all(
        topology_subset=["pcu", "dia", "srs"],
        max_rmsd=0.3,
        min_distance=1.0,
    )

Working with the result
-----------------------

``build`` returns a ``Framework``:

.. code-block:: python

    mof.structure          # pymatgen Structure (wrapped, site props: tags, ufftype)
    mof.write_cif("out.cif", symprec=None)   # symprec symmetrizes if set
    atoms = mof.to_ase()   # periodic ase.Atoms
    mof.view()             # ASE viewer

    gulp_input = mof.to_gulp()   # UFF4MOF optimization input for GULP
    mof.graph              # networkx bond graph: symbols, coords,
                           # UFF4MOF atom types, bond orders, tags

UFF4MOF relaxation
------------------

``relax`` optimizes the geometry and cell with the UFF4MOF force
field through LAMMPS, in-process, and returns a new ``Framework``
with the same bond graph:

.. code-block:: bash

    pip install "autografs[relax]"

.. code-block:: python

    relaxed = mof.relax()          # UFF4MOF, alternating cell + FIRE
    relaxed.energy                 # kcal/mol per unit cell
    relaxed.write_cif("relaxed.cif")

Cells smaller than the non-bonded cutoff (12.5 Angstrom by default)
are relaxed as an internal supercell and folded back transparently.
On Windows, the LAMMPS wheel additionally needs the Microsoft MPI
runtime (``winget install Microsoft.MSMPI``).

2D COFs
-------

Layer nets (hcb, sql, kgm, hxl, ...) are stored as 2D plane-group
topologies. A build on one produces a single flat layer in a padded
slab: the in-plane cell is optimized while c stays frozen, since the
interlayer spacing is dispersion-driven chemistry, not topology. The
COF-1 prototype:

.. code-block:: python

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
    cof = layer.stack(mode="AB")                    # two-layer cell,
                                                    # offset (1/3, 2/3)
    cof = layer.stack(mode="serrated", offset=(0.5, 0))
    cof.write_cif("cof1.cif")

``stack`` returns a new ``Framework``: AA keeps one layer per cell
with ``c = interlayer``; AB/serrated/staggered build a two-layer cell
with an in-plane-offset copy. Layers are van-der-Waals stacked (no
inter-layer bonds). The default ``interlayer=3.35`` Angstrom is
graphite-like; typical COFs fall in 3.3-3.6. Stacking a non-layered
framework raises ``autografs.exceptions.StackingError``.

Custom building blocks
----------------------

SBUs are defined in (multi-)XYZ files. Connection points are dummy
atoms with the symbol ``X``; the comment line carries the name:

.. code-block:: text

    5
    name=My_Tetrahedral pbc="F F F"
    Si         0.0000        0.0000        0.0000
    X          1.0000        1.0000        1.0000
    X          1.0000       -1.0000       -1.0000
    X         -1.0000        1.0000       -1.0000
    X         -1.0000       -1.0000        1.0000

.. code-block:: python

    mofgen = Autografs(xyzfile="my_sbus.xyz")

Custom topologies
-----------------

The topology library is a versioned JSON format (safe to share, unlike
pickles). To regenerate it from the RCSR database or convert your own
`CGD files <http://rcsr.anu.edu.au/help/cgd>`_:

.. code-block:: bash

    python scripts/cgd2pkl.py --use_rcsr -o topologies.json.gz
    python scripts/cgd2pkl.py -i my_nets.cgd -o my_topologies.json.gz

.. code-block:: python

    mofgen = Autografs(topofile="my_topologies.json.gz")

Requirements
------------

- Python >= 3.11
- pymatgen, ase, numpy, scipy, networkx, dill

Roadmap
-------

Some 2.x features (functionalization of built frameworks, supercells
with defects, rotation/flipping of placed SBUs) are not yet
reimplemented in the 3.x line. See ``v3_plan.md`` and ``progress.md``
in the repository for the current state and direction.

License
-------

MIT License - see LICENSE.txt for details.

The bundled building-block library ``pormake.xyz`` is converted from
`PORMAKE <https://github.com/Sangwon91/PORMAKE>`_ (MIT License,
Copyright (c) 2022 Sangwon; see ``src/autografs/data/PORMAKE_LICENSE.md``).
If you use those building blocks, please cite
*S. Lee et al., ACS Appl. Mater. Interfaces 2021, 13, 23647*.
