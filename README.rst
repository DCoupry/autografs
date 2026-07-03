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

- **2464 RCSR topologies** ship with the package - it works out of the
  box, no database generation step.
- **Geometric matching**: SBUs are matched to topology slots by
  optimally rotating their connection vectors (proper rotations only,
  so chiral building blocks are never silently mirrored). Point-group
  labels are metadata, not gates, which makes low-symmetry vertices
  usable.
- **Physically meaningful cells**: the cell is optimized so that
  bonded connection points coincide. The MOF-5 prototype (pcu + Zn4O +
  benzenedicarboxylate) comes out cubic at 12.8 Angstrom against the
  experimental 12.9.
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
    # Framework('pcu', 'Zn4 H12 C24 O13', abc=(12.77, 12.77, 12.77))

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
    )

- ``max_rmsd`` gates the *directional* mismatch between an SBU's
  connection vectors and its slot's (dimensionless; 0 is a perfect
  shape match). Incompatible geometry raises
  ``autografs.AlignmentError`` instead of returning a distorted
  structure.
- Slot indices (integers) may be used as mapping keys to place a
  specific SBU on a specific slot, overriding the slot-type choice.

To enumerate every compatible combination:

.. code-block:: python

    frameworks = mofgen.build_all(
        topology_subset=["pcu", "dia", "srs"],
        max_rmsd=0.3,
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
