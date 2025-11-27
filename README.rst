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

Original publication: `"Automatic Topological Generator for Framework Structures" <http://pubs.acs.org/doi/abs/10.1021/jp507643v>`_.

*This version is under active development*. Bug hunting is very much going on, and there are still some old functionalities that are not yet reimplemented.

Installation
------------

.. code-block:: bash

    pip install AuToGraFS

For a development install:

.. code-block:: bash

    git clone https://github.com/DCoupry/autografs.git
    cd autografs
    pip install -e ".[dev]"

Requirements
------------

- Python >= 3.13
- ase, scipy, numpy, networkx, pandas, pymatgen, dill

Examples
--------

From any Python script or command line:

.. code-block:: python

    from autografs import Autografs
    
    mofgen = Autografs()
    mof = mofgen.make(
        topology_name="pcu", 
        sbu_names=["Zn_mof5_octahedral", "Benzene_linear"]
    )
    mof.write()

Custom databases can be accessed by passing the path during instantiation:

.. code-block:: python

    mofgen = Autografs(topology_path="my_topo_path", sbu_path="my_sbu_path")

When looping over both SBU and topologies, it is better to set the topology directly:

.. code-block:: python

    for topology_name in my_topology_names:
        mofgen.set_topology(topology_name=topology_name)
        for sbu_names in my_sbu_names:
            mof = mofgen.make(sbu_names=sbu_names)

Probabilistic SBU Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to pass more than one SBU of each shape, optionally with an associated 
probabilistic weight. This weight defaults to ``1.0 / (number of similar SBU)``.

.. code-block:: python

    mof = mofgen.make(
        topology_name="pcu", 
        sbu_names=[
            "Zn_mof5_octahedral", 
            ("Benzene_linear", 2.0),
            ("Acetylene_linear", 0.5)
        ]
    )
    mof.write()

This is particularly helpful in combination with an initial supercell for statistically 
introducing defects:

.. code-block:: python

    mof = mofgen.make(
        topology_name="pcu", 
        sbu_names=[
            ("Zn_mof5_octahedral", 2.0),
            ("defect_octahedral", 0.5), 
            "Benzene_linear"
        ],
        supercell=(3, 3, 3)
    )
    mof.write()

Supercell Generation
~~~~~~~~~~~~~~~~~~~~

Supercells can also be generated post-alignment:

.. code-block:: python

    supercell_6x6x6 = mof.get_supercell(m=2)
    supercell_6x6x6.write()

Direct Modifications
~~~~~~~~~~~~~~~~~~~~

Defects and modifications can be introduced at any time directly:

.. code-block:: python

    # Get the site directly
    sbu = mof[7]
    
    # Change all hydrogens to Fluorine
    atoms = sbu.atoms.copy()
    symbols = atoms.get_chemical_symbols()
    symbols = [s if s != "H" else "F" for s in symbols]
    atoms.set_chemical_symbols(symbols)
    
    # By setting the atoms back, mmtypes and bonding are updated
    sbu.set_atoms(atoms=atoms, analyze=True)
    
    # Delete another SBU. H will cap the dangling bits.
    del mof[8]
    mof.write()

Rotation, Functionalization, and Flipping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Rotate SBU 7 by 45 degrees
    mof.rotate(index=7, angle=45.0)
    
    # If a C* axis or reflection plane is detected in SBU 8, flip around it
    mof.flip(index=8)
    
    # Replace all functionalizable H sites with NH2
    nh2 = mofgen.sbu["NH2_point_group"]
    sites = mof.list_functionalizable_sites(symbol="H")
    for site in sites:
        mof.functionalize(where=site, fg=nh2)
    mof.write()

Monitoring Bonds and Types
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ase.visualize import view
    
    # With the dummies included
    mmtypes = mof.get_mmtypes()
    bonds = mof.get_bonds()
    
    # Without the dummies
    atoms, bonds, mmtypes = mof.get_atoms(dummies=False)
    view(atoms)

Direct Slot Mapping
~~~~~~~~~~~~~~~~~~~

If you know the shape of each slot and its index within the topology:

.. code-block:: python

    topology = mofgen.get_topology(topology_name="pcu")
    sbu_dict = {}
    for slot_index, slot_shape in topology.shapes.items():
        # Do something to choose an SBU
        sbu_dict[slot_index] = "chosen_sbu_name"
    
    mof = mofgen.make(topology_name="pcu", sbu_dict=sbu_dict)
    mof.write()

Accessing Databases
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    sbudict = mofgen.sbu
    topodict = mofgen.topologies

Or using tools to find compatible objects:

.. code-block:: python

    sbu_list = mofgen.list_available_sbu(topology_name="pcu")
    topology_list = mofgen.list_available_topologies(
        sbu_names=["Zn_mof5_octahedral", "Benzene_linear"]
    )

Multi-Component Frameworks
~~~~~~~~~~~~~~~~~~~~~~~~~~

AuToGraFS is aware of topologically equivalent positions and can generate 
multi-component frameworks:

.. code-block:: python

    sbu_dicts = mofgen.list_available_frameworks()
    for sbu_dict in sbu_dicts:
        mof = mofgen.make(sbu_dict=sbu_dict)
        mof.view()

Atom Typing Utility
~~~~~~~~~~~~~~~~~~~

A useful utility assigns bond orders and UFF atom types to a structure:

.. code-block:: python

    from autografs.mmanalysis import analyze_mm
    bonds, types = analyze_mm(sbu=mofgen.sbu["Zn_mof5_octahedral"])

License
-------

MIT License - see LICENSE.txt for details.

