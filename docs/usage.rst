Usage Guide
===========

Basic Framework Generation
--------------------------

From any Python script or command line:

.. code-block:: python

    from autografs import Autografs
    
    mofgen = Autografs()
    mof = mofgen.make(
        topology_name="pcu", 
        sbu_names=["Zn_mof5_octahedral", "Benzene_linear"]
    )
    mof.write()

Looping Over Topologies and SBUs
--------------------------------

When looping over both SBU and topologies, it is better to set the topology directly:

.. code-block:: python

    for topology_name in my_topology_names:
        mofgen.set_topology(topology_name=topology_name)
        for sbu_names in my_sbu_names:
            mof = mofgen.make(sbu_names=sbu_names)

Probabilistic SBU Selection
---------------------------

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

Introducing Defects
-------------------

Probabilistic SBU selection is particularly helpful in combination with an initial 
supercell for statistically introducing defects:

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
--------------------

Supercells can also be generated post-alignment, carrying everything done before:

.. code-block:: python

    supercell_6x6x6 = mof.get_supercell(m=2)
    supercell_6x6x6.write()

Direct Modifications
--------------------

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
-----------------------------------------

Methods are available for rotation, functionalization, and flipping:

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

Monitoring Bonding and Types
----------------------------

At any moment, you can monitor the bonding matrix and mmtypes, or get a cleaned 
version without dummies:

.. code-block:: python

    from ase.visualize import view
    
    # With the dummies included
    mmtypes = mof.get_mmtypes()
    bonds = mof.get_bonds()
    
    # Without the dummies
    atoms, bonds, mmtypes = mof.get_atoms(dummies=False)
    view(atoms)

Direct Slot Mapping
-------------------

If you know the shape of each slot and its index within the topology, you can 
directly pass a dictionary mapping the SBU to a particular slot:

.. code-block:: python

    # Method to investigate the topology shapes and slots
    topology = mofgen.get_topology(topology_name="pcu")
    sbu_dict = {}
    for slot_index, slot_shape in topology.shapes.items():
        # Do something to choose an SBU
        ...
        sbu_dict[slot_index] = "chosen_sbu_name"
    
    # Now pass it directly
    mof = mofgen.make(topology_name="pcu", sbu_dict=sbu_dict)
    mof.write()

Accessing Databases
-------------------

You can access the databases as dictionaries:

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
--------------------------

AuToGraFS is aware of topologically equivalent positions and can generate 
multi-component frameworks with minimal effort:

.. code-block:: python

    sbu_dicts = mofgen.list_available_frameworks()
    for sbu_dict in sbu_dicts:
        mof = mofgen.make(sbu_dict=sbu_dict)
        mof.view()

Atom Typing Utility
-------------------

A useful utility is the Atom typer, which assigns bond orders and UFF atom types 
to a structure:

.. code-block:: python

    from autografs.mmanalysis import analyze_mm
    bonds, types = analyze_mm(sbu=mofgen.sbu["Zn_mof5_octahedral"])
