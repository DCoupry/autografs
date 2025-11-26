Getting Started
===============

Installation
------------

From PyPI
~~~~~~~~~

.. code-block:: bash

    pip install AuToGraFS

From Source
~~~~~~~~~~~

For a development installation:

.. code-block:: bash

    git clone https://github.com/DCoupry/autografs.git
    cd autografs
    pip install -e ".[dev]"

Requirements
------------

- Python >= 3.8
- ase >= 3.22.0
- scipy >= 1.7.0
- numpy >= 1.20.0
- networkx >= 2.6.0
- pandas >= 1.3.0
- pymatgen >= 2022.0.0

Basic Usage
-----------

The main entry point is the ``Autografs`` class:

.. code-block:: python

    from autografs import Autografs
    
    # Initialize the generator
    mofgen = Autografs()
    
    # Generate a MOF structure
    mof = mofgen.make(
        topology_name="pcu", 
        sbu_names=["Zn_mof5_octahedral", "Benzene_linear"]
    )
    
    # Write the structure to a file
    mof.write()

Custom Databases
----------------

You can use custom topology and SBU databases:

.. code-block:: python

    mofgen = Autografs(
        topology_path="my_topo_path",
        sbu_path="my_sbu_path"
    )
