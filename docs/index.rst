AuToGraFS Documentation
=======================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   usage
   api
   changelog

Automatic Topological Generator for Framework Structures
--------------------------------------------------------

AuToGraFS is a Python library for the automatic generation of framework structures,
particularly Metal-Organic Frameworks (MOFs).

Original publication: `"Automatic Topological Generator for Framework Structures" <http://pubs.acs.org/doi/abs/10.1021/jp507643v>`_

Features
--------

- Generate framework structures from topologies and building blocks
- Support for custom topology and SBU databases
- Supercell generation
- Defect introduction and modifications
- Bond order and atom type analysis

Quick Start
-----------

.. code-block:: python

    from autografs import Autografs
    
    mofgen = Autografs()
    mof = mofgen.make(
        topology_name="pcu", 
        sbu_names=["Zn_mof5_octahedral", "Benzene_linear"]
    )
    mof.write()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
