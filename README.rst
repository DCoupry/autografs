AuToGraFS
=========

Original publication: `"Automatic Topological Generator for Framework Structures"`__.

.. _here: http://pubs.acs.org/doi/abs/10.1021/jp507643v 

__ here_

*This is a beta version*. Bug hunting is very much going on, and there are still some old funtionalities that are not yet reimplemented.

TODO:
-----
- implement pillaring
- more robust mmanalysis module
- unit testing coverage is non-existant
- documentation writing
- better handling of the databases:
  + sql for faster selecting using ase.db module
  + precomputing sbu-topologies correspondance
  + list of paths to custom databases...

Dependencies to install first:
------------------------------

1. python 3.6
2. ase, scipy, numpy

Examples:
---------

.. highlight:: bash

Clone this repository and add it to you pythonpath::

	$ cd $HOME
	$ git clone https://github.com/DCoupry/autografs.git
	$ export PYTHONPATH=$HOME/autografs:$PYTHONPATH`

then from any python script or command line:

.. highlight:: python

>>> from autografs import Autografs
>>> mofgen = Autografs()
>>> mof = mofgen.make(topology_name="pcu", 
>>>                   sbu_names=["Zn_mof5_octahedral", "Benzene_linear"])
>>> mof.write()

It is possible to pass more than one SBU of each shape, optionally with an associated probabilistic weight.
This weight defaults to 1.0/(number of similar sbu).

>>> mof = mofgen.make(topology_name="pcu", 
>>>                   sbu_names=["Zn_mof5_octahedral", ("Benzene_linear",2.0),("Acetylene_linear",0.5)])
>>> mof.write()

This is particularly helpful in combination with an initial supercell for statistically introduce defects.

>>> mof = mofgen.make(topology_name="pcu", 
>>>                   sbu_names=[("Zn_mof5_octahedral",2.0),("defect_octahedral",0.5), "Benzene_linear"],
>>>                   supercell=(3,3,3))
>>> mof.write()

Supercell can also be generated post-alignement, carrying everything done before.

>>> supercell_6x6x6 = mof.get_supercell(m=2)
>>> supercell_6x6x6.write()

Defects and modifications can be introduced at any time directly:

>>> # get the site directly
>>> sbu = mof[7]
>>> # change all hydrogens to Fluorine
>>> atoms = sbu.atoms.copy()
>>> symbols = atoms.get_chemical_symbols()
>>> symbols = [s if s!="H" else "F" for s in symbols]
>>> atoms.set_chemical_symbols(symbols)
>>> # by setting the atoms back, 
>>> # mmtypes and bonding are updated.
>>> sbu.set_atoms(atoms=atoms,analyze=True)
>>> # delete another sbu. H will cap the dangling bits.
>>> del mof[8]
>>> mof.write()

Methods are also available for the rotation, functionalization and flipping.

>>> # rotate the sbu 7 buy 45 degrees
>>> mof.rotate(index=7,angle=45.0)
>>> # if a C* axis or reflection plane is detected
>>> # in the sbu 8 , flip around it
>>> mof.flip(index=8)
>>> # replace all functionalizable H sites with NH2
>>> nh2 = mofgen.sbu["NH2_point_group"]
>>> sites = mof.list_functionalizable_sites(self,symbol="H")
>>> for site in sites:
>>>     mof.functionalize(where=site,fg=nh2)
>>> mof.write()

At any moment, we can monitor the bonding matrix and mmtypes, or get a cleaned version without dummies.

>>> from ase.visualize import view
>>> # with the dummies included
>>> mmtypes = mof.get_mmtypes()
>>> bonds = mof.get_bonds()
>>> # without the dummies
>>> atoms,bonds,mmtypes = mof.get_atoms(dummies=False)
>>> view(atoms)

If you know the shape of each slot and its index within the topology, it is possible to directly pass a dictionary mapping
the SBU to a particular slot.

>>> # method to investigate the topology shapes and slots
>>> topology = mofgen.get_topology(topology_name="pcu")
>>> sbu_dict = {}
>>> for slot_index,slot_shape in topology.shapes.items():
>>>     # do something to choose an sbu
>>>     ...
>>>     sbu_dict[slot_index] = "chosen_sbu_name"
>>> # now pass it directly
>>> mof = mofgen.make(topology_name="pcu", sbu_dict=sbu_dict)
>>> mof.write()

You can access the databases as dictionaries using the following:

>>> sbudict  = mofgen.sbu
>>> topodict = mofgen.topologies

Or using tools to find compatible objects:

>>> sbu_list = mofgen.list_available_sbu(topology_name="pcu")
>>> topology_list = mofgen.list_available_topologies(sbu_names=["Zn_mof5_octahedral", "Benzene_linear"])

A useful utility is the Atom typer, which assigns bond orders and UFF atom types to a structure:

>>> from autografs.mmanalysis import analyze_mm
>>> bonds, types = analyze_mm(sbu=mofgen.sbu["Zn_mof5_octahedral"])

