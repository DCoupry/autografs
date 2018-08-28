AuToGraFS
=========

Original publication: `"Automatic Topological Generator for Framework Structures"`__.

.. _here: http://pubs.acs.org/doi/abs/10.1021/jp507643v 

__ here_

*This version is under active development*. Bug hunting is very much going on, and there are still some old funtionalities that are not yet reimplemented.

TODO:
-----

Medium priority:

- the symmetry axes detection has trouble with planar shapes
- implement pillaring by using custom alignement rules
- more robust mmanalysis module

Low priority:

- unit testing coverage is non-existant
- documentation writing
- better handling of the databases:
  + precomputing sbu-topologies correspondance
  + use better format than pickle

Install:
--------

.. highlight:: bash

$ pip install --user --upgrade AuToGraFS


For a manual install, first install the dependencies,

1. python >=3.4
2. ase, scipy, numpy<1.15.0


then clone this repository and add it to you pythonpath::

	$ cd $HOME
	$ git clone https://github.com/DCoupry/autografs.git
	$ export PYTHONPATH=$HOME/autografs:$PYTHONPATH`


Examples:
---------

From any python script or command line:

.. highlight:: python

>>> from autografs import Autografs
>>> mofgen = Autografs()
>>> mof = mofgen.make(topology_name="pcu", 
>>>                   sbu_names=["Zn_mof5_octahedral", "Benzene_linear"])
>>> mof.write()

Custom databases can be accessed by passing the path during instanciation

>>> mofgen = Autografs(topology_path="my_topo_path",sbu_path="my_sbu_path")
>>> mof = mofgen.set_topology(topology_name=custom_topology_name,sbu_name=custom_sbu_names)

When looping over both SBU and topologies, it is better to set the topology directly
(here, my_topologies and my_sbu_names are appropriate dummy colletions)

>>> for topologi_name in my_topology_names:
>>>     mofgen.set_topology(topology_name=topology_name)
>>>     for sbu_names in my_sbu_names:
>>>          mof = mofgen.make(sbu_names=sbu_names)

It is possible to pass more than one SBU of each shape, optionally with an associated probabilistic weight.
This weight defaults to 1.0/(number of similar sbu).

>>> mof = mofgen.make(topology_name="pcu", 
>>>                   sbu_names=["Zn_mof5_octahedral", ("Benzene_linear",2.0),("Acetylene_linear",0.5)])
>>> mof.write()

This is particularly helpful in combination with an initial supercell for statistically introducing defects.

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

AAuToGraFS is also aware of topologically equivalent positions, and can generate multi components frameworks
with minimal effort.

>>> sbu_dicts = mofgen.list_available_frameworks()
>>> for sbu_dict in sbu_dicts:
>>>     mof = mofgen.make(sbu_dict=sbu_dict)
>>>     mof.view()

A useful utility is the Atom typer, which assigns bond orders and UFF atom types to a structure:

>>> from autografs.mmanalysis import analyze_mm
>>> bonds, types = analyze_mm(sbu=mofgen.sbu["Zn_mof5_octahedral"])

