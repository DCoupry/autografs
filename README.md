AuToGraFS
=========

[Automatic Topological Generator for Framework Structures](http://pubs.acs.org/doi/abs/10.1021/jp507643v "Original publication")

Complete manual coming very soon...

Dependencies to install first

1. python 2.7
2. ase
3. pandas

Clone this repository and add it to you pythonpath:
```
cd $HOME
git clone https://github.com/DCoupry/AuToGraFS.git
export PYTHONPATH=$HOME/AuToGraFS:$PYTHONPATH
```

then from any python script or command line:
```
>>> from autografs import *
>>> mofgen = Autografs(path="$HOME/AuToGraFS/autografs/database")
>>> mof = mofgen.make(label="IRMOF-5", topology="pcu", center="Zn_mof5_octahedral", linker="Benzene_linear")
>>> mof.view()
```
A useful utility is the Atomtyper, which assigns bond orders and UFF atom types to a structure:
```
>>> from atomtyper import *
>>> typer = MolTyper(structure=mof.get_atoms(clean=True))
>>> typer.type_mol(library="uff4mof.csv", reference_library="rappe.csv")
>>> print typer.get_mmtypes()
```
Both the Atomtyper and the AuToGraFS are available as CLI tools:
``` 
python -m "autografs"
python -m "atomtyper" -f mof.cif
```
