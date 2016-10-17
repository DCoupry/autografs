AuToGraFS
=========

Automatic Topological Generator for Framework Structures

Manual coming very soon...

Dependencies to install first::
1. python 2.7
2. ase
3. pandas

Clone this repository and add it to you pythonpath:
```
cd $HOME
git clone https://github.com/DCoupry/AuToGraFS.git
export PYTHONPATH=$HOME/AuToGraFS/autografs
```

then from any python script or command line:
```
>>> from autografs import *
>>> mofgen = Autografs(path="$HOME/AuToGraFS/autografs/database")
>>> mof = mofgen.make(label="IRMOF-5", topology="pcu", center="Zn_mof5_octahedral", linker="Benzene_linear")
>>> mof.view()
```

