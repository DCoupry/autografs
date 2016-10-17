from autografs import *
from atomtyper import *

m=Autografs(path="database")
mof=m.make(label="IRMOF-5", topology="pcu", center="Zn_mof5_octahedral", linker="Benzene_linear")
typer = MolTyper(structure=mof.get_atoms(clean=True))
typer.type_mol(library="uff4mof.csv", reference_library="rappe.csv")
print typer.get_mmtypes()
mof.view()
