from ase.io import read
from pointgroup import PointGroup
import os

def run(tol=0.3):
	cwd = os.path.dirname(__file__)
	for symm in ["C3h","Ci","D3d","D3h","D5d","Oh","Td"]:
		print("testing {0}".format(symm))
		molpath = os.path.join(cwd,"test/{0}.xyz".format(symm))
		logpath = os.path.join(cwd,"test/{0}.log".format(symm))
		mol = read(molpath)
		pg = PointGroup(mol,tol=tol)#,out=open(logpath,"w"))
		if pg.schoenflies==symm:
			print("Test {} passed".format(symm))
		else:
			print("Test {} failed".format(symm))
		print(pg.schoenflies)

if __name__ == "__main__":
	run(0.1)