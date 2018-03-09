import os
import sys
import numpy
import _pickle as pickle

import ase
from ase              import Atom, Atoms
from ase.spacegroup   import crystal
from ase.data         import chemical_symbols
from ase.neighborlist import NeighborList


from scipy.cluster.hierarchy import fclusterdata as cluster
from progress.bar            import Bar

import warnings

from autografs.utils.pointgroup import PointGroup
from autografs.utils           import __data__



warnings.filterwarnings("error")

def download_topologies(url="http://rcsr.anu.edu.au/downloads/RCSRnets.cgd"):
    """
    TODO
    """
    
    import requests
    import shutil

    print(__data__)
    root = os.path.join(__data__,"topologies")
    path = os.path.join(root,"nets.cgd")
    resp = requests.get(url, stream=True)
    if resp.status_code == 200:
        print("Successfully downloaded the nets from RCSR.")
        resp.raw.decode_content = True
        with open(path,"wb") as outpt:
            shutil.copyfileobj(resp.raw, outpt)        
    return


def read_topologies(update=False):
    """
    TODO
    """
    
    root = os.path.join(__data__,"topologies")
    topologies = {}
    # we need the names of the groups and their correspondance in ASE spacegroup data
    # this was compiled using Levenshtein distances and regular expressions
    groups_file = os.path.join(root,"HermannMauguin.dat")
    grpf = open(groups_file,"rb")
    groups = {l.split()[0]:l.split()[1] for l in grpf.read().decode("utf8").splitlines()}
    grpf.close()
    # read the rcsr topology data
    topology_file = os.path.join(root,"nets.cgd")
    if ((not os.path.isfile(topology_file)) or (update)):
        download_topologies()
    else:
        print("Using saved nets from RCSR")
    # the script as such starts here
    with open(topology_file,"rb") as tpf:
        text = tpf.read().decode("utf8")
        # split the file by topology
        topologies_raw = [t.strip().strip("CRYSTAL") for t in text.split("END")]
        topologies_len = len(topologies_raw)
        print("{0:<5} topologies before treatment".format(topologies_len))
        bar = Bar('Processing', max=topologies_len)
        for topology_raw in topologies_raw:
            bar.next()
            # read from the template.
            # the edges are easier to comprehend by edge center
            try:
                lines = topology_raw.splitlines()
                lines = [l.split() for l in lines if len(l)>2]
                name = None
                group = None
                cell = []
                symbols = []
                nodes = []
                for l in lines:
                    if l[0].startswith("NAME"):
                        name = l[1].strip()
                    elif l[0].startswith("GROUP"):
                        group = l[1]
                    elif l[0].startswith("CELL"):
                        cell = numpy.array(l[1:], dtype=numpy.float32)
                    elif l[0].startswith("NODE"):
                        this_symbol = chemical_symbols[int(l[2])]
                        this_node = numpy.array(l[3:], dtype=numpy.float32)
                        nodes.append(this_node)
                        symbols.append(this_symbol)
                    elif (l[0].startswith("#") and l[1].startswith("EDGE_CENTER")):
                        # linear connector
                        this_node = numpy.array(l[2:], dtype=numpy.float32)
                        nodes.append(this_node)
                        symbols.append("He")
                    elif l[0].startswith("EDGE"):
                        # now we append some dummies
                        s    = int((len(l)-1)/2)
                        midl = int((len(l)+1)/2)
                        x0  = numpy.array(l[1:midl],dtype=numpy.float32).reshape(-1,1)
                        x1  = numpy.array(l[midl:] ,dtype=numpy.float32).reshape(-1,1)
                        xx  = numpy.concatenate([x0,x1],axis=1).T
                        com = xx.mean(axis=0)
                        xx -= com
                        xx  = xx.dot(numpy.eye(s)*0.5)
                        xx += com
                        nodes   += [xx[0],xx[1]]
                        symbols += ["X","X"]
                nodes = numpy.array(nodes)
                if len(cell)==3:
                    # 2D net, only one angle and two vectors.
                    # need to be completed up to 6 parameters
                    pbc  = [True,True,False] 
                    cell = numpy.array(list(cell[0:2])+[10.0,90.0,90.0]+list(cell[2:]), dtype=numpy.float32)
                    # node coordinates also need to be padded
                    nodes = numpy.pad(nodes, ((0,0),(0,1)), 'constant', constant_values=0.0)
                elif len(cell)<3:
                    continue
                else:
                    pbc = True
                # now some postprocessing for the space groups
                setting = 1
                if ":" in group:
                    # setting might be 2
                    group, setting = group.split(":")
                    try: 
                        setting = int(setting.strip())
                    except ValueError:
                        setting = 1
                # ASE does not have all the spacegroups implemented yet
                if group not in groups.keys():
                    continue
                else:
                    # generate the crystal
                    group     = int(groups[group])
                    topology  = crystal(symbols=symbols,
                                        basis=nodes,
                                        spacegroup=group,
                                        setting=setting,
                                        cellpar=cell,
                                        pbc=pbc,
                                        primitive_cell=False,
                                        onduplicates="keep")
                    fragments = _get_fragments(topology)
                    shapes    = list(set([(fragment[0],fragment[1]) for fragment in fragments.values()]))
                    if "C1" in list(zip(*shapes))[0]:
                        # no symmetry detected for at least one element. 
                        # Cannot be used down the road, so we pass
                        continue
                    else:
                        topologies[name] = {"Shapes"    : shapes,
                                            "Fragments" : fragments,
                                            "Topology"  : topology}
            except Exception as expt:
                print(expt)
                raise
                continue
        bar.finish()
    return topologies

def read_topologies_database(update_db=False,update_source=False):
    """
    TODO
    """


    root = os.path.join(__data__,"topologies")
    db_file = os.path.join(root,"topologies.pkl")
    if ((not os.path.isfile(db_file)) or (update_db)):
        print("Reloading the topologies from scratch")
        topologies = read_topologies(update=update_source)
        topologies_len = len(topologies)
        print("{0:<5} topologies saved".format(topologies_len))
        with open(db_file,"wb") as pkl:
            pickle.dump(obj=topologies,file=pkl)
        return topologies
    else:
        print("Using saved topologies")
        with open(db_file,"rb") as pkl:
            topologies = pickle.load(file=pkl)
            topologies_len = len(topologies)
            print("{0:<5} topologies loaded".format(topologies_len))
            return topologies

def _get_cutoffs(topology,dummy_indices,other_indices):
    """
    Helper function to obtain the cutoffs leading to the desired connectivity
    """
    # initialize cutoffs to small non-zero skin partameter
    skin    = 5e-3
    cutoffs = numpy.zeros(len(topology))+skin
    # we iterate over non-dummies
    for other_index in other_indices:
        # cutoff starts impossibly big
        cutoff   = 10000.0
        # we get the distances to all dummies and cluster accordingly
        dists    = topology.get_distances(other_index,dummy_indices,mic=True)
        mindist  = numpy.amin(dists)
        dists    = dists[dists<2.0*mindist].reshape(-1,1)
        clusters = cluster(dists, mindist*0.5, criterion='distance')
        for cluster_index in set(clusters):
            # check this cluster distances
            indices   = numpy.where(clusters==cluster_index)[0]
            cutoff_tmp = dists[indices].mean() 
            if cutoff_tmp<cutoff :
                # if better score, replace the cutoff
                cutoff = cutoff_tmp
        cutoffs[other_index] = cutoff
    return cutoffs
    
def _get_fragments(topology):
    """
    TODO
    """
    
    dummy_indices  = numpy.array([atom.index for atom in topology if atom.symbol=="X"])
    other_indices  = numpy.array([atom.index for atom in topology if atom.symbol!="X"])
    
    fragments    = {}
    cutoffs      = _get_cutoffs(topology,dummy_indices=dummy_indices,other_indices=other_indices) 
    neighborlist = NeighborList(cutoffs=cutoffs,bothways=True,self_interaction=False,skin=0.0)
    neighborlist.build(topology)
    for other_index in other_indices:
        neighbors_indices,neighbors_offsets = neighborlist.get_neighbors(other_index)
        neighbors_indices,neighbors_offsets = zip(*[(idx,off) for idx,off in list(zip(neighbors_indices,neighbors_offsets)) if idx in dummy_indices])
        neighbors_indices = numpy.array(neighbors_indices)
        neighbors_offsets = numpy.array(neighbors_offsets)
        positions = topology.positions[neighbors_indices] + neighbors_offsets.dot(topology.cell)
        cop       = positions.mean(axis=0)
        positions -= cop
        positions = positions.dot(numpy.eye(3)*0.5) + cop
        # the dummies do not have a mass for the inertia tensor
        fragment = Atoms("X"*len(neighbors_indices),positions) 
        pg       = PointGroup(fragment,cutoffs.mean())
        # save that info
        fragments[other_index] = [pg.schoenflies,topology[other_index].number,fragment,pg.symmops]
        fragments = _tag(fragments,topology.cell)
    return fragments

def _tag(fragments,cell):
    # now we need to tag the corresponding dummies
    tag_value = 1
    keys = list(fragments.keys())
    print(keys)
    for findex0 in keys:
        f0 = fragments[findex0][2]
        # if (f0.get_tags()!=0).all():
            # pass
        for x0i,x0 in enumerate(f0):
            # if x0.tag!=0:
                # pass
            mindist = 1000.0
            best = (None,None)
            for findex1 in keys:
                f1 = fragments[findex1][2]
                # if (f1.get_tags()!=0).all():
                #     pass
                for x1i,x1 in enumerate(f1):
                    if x1i==x0i and findex1==findex0:
                        pass
                    else:
                        atoms = Atoms(cell=cell)
                        atoms.append(x1)
                        atoms.append(x0)
                        ase.visualize.view(atoms)
                        dist = atoms.get_distance(0,1,mic=True)
                        print(findex1,x1i,dist)
                        if dist < mindist:
                            best = (findex1,x1i)
                            mindist = dist
            findex1,x1i = best
            print(findex1,x1i)
            fragments[findex1][2][x1i] = tag_value
            fragments[findex0][2][x0i] = tag_value
            tag_value += 1
        print(f0.get_tags())
    ase.visualize.view(test)
    raise
    return fragments
        




if __name__ == "__main__":

    topologies = read_topologies_database(update_db=True,update_source=True)

