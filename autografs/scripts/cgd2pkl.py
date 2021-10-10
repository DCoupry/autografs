import argparse
import codecs
import io
import os
import json
import logging
import pkgutil
import warnings
from collections import Counter
from typing import Dict, List, Tuple

import dill
import numpy
import pymatgen.symmetry
import requests
from autografs.structure import Fragment, Topology
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.core.structure import Molecule, Structure
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from tqdm.auto import tqdm

from ase.visualize import view

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def download_cgd(url: str) -> str:
    """inspired heavily by: https://stackoverflow.com/a/62113293"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with io.BytesIO() as bytIO, tqdm(
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = bytIO.write(data)
            bar.update(size)
        cgd = bytIO.getvalue()
    cgd = codecs.escape_decode(cgd)[0].decode("utf-8")
    return cgd


def topology_from_string(cgd : str, spacegroups: Dict[str, int]) -> Tuple[str, Structure]:
    lines = cgd.splitlines()
    lines = [l[2:].split() for l in lines if len(l) > 2]
    elements = []
    xyz = []
    is_2D = False
    # might benefit from new case syntax in 3.10
    for l in lines:
        if l[0].startswith("NAME"):
            name = l[1].strip()
        elif l[0].startswith("GROUP"):
            groupname = l[1].strip()
            if groupname not in spacegroups.keys():
                return groupname, None
            sg = spacegroups[groupname]
        elif l[0].startswith("CELL"):
            parameters = [float(p) for p in l[1:]]
            if len(parameters) == 3:
                # 2D net, only one angle and two vectors.
                # need to be completed up to 6 parameters
                parameters = parameters[0:2] + [10.0, 90.0, 90.0] + parameters[2:]
                is_2D = True
            lattice = Lattice.from_parameters(*parameters)
        elif l[0].startswith("NODE"):
            elements.append(get_el_sp(int(l[2])))
            xyz.append(numpy.array(l[3:], dtype=float))
        elif l[0].startswith("EDGE_CENTER"):
            # add a linear connector, represented by He
            elements.append(get_el_sp(int(2)))
            xyz.append(numpy.array(l[1:], dtype=float))
        elif l[0].startswith("EDGE"):
            # now we append some dummies
            s = int((len(l) - 1) / 2)
            midl = int((len(l) + 1) / 2)
            x0 = numpy.array(l[1:midl], dtype=float).reshape(-1, 1)
            x1 = numpy.array(l[midl:], dtype=float).reshape(-1, 1)
            xx = numpy.concatenate([x0, x1], axis=1).T
            com = xx.mean(axis=0)
            xx -= com
            xx = xx.dot(numpy.eye(s) * 0.5)
            xx += com
            dummy_element = get_el_sp("X")
            xyz += [xx[0], xx[1]]
            elements += [dummy_element, dummy_element]
    # concatenate the coordinates
    xyz = numpy.stack(xyz, axis=0)
    if is_2D:
        # node coordinates need to be padded
        xyz = numpy.pad(xyz, ((0, 0), (0, 1)),
                        'constant',
                         constant_values=0.0)
    # generate the crystal
    topology = Structure.from_spacegroup(sg=sg,
                                         lattice=lattice,
                                         species=elements,
                                         coords=xyz)
    # remove any duplicate sites
    topology.merge_sites(tol=1e-3, mode="delete")
    return name, topology


def analyze(topology: Structure, skin: float = 5e-3) -> List[Molecule]:
    # containers for the output
    fragments = []
    # we iterate over non-dummies
    dmat = topology.distance_matrix
    dummies = numpy.array(topology.indices_from_symbol("X"))
    not_dummies = numpy.array([i for i in range(len(topology)) if i not in dummies])
    # initialize and set tags
    tags = numpy.zeros(len(topology), dtype=int)
    tags[dummies] = dummies + 1
    topology.add_site_property(property_name="tags", values=tags)
    # TODO : site properties
    # get the distances between centers and connections
    distances = dmat[not_dummies][:, dummies]
    coordinations = numpy.array(topology.atomic_numbers)[not_dummies]
    partitions = numpy.argsort(distances, axis=1)
    for center_idx, best_dummies in enumerate(partitions):
        coordination = coordinations[center_idx]
        if coordination < len(best_dummies):
            best_dummies = best_dummies[:coordination]
        cutoff = distances[center_idx][best_dummies].max() + skin
        # now extract the corresponding fragment
        fragment_center = topology.sites[not_dummies[center_idx]]
        fragment_sites = topology.get_neighbors(fragment_center, r=cutoff)
        # some topologies in the RCSR have a crazy size
        # we ignore them with the heat of a thousand suns
        assert len(fragment_sites) <= 12, "Fragment size larger than limit of 12."
        # store as molecule to use the point group analysis
        fragment = Molecule.from_sites(fragment_sites)#, charge=1, spin_multiplicity=1)
        # this is needed because X have no mass, leading to a
        # symmetrization error.
        fragment.replace_species({"X": "He"})
        pg = PointGroupAnalyzer(fragment, tolerance=0.1)
        # from pymatgen.io.ase import AseAtomsAdaptor
        # view(AseAtomsAdaptor.get_atoms(fragment))
        if pg.sch_symbol == "C1":
            raise("NoSymm")
        fragment.replace_species({"He":"X"})
        fragments.append(Fragment(atoms=fragment, symmetry=pg, name="slot"))
    return fragments


def read_cgd_data(cgd: str, spacegroups: Dict[str, int]) -> Dict[str, Topology]:
    # the final object is a dictionary of crystal structures
    # in pymatgen format representing net topologies, accessible by name
    topologies = {}
    # keep track of weird behaviours
    unknown_symbols = []
    sb_error_counter = 0
    pg_error_counter = 0
    # split the file by topology
    split_cgd = [t.strip().strip("CRYSTAL") for t in cgd.split("END")]
    for cgd_string in tqdm(split_cgd, desc="Creating topologies"):
        if not cgd_string:
            continue
        # read from the template.
        try:
            name, struct = topology_from_string(cgd=cgd_string, spacegroups=spacegroups)
            if struct is None:
                sb_error_counter += 1
                unknown_symbols.append(name)
                continue
            fragments = analyze(struct, skin=5e-3)
            topology = Topology(name=name, slots=fragments, cell=struct.lattice)
            topologies[name] = topology
        except Exception:
            # mainly KeyError for bad international symbols,
            # occasional ValueError, for empty coordinates
            # TODO: separate counters
            pg_error_counter += 1
    logger.info(f"{len(split_cgd)} Topologies treated with:")
    logger.info(f"  + {len(topologies)} successful treatments.")
    logger.info(f"  + {sb_error_counter} bad international symbols errors.")
    for bad_symbol, count in Counter(unknown_symbols).most_common():
        logger.info(f"    - {bad_symbol} : {count}")
    logger.info(f"  + {pg_error_counter} fragment symmetrization errors.")
    return topologies



def main(args: argparse.Namespace) -> None:
    topologies = {}
    # dictionary of symmetry groups from pymatgen
    spacegroups = json.loads(pkgutil.get_data(pymatgen.symmetry.__name__, "symm_data.json"))
    # only keep a dict of the name to index of the spacegroups
    spacegroups = {k:v["int_number"] for k, v in spacegroups["space_group_encoding"].items()}
    if args.use_rcsr or args.input is None:
        logger.info("Downloading RCSR nets from http://rcsr.anu.edu.au/downloads/RCSRnets-2019-06-01.cgd")
        # cgd string contining all the RCSR nets
        cgd = download_cgd("http://rcsr.anu.edu.au/downloads/RCSRnets-2019-06-01.cgd")
        # converting defaults to autografs topologies
        topologies.update(read_cgd_data(cgd, spacegroups))
    if args.input is not None:
        assert os.path.isfile(args.input)
        with open(args.input, "rb") as inpt:
            cgd = inpt.read().decode("utf8")
            topologies.update(read_cgd_data(cgd, spacegroups))
    # saving to data folder
    with open(args.output, "wb") as uit:
        dill.dump(topologies, uit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convenience script to generate topology pickles from cgd files.')
    parser.add_argument('-i', '--input', type=str, default=None, help='path to a cgd format file. See http://rcsr.anu.edu.au for a downloadable example.')
    parser.add_argument('-o', '--output', type=str, help='path to the pickle output where the results will be stored.')
    parser.add_argument('--use_rcsr', action="store_true", help='Flag to download and use the RCSR nets in addition to the given inputs.')
    args = parser.parse_args()
    main(args)
