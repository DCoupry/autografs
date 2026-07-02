#!/usr/bin/env python3
"""
Convert CGD crystallographic topology files to AuToGraFS pickle format.

This script downloads and processes topology data from the RCSR database
or custom CGD files, creating serialized topology objects for use with
AuToGraFS.

Usage
-----
Download and process RCSR topologies::

    python scripts/cgd2pkl.py -o output.pkl --use_rcsr

Process a custom CGD file::

    python scripts/cgd2pkl.py -i custom.cgd -o output.pkl

Combine RCSR with custom topologies::

    python scripts/cgd2pkl.py -i custom.cgd -o output.pkl --use_rcsr

Notes
-----
The CGD format is documented at http://rcsr.anu.edu.au/help/cgd.
"""
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
from autografs import topology_io
from autografs.fragment import Fragment
from autografs.topology import Topology
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.core.structure import Molecule, Structure
from pymatgen.symmetry.analyzer import PointGroupAnalyzer, SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Nets with vertices above this connectivity are skipped: some RCSR
# entries have enormous coordination figures that are not usable as slots.
MAX_FRAGMENT_SITES = 12

# Placeholder species for dummies during spglib analysis: dummies have
# Z=0 which spglib cannot handle, and He is already taken (it encodes
# 2-coordinated edge centers). Rn cannot occur as a node species since
# node elements encode coordination numbers <= MAX_FRAGMENT_SITES.
DUMMY_PLACEHOLDER = "Rn"


class TopologyExtractionError(Exception):
    """Raised when a net cannot be converted into an AuToGraFS topology."""


def build_group_lookup() -> Dict[str, str]:
    """Map de-underscored spacegroup symbols to their canonical symbols.

    RCSR writes screw axes without underscores (I4132), pymatgen uses
    them (I4_132). Removing underscores from every pymatgen symbol
    yields a collision-free reverse lookup (verified over the full
    encoding table). Without this translation, over half of the RCSR
    nets fail the spacegroup lookup and are silently dropped.
    """
    spacegroups = json.loads(
        pkgutil.get_data(pymatgen.symmetry.__name__, "symm_data.json")
    )["space_group_encoding"]
    lookup: Dict[str, str] = {}
    for key in spacegroups:
        lookup.setdefault(key.replace("_", ""), key)
    return lookup


def normalize_group_symbol(symbol: str, group_lookup: Dict[str, str]) -> str:
    """Translate an RCSR GROUP symbol into a pymatgen SpaceGroup symbol.

    Underscores are restored in the base symbol (I4132 -> I4_132) and
    setting suffixes (':2' for origin choice, ':H'/':R' for
    rhombohedral axes) are preserved: pymatgen's SpaceGroup supports
    them natively, and origin choice matters - feeding origin-2
    coordinates to origin-1 operators generates a wrong net.
    """
    base, _, suffix = symbol.partition(":")
    base = group_lookup.get(base.replace("_", ""), base)
    return f"{base}:{suffix}" if suffix else base


def download_cgd(url: str) -> str:
    """inspired heavily by: https://stackoverflow.com/a/62113293"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with (
        io.BytesIO() as bytIO,
        tqdm(
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in resp.iter_content(chunk_size=1024):
            size = bytIO.write(data)
            bar.update(size)
        cgd = bytIO.getvalue()
    cgd = codecs.escape_decode(cgd)[0].decode("utf-8")
    return cgd


def topology_from_string(
    cgd: str, group_lookup: Dict[str, str]
) -> Tuple[str, Structure, int]:
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
            raw_groupname = l[1].strip()
            groupname = normalize_group_symbol(raw_groupname, group_lookup)
            try:
                spacegroup = SpaceGroup(groupname)
            except ValueError:
                # e.g. 2D plane groups, nonstandard monoclinic settings
                return raw_groupname, None, 0
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
        xyz = numpy.pad(xyz, ((0, 0), (0, 1)), "constant", constant_values=0.0)
    # generate the crystal. Pass the normalized symbol string: it keeps
    # the setting suffix, which both SpaceGroup(...).symbol and the
    # group's int number would silently drop (reverting e.g. Fd-3m:2
    # to origin choice 1 and generating a wrong net).
    topology = Structure.from_spacegroup(
        sg=groupname, lattice=lattice, species=elements, coords=xyz
    )
    # remove any duplicate sites
    topology.merge_sites(tol=1e-3, mode="delete")
    return name, topology, spacegroup.int_number


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
        # some topologies in the RCSR have a crazy size: skip them
        if len(fragment_sites) > MAX_FRAGMENT_SITES:
            raise TopologyExtractionError(
                f"Fragment size {len(fragment_sites)} larger than "
                f"limit of {MAX_FRAGMENT_SITES}."
            )
        # store as molecule to use the point group analysis
        fragment = Molecule.from_sites(
            fragment_sites
        )  # , charge=1, spin_multiplicity=1)
        # symmetry detection runs on a normalized copy: RCSR cells are
        # scaled to unit edge length, so raw arms (~0.25 A) are smaller
        # than the analyzer's distance tolerance and the point group
        # misdetects (e.g. O instead of Td on dia nodes). He stands in
        # for the massless dummies, which break symmetrization.
        centered = fragment.cart_coords - fragment.cart_coords.mean(axis=0)
        arm = numpy.linalg.norm(centered, axis=1).mean()
        if arm < 1e-6:
            raise TopologyExtractionError("Degenerate fragment geometry.")
        normalized = Molecule(["He"] * len(fragment), centered / arm)
        pg = PointGroupAnalyzer(normalized, tolerance=0.1)
        if pg.sch_symbol == "C1":
            raise TopologyExtractionError("No symmetry detected (C1) in fragment.")
        fragments.append(Fragment(atoms=fragment, symmetry=pg, name="slot"))
    return fragments


def orbit_equivalence_classes(
    topology: Structure, center_indices: List[int]
) -> List[int]:
    """Crystallographic orbit id for each slot center.

    Runs spglib through pymatgen's SpacegroupAnalyzer on the generated
    net and maps every center site to its symmetry orbit. Slots in one
    orbit are truly equivalent; two orbits that merely share a point
    group and size stay distinct (they may host different SBUs).

    Returns an empty list when the analysis fails, in which case the
    Topology falls back to grouping by point group and size.
    """
    try:
        analyzable = topology.copy()
        analyzable.replace_species({"X": DUMMY_PLACEHOLDER})
        sga = SpacegroupAnalyzer(analyzable, symprec=1e-3)
        dataset = sga.get_symmetry_dataset()
        equivalent = getattr(dataset, "equivalent_atoms", None)
        if equivalent is None:  # older pymatgen returned a dict
            equivalent = dataset["equivalent_atoms"]
    except Exception as exc:
        logger.debug(f"Orbit analysis failed: {exc!r}")
        return []
    classes: List[int] = []
    remap: Dict[int, int] = {}
    for idx in center_indices:
        representative = int(equivalent[idx])
        classes.append(remap.setdefault(representative, len(remap)))
    return classes


def read_cgd_data(cgd: str) -> Dict[str, Topology]:
    # the final object is a dictionary of crystal structures
    # in pymatgen format representing net topologies, accessible by name
    topologies = {}
    # keep track of weird behaviours
    unknown_symbols = []
    parse_error_counter = 0
    extraction_errors: Dict[str, str] = {}
    group_lookup = build_group_lookup()
    # split the file by topology
    split_cgd = [t.strip().strip("CRYSTAL") for t in cgd.split("END")]
    for cgd_string in tqdm(split_cgd, desc="Creating topologies"):
        if not cgd_string:
            continue
        # read from the template.
        try:
            name, struct, sg_number = topology_from_string(
                cgd=cgd_string, group_lookup=group_lookup
            )
        except (KeyError, ValueError, IndexError) as exc:
            # malformed entries: empty coordinates, bad fields, ...
            parse_error_counter += 1
            logger.debug(f"Could not parse a CGD entry: {exc!r}")
            continue
        if struct is None:
            # name holds the unrecognized spacegroup symbol here
            unknown_symbols.append(name)
            continue
        try:
            fragments = analyze(struct, skin=5e-3)
        except TopologyExtractionError as exc:
            extraction_errors[name] = str(exc)
            continue
        dummies = set(struct.indices_from_symbol("X"))
        centers = [i for i in range(len(struct)) if i not in dummies]
        equivalence_classes = orbit_equivalence_classes(struct, centers)
        topologies[name] = Topology(
            name=name,
            slots=fragments,
            cell=struct.lattice,
            equivalence_classes=equivalence_classes or None,
            spacegroup_number=sg_number,
        )
    logger.info(f"{len(split_cgd)} Topologies treated with:")
    logger.info(f"  + {len(topologies)} successful treatments.")
    logger.info(f"  + {len(unknown_symbols)} bad international symbols errors.")
    for bad_symbol, count in Counter(unknown_symbols).most_common():
        logger.info(f"    - {bad_symbol} : {count}")
    logger.info(f"  + {parse_error_counter} unparseable entries.")
    logger.info(f"  + {len(extraction_errors)} fragment extraction errors:")
    for net_name, reason in sorted(extraction_errors.items()):
        logger.info(f"    - {net_name} : {reason}")
    return topologies


def main(args: argparse.Namespace = None) -> None:
    if args is None:
        parser = argparse.ArgumentParser(
            description="Convenience script to generate topology pickles from cgd files."
        )
        parser.add_argument(
            "-i",
            "--input",
            type=str,
            default=None,
            help="path to a cgd format file. See http://rcsr.anu.edu.au for a downloadable example.",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=str,
            help=(
                "output path for the topology library. Use .json or "
                ".json.gz (recommended); legacy .pkl writes a dill pickle."
            ),
        )
        parser.add_argument(
            "--use_rcsr",
            action="store_true",
            help="Flag to download and use the RCSR nets in addition to the given inputs.",
        )
        args = parser.parse_args()

    topologies = {}
    if args.use_rcsr or args.input is None:
        logger.info(
            "Downloading RCSR nets from http://rcsr.anu.edu.au/downloads/RCSRnets-2019-06-01.cgd"
        )
        # cgd string contining all the RCSR nets
        cgd = download_cgd("http://rcsr.anu.edu.au/downloads/RCSRnets-2019-06-01.cgd")
        # converting defaults to autografs topologies
        topologies.update(read_cgd_data(cgd))
    if args.input is not None:
        assert os.path.isfile(args.input)
        with open(args.input, "rb") as inpt:
            cgd = inpt.read().decode("utf8")
            topologies.update(read_cgd_data(cgd))
    # saving to data folder
    if args.output.endswith((".json", ".json.gz")):
        topology_io.save_topologies(topologies, args.output)
    else:
        logger.warning(
            "Writing a dill pickle; prefer .json.gz (safe to share, "
            "survives pymatgen upgrades)."
        )
        with open(args.output, "wb") as uit:
            dill.dump(topologies, uit)


if __name__ == "__main__":
    main()
