"""
Convert CGD crystallographic topology files into AuToGraFS libraries.

Reads nets in the CGD format used by the RCSR database and Systre,
generates the periodic blueprint for each (spacegroup expansion, slot
extraction, orbit equivalence classes), and writes the topology
library in the JSON format read by Autografs.

Installed as the ``autografs-topologies`` console command::

    autografs-topologies --use_rcsr -o topologies.json.gz
    autografs-topologies -i custom.cgd -o my_topologies.json.gz

Notes
-----
The CGD format is documented at http://rcsr.anu.edu.au/help/cgd.
'# EDGE_CENTER' lines in RCSR files look commented out, but the parser
strips the first two characters of every line, deliberately
uncommenting them: edge centers create the 2-connected slots that
linear building blocks occupy. Preserve that behavior when refactoring.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import pkgutil
import warnings
from collections import Counter
from pathlib import Path

import dill
import numpy as np
import pymatgen.symmetry
import requests
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.core.structure import Molecule, Structure
from pymatgen.symmetry.analyzer import PointGroupAnalyzer, SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup
from tqdm.auto import tqdm

from autografs import plane_groups, topology_io
from autografs.exceptions import TopologyExtractionError
from autografs.fragment import Fragment
from autografs.topology import Topology

logger = logging.getLogger(__name__)

RCSR_URL = "http://rcsr.anu.edu.au/downloads/RCSRnets-2019-06-01.cgd"

# Nets with vertices above this connectivity are skipped: some RCSR
# entries have enormous coordination figures that are not usable as
# slots. 24 keeps the highest-connectivity MOF chemistry (e.g. the
# 24-c rht net); override with --max-connectivity.
MAX_FRAGMENT_SITES = 24

# Placeholder species for dummies during spglib analysis: dummies have
# Z=0 which spglib cannot handle, and He is already taken (it encodes
# 2-coordinated edge centers). Rn cannot occur as a node species since
# node elements encode coordination numbers <= MAX_FRAGMENT_SITES.
DUMMY_PLACEHOLDER = "Rn"

# GROUP symbols pymatgen cannot parse as written. RCSR still uses the
# pre-2002 glide notation for No. 64; the nonstandard monoclinic
# setting symbols (I12/a1, P121/n1, ...) parse natively and need no
# entry here - their operators were checked against ITA.
GROUP_SYNONYMS = {"Cmca": "Cmce"}


def build_group_lookup() -> dict[str, str]:
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
    lookup: dict[str, str] = {}
    for key in spacegroups:
        lookup.setdefault(key.replace("_", ""), key)
    return lookup


def normalize_group_symbol(symbol: str, group_lookup: dict[str, str]) -> str:
    """Translate an RCSR GROUP symbol into a pymatgen SpaceGroup symbol.

    Underscores are restored in the base symbol (I4132 -> I4_132) and
    setting suffixes (':2' for origin choice, ':H'/':R' for
    rhombohedral axes) are preserved: pymatgen's SpaceGroup supports
    them natively, and origin choice matters - feeding origin-2
    coordinates to origin-1 operators generates a wrong net.
    """
    base, _, suffix = symbol.partition(":")
    base = GROUP_SYNONYMS.get(base, base)
    base = group_lookup.get(base.replace("_", ""), base)
    return f"{base}:{suffix}" if suffix else base


def download_cgd(url: str = RCSR_URL) -> str:
    """Download a CGD file with a progress bar and return its text."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with (
        io.BytesIO() as buffer,
        tqdm(total=total, unit="iB", unit_scale=True, unit_divisor=1024) as bar,
    ):
        for chunk in response.iter_content(chunk_size=1024):
            bar.update(buffer.write(chunk))
        return buffer.getvalue().decode("utf-8")


def topology_from_string(
    cgd: str, group_lookup: dict[str, str]
) -> tuple[str, Structure | None, int, bool]:
    """Parse one CGD entry into a periodic pymatgen Structure.

    Returns (name, structure, group number, is_2d); structure is None
    when the GROUP symbol cannot be translated, with the raw symbol in
    the name position. For 2D entries the group number is the ITA
    plane-group number (1-17) and is_2d is True.
    """
    lines = [line[2:].split() for line in cgd.splitlines() if len(line) > 2]
    elements = []
    xyz = []
    is_2d = False
    plane_group: str | None = None
    for tokens in lines:
        if tokens[0].startswith("NAME"):
            name = tokens[1].strip()
        elif tokens[0].startswith("GROUP"):
            raw_groupname = tokens[1].strip()
            # plane groups first: layer nets are expanded with explicit
            # 2D operators, never through a 3D setting (see the module
            # docstring of autografs.plane_groups for why)
            if raw_groupname in plane_groups.PLANE_GROUPS:
                plane_group = raw_groupname
                continue
            groupname = normalize_group_symbol(raw_groupname, group_lookup)
            try:
                spacegroup = SpaceGroup(groupname)
            except ValueError:
                # e.g. nonstandard monoclinic settings
                return raw_groupname, None, 0, False
        elif tokens[0].startswith("CELL"):
            parameters = [float(p) for p in tokens[1:]]
            if len(parameters) == 3:
                # 2D net, only one angle and two vectors.
                # need to be completed up to 6 parameters
                parameters = parameters[0:2] + [10.0, 90.0, 90.0] + parameters[2:]
                is_2d = True
            lattice = Lattice.from_parameters(*parameters)
        elif tokens[0].startswith("NODE"):
            # the element encodes the coordination number (Z = CN)
            elements.append(get_el_sp(int(tokens[2])))
            xyz.append(np.array(tokens[3:], dtype=float))
        elif tokens[0].startswith("EDGE_CENTER"):
            # add a linear connector, represented by He
            elements.append(get_el_sp(2))
            xyz.append(np.array(tokens[1:], dtype=float))
        elif tokens[0].startswith("EDGE"):
            # append two dummies at the quarter points of the edge
            midpoint = int((len(tokens) + 1) / 2)
            ends = np.stack(
                [
                    np.array(tokens[1:midpoint], dtype=float),
                    np.array(tokens[midpoint:], dtype=float),
                ]
            )
            center = ends.mean(axis=0)
            quarter_points = center + 0.5 * (ends - center)
            dummy_element = get_el_sp("X")
            xyz += [quarter_points[0], quarter_points[1]]
            elements += [dummy_element, dummy_element]
    coords = np.stack(xyz, axis=0)
    if is_2d:
        # node coordinates need to be padded to 3D
        coords = np.pad(coords, ((0, 0), (0, 1)), "constant", constant_values=0.0)
    if plane_group is not None:
        if not is_2d:
            raise ValueError(f"Plane group {plane_group} on a 3D CELL in entry {name}.")
        topology = plane_groups.structure_from_plane_group(
            symbol=plane_group, lattice=lattice, species=elements, frac_coords=coords
        )
        group_number = plane_groups.PLANE_GROUPS[plane_group].number
    else:
        if is_2d:
            raise ValueError(f"2D CELL without a plane group in entry {name}.")
        # generate the crystal. Pass the normalized symbol string: it
        # keeps the setting suffix, which both SpaceGroup(...).symbol
        # and the group's int number would silently drop (reverting
        # e.g. Fd-3m:2 to origin choice 1 and generating a wrong net).
        topology = Structure.from_spacegroup(
            sg=groupname, lattice=lattice, species=elements, coords=coords
        )
        group_number = spacegroup.int_number
    # remove any duplicate sites
    topology.merge_sites(tol=1e-3, mode="delete")
    return name, topology, group_number, is_2d


def analyze(
    topology: Structure,
    skin: float = 5e-3,
    max_sites: int = MAX_FRAGMENT_SITES,
) -> list[Fragment]:
    """Extract one slot Fragment per non-dummy site of the net."""
    fragments = []
    dmat = topology.distance_matrix
    dummies = np.array(topology.indices_from_symbol("X"))
    not_dummies = np.array([i for i in range(len(topology)) if i not in dummies])
    # initialize and set tags
    tags = np.zeros(len(topology), dtype=int)
    tags[dummies] = dummies + 1
    topology.add_site_property(property_name="tags", values=tags)
    # get the distances between centers and connections
    distances = dmat[not_dummies][:, dummies]
    coordinations = np.array(topology.atomic_numbers)[not_dummies]
    partitions = np.argsort(distances, axis=1)
    for center_idx, best_dummies in enumerate(partitions):
        coordination = coordinations[center_idx]
        if coordination < len(best_dummies):
            best_dummies = best_dummies[:coordination]
        cutoff = distances[center_idx][best_dummies].max() + skin
        # now extract the corresponding fragment
        fragment_center = topology.sites[not_dummies[center_idx]]
        fragment_sites = topology.get_neighbors(fragment_center, r=cutoff)
        # some topologies in the RCSR have a crazy size: skip them
        if len(fragment_sites) > max_sites:
            raise TopologyExtractionError(
                f"Fragment size {len(fragment_sites)} larger than limit of {max_sites}."
            )
        # store as molecule to use the point group analysis
        fragment = Molecule.from_sites(fragment_sites)
        # symmetry detection runs on a normalized copy: RCSR cells are
        # scaled to unit edge length, so raw arms (~0.25 A) are smaller
        # than the analyzer's distance tolerance and the point group
        # misdetects (e.g. O instead of Td on dia nodes). He stands in
        # for the massless dummies, which break symmetrization.
        centered = fragment.cart_coords - fragment.cart_coords.mean(axis=0)
        arm = np.linalg.norm(centered, axis=1).mean()
        if arm < 1e-6:
            raise TopologyExtractionError("Degenerate fragment geometry.")
        normalized = Molecule(["He"] * len(fragment), centered / arm)
        # the point group is metadata only: slot/SBU compatibility is
        # geometric (directional matching), so low-symmetry (C1)
        # vertices are perfectly usable and do not reject the net
        pg = PointGroupAnalyzer(normalized, tolerance=0.1)
        fragments.append(Fragment(atoms=fragment, symmetry=pg, name="slot"))
    return fragments


def orbit_equivalence_classes(
    topology: Structure, center_indices: list[int]
) -> list[int]:
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
    classes: list[int] = []
    remap: dict[int, int] = {}
    for idx in center_indices:
        representative = int(equivalent[idx])
        classes.append(remap.setdefault(representative, len(remap)))
    return classes


def read_cgd_data(cgd: str, max_sites: int = MAX_FRAGMENT_SITES) -> dict[str, Topology]:
    """Convert the entries of a CGD file into Topology objects."""
    topologies: dict[str, Topology] = {}
    # keep track of weird behaviours
    unknown_symbols = []
    parse_error_counter = 0
    extraction_errors: dict[str, str] = {}
    group_lookup = build_group_lookup()
    # split the file by topology
    split_cgd = [t.strip().strip("CRYSTAL") for t in cgd.split("END")]
    for cgd_string in tqdm(split_cgd, desc="Creating topologies"):
        if not cgd_string:
            continue
        # read from the template.
        try:
            name, struct, sg_number, is_2d = topology_from_string(
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
            fragments = analyze(struct, skin=5e-3, max_sites=max_sites)
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
            is_2d=is_2d,
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


def main(argv: list[str] | None = None) -> None:
    """Console entry point: autografs-topologies."""
    parser = argparse.ArgumentParser(
        prog="autografs-topologies",
        description="Generate an AuToGraFS topology library from CGD files.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="path to a cgd format file. See http://rcsr.anu.edu.au.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
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
    parser.add_argument(
        "--max-connectivity",
        type=int,
        default=MAX_FRAGMENT_SITES,
        help=(
            "skip nets whose vertices exceed this connectivity "
            f"(default {MAX_FRAGMENT_SITES}; the rht net needs 24)."
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    warnings.filterwarnings("ignore")

    topologies: dict[str, Topology] = {}
    if args.use_rcsr or args.input is None:
        logger.info(f"Downloading RCSR nets from {RCSR_URL}")
        cgd = download_cgd()
        topologies.update(read_cgd_data(cgd, max_sites=args.max_connectivity))
    if args.input is not None:
        input_path = Path(args.input)
        if not input_path.is_file():
            raise SystemExit(f"Input file not found: {input_path}")
        cgd = input_path.read_bytes().decode("utf8")
        topologies.update(read_cgd_data(cgd, max_sites=args.max_connectivity))

    if args.output.endswith((".json", ".json.gz")):
        topology_io.save_topologies(topologies, args.output)
    else:
        logger.warning(
            "Writing a dill pickle; prefer .json.gz (safe to share, "
            "survives pymatgen upgrades)."
        )
        with open(args.output, "wb") as handle:
            dill.dump(topologies, handle)


if __name__ == "__main__":
    main()
