"""
Serialization of topology libraries to a versioned JSON format.

This replaces the dill pickle distribution format. Pickles execute
arbitrary code on load (a real concern for shared topology packs), and
they couple stored data to pymatgen's internal object layout, so a
pymatgen upgrade can orphan every existing file. A topology is just
names, cells, species, coordinates and tags - plain data - and is
stored as such here. Files are diffable, versionable, and safe to load.

Functions
---------
topology_to_dict / topology_from_dict
    Convert a single Topology to/from a JSON-compatible dict.
save_topologies / load_topologies
    Write/read a whole library, gzip-compressed when the path ends
    in .gz.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

import numpy as np
from pymatgen.core.structure import Molecule

from autografs.fragment import Fragment
from autografs.topology import Topology

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1

# stored coordinates are rounded: beyond this precision there is only
# floating point noise, and file size matters at RCSR scale
COORD_DECIMALS = 8


def topology_to_dict(topology: Topology) -> dict:
    """Convert a Topology into a JSON-compatible dict.

    Parameters
    ----------
    topology : Topology
        The topology to serialize.

    Returns
    -------
    dict
        Plain-data representation: cell matrix, spacegroup number, and
        per-slot species/coords/tags/pointgroup/equivalence_class.
    """
    slots = []
    for slot in topology.slots:
        atoms = slot.atoms
        n_atoms = len(atoms)
        tags = atoms.site_properties.get("tags", [0] * n_atoms)
        slots.append(
            {
                "species": [site.specie.symbol for site in atoms],
                "coords": np.round(atoms.cart_coords, COORD_DECIMALS).tolist(),
                "tags": [int(t) for t in tags],
                "pointgroup": slot.pointgroup,
                "equivalence_class": slot.equivalence_class,
            }
        )
    return {
        "cell": np.round(topology.cell.matrix, COORD_DECIMALS).tolist(),
        "spacegroup_number": getattr(topology, "spacegroup_number", None),
        "slots": slots,
    }


def topology_from_dict(name: str, data: dict) -> Topology:
    """Reconstruct a Topology from its dict representation.

    Slot fragments carry the stored point group symbol, so no point
    group analysis runs at load time.

    Parameters
    ----------
    name : str
        The topology name (dict key in the library file).
    data : dict
        Output of topology_to_dict.

    Returns
    -------
    Topology
        The reconstructed topology.
    """
    slots: list[Fragment] = []
    equivalence_classes: list[int | None] = []
    for slot_data in data["slots"]:
        atoms = Molecule(
            slot_data["species"],
            slot_data["coords"],
            site_properties={"tags": slot_data["tags"]},
        )
        slots.append(
            Fragment(atoms=atoms, name="slot", pointgroup=slot_data["pointgroup"])
        )
        equivalence_classes.append(slot_data.get("equivalence_class"))
    known_classes = (
        equivalence_classes if all(c is not None for c in equivalence_classes) else None
    )
    return Topology(
        name=name,
        slots=slots,
        cell=np.array(data["cell"], dtype=float),
        equivalence_classes=known_classes,
        spacegroup_number=data.get("spacegroup_number"),
    )


def save_topologies(topologies: dict[str, Topology], path: str | Path) -> None:
    """Write a topology library to a JSON file.

    Parameters
    ----------
    topologies : dict[str, Topology]
        The library, keyed by topology name.
    path : str or Path
        Output path. Written gzip-compressed and compact if it ends in
        .gz; pretty-printed otherwise, so plain .json files (e.g. test
        fixtures) stay diffable. Output ordering is deterministic
        either way, so even .gz libraries diff cleanly after zcat.
    """
    path = Path(path)
    payload = {
        "format_version": FORMAT_VERSION,
        "topologies": {
            name: topology_to_dict(topology)
            for name, topology in sorted(topologies.items())
        },
    }
    if path.suffix == ".gz":
        text = json.dumps(payload, separators=(",", ":"))
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        text = json.dumps(payload, indent=1)
        path.write_text(text, encoding="utf-8")
    logger.info(f"Saved {len(topologies)} topologies to {path}")


def load_topologies(path: str | Path) -> dict[str, Topology]:
    """Load a topology library from a JSON file.

    Parameters
    ----------
    path : str or Path
        Input path. Read gzip-compressed if it ends in .gz.

    Returns
    -------
    dict[str, Topology]
        The library, keyed by topology name.

    Raises
    ------
    ValueError
        If the file declares an unsupported format version.
    """
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported topology format version {version!r} in {path}; "
            f"this build of AuToGraFS reads version {FORMAT_VERSION}."
        )
    return {
        name: topology_from_dict(name, data)
        for name, data in payload["topologies"].items()
    }
