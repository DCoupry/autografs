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
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
from pymatgen.core.structure import Molecule

from autografs.fragment import Fragment
from autografs.topology import Topology

logger = logging.getLogger(__name__)


class LazyTopologyLibrary(Mapping):
    """A topology library that materializes entries on first access.

    Reconstructing all ~2500 Topology objects (tens of thousands of
    pymatgen Molecules) takes ~10 s, but a session typically touches a
    handful of nets. This mapping keeps the parsed JSON payload and
    builds each Topology the first time it is requested, caching the
    result. Iteration and membership tests never materialize anything.
    """

    def __init__(self, payload: dict[str, dict]) -> None:
        self._raw = payload
        self._cache: dict[str, Topology] = {}

    def __getitem__(self, name: str) -> Topology:
        if name not in self._cache:
            self._cache[name] = topology_from_dict(name, self._raw[name])
        return self._cache[name]

    def __contains__(self, name: object) -> bool:
        # the Mapping default would materialize via __getitem__
        return name in self._raw

    def __iter__(self) -> Iterator[str]:
        return iter(self._raw)

    def __len__(self) -> int:
        return len(self._raw)

    def __repr__(self) -> str:
        return (
            f"LazyTopologyLibrary({len(self._raw)} topologies, "
            f"{len(self._cache)} materialized)"
        )


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
    payload = {
        "cell": np.round(topology.cell.matrix, COORD_DECIMALS).tolist(),
        "spacegroup_number": getattr(topology, "spacegroup_number", None),
        "slots": slots,
    }
    # written only when set: 3D entries and format-version-1 files stay
    # byte-identical, and .get() on load keeps old files readable
    if getattr(topology, "is_2d", False):
        payload["is_2d"] = True
    return payload


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
        [c for c in equivalence_classes if c is not None]
        if all(c is not None for c in equivalence_classes)
        else None
    )
    return Topology(
        name=name,
        slots=slots,
        cell=np.array(data["cell"], dtype=float),
        equivalence_classes=known_classes,
        spacegroup_number=data.get("spacegroup_number"),
        is_2d=data.get("is_2d", False),
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


def load_topologies(path: str | Path) -> LazyTopologyLibrary:
    """Load a topology library from a JSON file.

    Parameters
    ----------
    path : str or Path
        Input path. Read gzip-compressed if it ends in .gz.

    Returns
    -------
    LazyTopologyLibrary
        The library, keyed by topology name. Behaves as a read-only
        mapping; entries are reconstructed on first access.

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
    return LazyTopologyLibrary(payload["topologies"])
