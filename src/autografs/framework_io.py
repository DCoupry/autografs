"""
Serialization of Framework objects to a versioned JSON format.

CIF export loses everything that makes a built framework editable: the
bond graph, the bond orders, the UFF4MOF types, the anchor tags, and
the per-atom slot/SBU provenance that post-build editing (defects,
functionalization, rotation) requires. This module persists the graph
itself - plain data, safe to share, stable across pymatgen versions -
so a framework can be saved in one session and edited in another:

>>> mof.save("mof5.json.gz")
>>> mof = Framework.load("mof5.json.gz")
>>> defective = mof.supercell(2).defects(fraction=0.1, seed=42)

Functions
---------
framework_to_dict / framework_from_dict
    Convert a Framework to/from a JSON-compatible dict.
save_framework / load_framework
    Write/read one framework; gzip-compressed when the path ends
    in .gz.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

import networkx
import numpy as np

from autografs.framework import Framework

__all__ = [
    "framework_to_dict",
    "framework_from_dict",
    "save_framework",
    "load_framework",
]

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1


def framework_to_dict(framework: Framework) -> dict:
    """Convert a Framework into a JSON-compatible dict.

    Parameters
    ----------
    framework : Framework
        The framework to serialize.

    Returns
    -------
    dict
        Plain-data representation: name, energy, cell matrix, per-atom
        columns (symbols, coords, tags, ufftypes, and slot/sbu
        provenance when present), and the bond list with orders.

    Raises
    ------
    ValueError
        If the graph's node ids are not contiguous 0..n-1 (a violated
        Framework invariant; every builder and editing path maintains
        it).
    """
    graph = framework.graph
    nodes = sorted(graph)
    if nodes != list(range(len(nodes))):
        raise ValueError(
            "Framework node ids are not contiguous 0..n-1; the graph "
            "violates the Framework invariant and cannot be serialized."
        )
    data: dict = {
        "name": framework.name,
        "energy": framework.energy,
        "cell": np.asarray(framework.cell, dtype=float).tolist(),
        "symbols": [graph.nodes[n]["symbol"] for n in nodes],
        "coords": [
            np.asarray(graph.nodes[n]["coord"], dtype=float).tolist() for n in nodes
        ],
        "tags": [int(graph.nodes[n]["tag"]) for n in nodes],
        "ufftypes": [graph.nodes[n]["ufftype"] for n in nodes],
        "bonds": sorted(
            [min(u, v), max(u, v), float(d.get("bond_order", 1.0))]
            for u, v, d in graph.edges(data=True)
        ),
    }
    # slot/SBU provenance: written only when every atom carries it, so
    # legacy graphs still save and loading keeps them legacy
    if all("slot" in graph.nodes[n] for n in nodes):
        data["slots"] = [int(graph.nodes[n]["slot"]) for n in nodes]
        data["sbus"] = [graph.nodes[n]["sbu"] for n in nodes]
    # partial charges: same convention as provenance
    if nodes and all("charge" in graph.nodes[n] for n in nodes):
        data["charges"] = [float(graph.nodes[n]["charge"]) for n in nodes]
        if "charge_method" in graph.graph:
            data["charge_method"] = graph.graph["charge_method"]
    # rod-build marker: keeps the editing guards (editing._reject_rod)
    # working across a save/load round trip
    if graph.graph.get("rod_build"):
        data["rod_build"] = True
    return data


def framework_from_dict(data: dict) -> Framework:
    """Reconstruct a Framework from its dict representation.

    Parameters
    ----------
    data : dict
        Output of framework_to_dict.

    Returns
    -------
    Framework
        The reconstructed framework, editable exactly like the one
        that was saved (provenance included when it was present).
    """
    graph = networkx.Graph(cell=np.asarray(data["cell"], dtype=float))
    slots = data.get("slots")
    sbus = data.get("sbus")
    charges = data.get("charges")
    if "charge_method" in data:
        graph.graph["charge_method"] = data["charge_method"]
    if data.get("rod_build"):
        graph.graph["rod_build"] = True
    columns = zip(
        data["symbols"], data["coords"], data["tags"], data["ufftypes"], strict=True
    )
    for i, (symbol, coord, tag, ufftype) in enumerate(columns):
        attributes: dict = {
            "symbol": symbol,
            "coord": np.asarray(coord, dtype=float),
            "tag": int(tag),
            "ufftype": ufftype,
        }
        if slots is not None and sbus is not None:
            attributes["slot"] = int(slots[i])
            attributes["sbu"] = sbus[i]
        if charges is not None:
            attributes["charge"] = float(charges[i])
        graph.add_node(i, **attributes)
    for u, v, order in data["bonds"]:
        graph.add_edge(int(u), int(v), bond_order=float(order))
    framework = Framework(graph, name=data.get("name", "framework"))
    framework.energy = data.get("energy")
    return framework


def save_framework(framework: Framework, path: str | Path) -> Path:
    """Write a framework to a JSON file.

    Parameters
    ----------
    framework : Framework
        The framework to save.
    path : str or Path
        Output path. Written gzip-compressed and compact if it ends in
        .gz; pretty-printed otherwise.

    Returns
    -------
    Path
        The written path.
    """
    path = Path(path)
    payload = {
        "format_version": FORMAT_VERSION,
        "framework": framework_to_dict(framework),
    }
    if path.suffix == ".gz":
        text = json.dumps(payload, separators=(",", ":"))
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    logger.info(f"Saved {framework!r} to {path}")
    return path


def load_framework(path: str | Path) -> Framework:
    """Load a framework from a JSON file.

    Parameters
    ----------
    path : str or Path
        Input path. Read gzip-compressed if it ends in .gz.

    Returns
    -------
    Framework
        The loaded framework.

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
            f"Unsupported framework format version {version!r} in {path}; "
            f"this build of AuToGraFS reads version {FORMAT_VERSION}."
        )
    return framework_from_dict(payload["framework"])
