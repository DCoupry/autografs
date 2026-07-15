"""
Batch SBU harvesting: a curated building-block library from real crystals.

Runs :func:`autografs.deconstruct.deconstruct` over many experimental
structures and accumulates their building units into one deduplicated,
library-ready fragment set - the roadmap payoff of growing the SBU
library from real chemistry instead of hand curation.

Deduplication is cross-structure: the same paddlewheel appearing in
fifty MOFs yields one fragment, tagged with every source it came from
(``provenance``). Deconstruction failures never abort the batch; each
is recorded in ``failures`` with its reason, so a real success rate
falls out of a run (fully automatic MOF deconstruction tops out around
70% on diverse databases, so partial failure is expected, not a bug).

>>> result = harvest("cif_directory/", topologies=mofgen.topologies)
>>> result.report()
'harvested 34 fragments from 47/50 structures ...'
>>> result.write_xyz("harvested_sbus.xyz", kinds=("node", "linker"))
>>> mofgen2 = Autografs(xyzfile="harvested_sbus.xyz")   # build with them

Monotopic organic units (single connection point - bound solvent,
modulators, capping residues) are classified ``cap`` and excluded by
default from ``write_xyz`` and the ``building_units`` view; nodes and
linkers are what feeds the builder.
"""

from __future__ import annotations

import glob
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pymatgen.core.structure import Structure

from autografs.deconstruct import (
    deconstruct,
    merge_fragment,
    write_fragments_xyz,
)
from autografs.exceptions import AutografsError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from autografs.fragment import Fragment
    from autografs.topology import Topology

__all__ = [
    "HarvestResult",
    "harvest",
]

logger = logging.getLogger(__name__)

# a per-structure disambiguation suffix ("_2") sits after the trailing
# "<n>X" of a base name; stripping it recovers the cross-structure base
_SUFFIX = re.compile(r"(X)_\d+$")

# fragment kinds that are actual building blocks (not bound solvent)
BUILDING_KINDS = ("node", "linker")

Source = str | Path | Structure


@dataclass
class HarvestResult:
    """Outcome of a batch harvest.

    Attributes
    ----------
    fragments : dict[str, Fragment]
        Deduplicated building blocks across every processed structure,
        dummy-capped and library-ready.
    kinds : dict[str, str]
        Fragment name to its kind ("node", "linker" or "cap").
    provenance : dict[str, list[str]]
        Fragment name to the sorted labels of the sources it appeared
        in.
    nets : dict[str, list[str]]
        Source label to the net candidates identified for it (empty
        when identification was skipped or found nothing).
    failures : dict[str, str]
        Source label to the failure reason, for structures that could
        not be deconstructed.
    n_processed : int
        Number of structures successfully deconstructed.
    """

    fragments: dict[str, Fragment] = field(default_factory=dict)
    kinds: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, list[str]] = field(default_factory=dict)
    nets: dict[str, list[str]] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    n_processed: int = 0

    @property
    def building_units(self) -> dict[str, Fragment]:
        """Nodes and linkers only (bound-solvent caps excluded)."""
        return {
            name: fragment
            for name, fragment in self.fragments.items()
            if self.kinds.get(name) in BUILDING_KINDS
        }

    @property
    def caps(self) -> dict[str, Fragment]:
        """Monotopic organic units: bound solvent, modulators, caps."""
        return {
            name: fragment
            for name, fragment in self.fragments.items()
            if self.kinds.get(name) == "cap"
        }

    def write_xyz(
        self, path: str | Path, kinds: Iterable[str] | None = BUILDING_KINDS
    ) -> Path:
        """Write harvested fragments as a multi-structure XYZ library.

        Parameters
        ----------
        path : str or Path
            Destination file.
        kinds : Iterable[str] or None, optional
            Which kinds to include; defaults to nodes and linkers.
            Pass None to write every fragment, caps included.
        """
        if kinds is None:
            selected = self.fragments
        else:
            wanted = set(kinds)
            selected = {
                name: fragment
                for name, fragment in self.fragments.items()
                if self.kinds.get(name) in wanted
            }
        return write_fragments_xyz(selected, path)

    def report(self) -> str:
        """One-line human-readable summary of the harvest."""
        n_sources = self.n_processed + len(self.failures)
        kind_counts = ", ".join(
            f"{sum(k == kind for k in self.kinds.values())} {kind}(s)"
            for kind in ("node", "linker", "cap")
        )
        return (
            f"harvested {len(self.fragments)} fragments ({kind_counts}) "
            f"from {self.n_processed}/{n_sources} structures"
            + (f", {len(self.failures)} failed" if self.failures else "")
        )

    def __repr__(self) -> str:
        return f"HarvestResult({self.report()})"


def _iter_sources(sources: Source | Iterable[Source]) -> list[tuple[str, Source]]:
    """Normalize the harvest input into (label, source) pairs.

    Accepts a directory (globbed for ``*.cif``/``*.CIF``), a glob
    string, a single path/Structure, or any iterable of those.
    """
    if isinstance(sources, Structure):
        return [("structure_0", sources)]
    if isinstance(sources, (str, Path)):
        path = Path(sources)
        if path.is_dir():
            files = sorted(p for p in path.iterdir() if p.suffix.lower() == ".cif")
            return [(p.stem, p) for p in files]
        if any(ch in str(sources) for ch in "*?[") and not path.exists():
            # glob.glob, not Path().glob: pathlib refuses absolute
            # patterns ("Non-relative patterns are unsupported")
            matches = sorted(Path(match) for match in glob.glob(str(sources)))
            return [(p.stem, p) for p in matches]
        return [(path.stem, path)]
    # an iterable of sources
    pairs: list[tuple[str, Source]] = []
    for i, item in enumerate(sources):
        if isinstance(item, Structure):
            pairs.append((f"structure_{i}", item))
        else:
            pairs.append((Path(item).stem, item))
    return pairs


def harvest(
    sources: Source | Iterable[Source],
    topologies: Mapping[str, Topology] | None = None,
) -> HarvestResult:
    """Deconstruct many structures into one deduplicated SBU library.

    Parameters
    ----------
    sources : path, Structure, or iterable of them
        A directory (its ``.cif`` files are globbed), a glob pattern, a
        single CIF path or pymatgen Structure, or any iterable mixing
        paths and Structures.
    topologies : Mapping[str, Topology] or None, optional
        Topology library for net identification per structure (e.g.
        Autografs.topologies). When None, ``nets`` stays empty.

    Returns
    -------
    HarvestResult
        The merged fragment library with per-fragment provenance,
        per-source net candidates, and a failure report.

    Examples
    --------
    >>> result = harvest("core_mof_subset/", topologies=mofgen.topologies)
    >>> len(result.building_units)
    28
    >>> result.write_xyz("harvested.xyz")
    """
    result = HarvestResult()
    pairs = _iter_sources(sources)
    logger.info(f"harvesting SBUs from {len(pairs)} structure(s)...")
    for label, source in pairs:
        try:
            decon = deconstruct(source, topologies=topologies)
        except AutografsError as exc:
            result.failures[label] = f"{type(exc).__name__}: {exc}"
            logger.warning(f"\t[!] {label}: {exc}")
            continue
        except Exception as exc:  # malformed CIF, pymatgen parse errors
            result.failures[label] = f"{type(exc).__name__}: {exc}"
            logger.warning(f"\t[!] {label}: could not read/parse ({exc})")
            continue
        result.n_processed += 1
        if topologies is not None:
            result.nets[label] = decon.net_candidates
        for name, fragment in decon.fragments.items():
            base_name = _SUFFIX.sub(r"\1", name)
            merged_name = merge_fragment(result.fragments, fragment, base_name)
            result.kinds[merged_name] = name.split("_", 1)[0]
            result.provenance.setdefault(merged_name, [])
            if label not in result.provenance[merged_name]:
                result.provenance[merged_name].append(label)
    for sources_list in result.provenance.values():
        sources_list.sort()
    logger.info(f"\t[x] {result.report()}")
    return result
