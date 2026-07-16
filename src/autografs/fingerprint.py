"""Canonical assembly fingerprints: (net, building blocks, fold).

The label that lets an enumerated hypothetical framework be compared
against a deconstructed experimental corpus: two structures share a
fingerprint when they realize the same net, at the same
interpenetration fold, from the same multiset of building blocks.

Block identity is meaningful only inside ONE shared vocabulary - a
harvest's deduplicated fragment library, where merge_fragment has
already collapsed geometrically identical blocks to one name. Built
frameworks record their SBU names as per-atom provenance, so when they
are built *from* that library the two directions meet:

>>> harvest = mofgen.harvest("corpus/")               # the vocabulary
>>> experimental = fingerprint.from_deconstruction(
...     mofgen.deconstruct("IRMOF-1.cif"), library=harvest.fragments)
>>> hypothetical = fingerprint.from_framework(built)  # built from it
>>> experimental == hypothetical                      # realized?

Fingerprints are frozen and hashable: a set of corpus fingerprints
turns "is this combination realized?" into a lookup.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from math import gcd
from typing import TYPE_CHECKING

from autografs.deconstruct import Deconstruction, match_fragment

if TYPE_CHECKING:
    from collections.abc import Mapping

    from autografs.fragment import Fragment
    from autografs.framework import Framework

__all__ = [
    "AssemblyFingerprint",
    "from_deconstruction",
    "from_framework",
]

# fragment names follow "<kind>_<formula>_<n>X" with an optional
# deduplication suffix ("_2", "_3", ...) appended by merge_fragment;
# re-expression in another vocabulary restarts the suffix walk from
# the bare base name
_SUFFIX = re.compile(r"_\d+$")


def _reduced(counts: Counter[str]) -> tuple[tuple[str, int], ...]:
    """Sorted block pairs, counts divided by their gcd: a supercell of
    the same assembly must fingerprint identically."""
    if not counts:
        return ()
    divisor = gcd(*counts.values())
    return tuple(sorted((name, count // divisor) for name, count in counts.items()))


@dataclass(frozen=True)
class AssemblyFingerprint:
    """One assembly: nets realized, block multiset, catenation fold.

    Attributes
    ----------
    nets : tuple[str, ...]
        Sorted net candidates (usually one; empty = unidentified,
        which never equals an identified fingerprint).
    blocks : tuple[tuple[str, int], ...]
        Sorted (block name, count) pairs. Names are vocabulary names;
        blocks absent from the vocabulary appear as
        ``"unmatched:<base name>"`` and therefore never collide with a
        buildable combination.
    fold : int
        Interpenetration fold (1 = non-catenated).
    """

    nets: tuple[str, ...]
    blocks: tuple[tuple[str, int], ...]
    fold: int = 1

    @property
    def is_buildable_vocabulary(self) -> bool:
        """False when any block failed to match the vocabulary."""
        return not any(name.startswith("unmatched:") for name, _ in self.blocks)

    def __str__(self) -> str:
        nets = ",".join(self.nets) or "?"
        blocks = " + ".join(
            f"{count}x{name}" if count > 1 else name for name, count in self.blocks
        )
        fold = f" (x{self.fold})" if self.fold > 1 else ""
        return f"{nets}: {blocks}{fold}"


def from_deconstruction(
    result: Deconstruction,
    library: Mapping[str, Fragment] | None = None,
) -> AssemblyFingerprint:
    """Fingerprint of a deconstructed structure.

    Blocks are the node and linker units (caps - bound solvent,
    modulators - are not part of the assembly and are excluded, the
    same convention as harvesting). With ``library`` given, every
    block is re-expressed in that vocabulary through the same
    geometric identity test the harvest dedup uses; blocks with no
    match keep an ``unmatched:`` marker.

    Raises
    ------
    ValueError
        If the structure contains rod units: a rod has no finite
        block, so no block multiset exists (see #91).
    """
    if getattr(result, "rod_units", []):
        raise ValueError(
            "Rod units have no finite building block; assembly "
            "fingerprints are undefined for rod MOFs."
        )
    counts: Counter[str] = Counter()
    for unit in result.units:
        if unit.kind == "cap":
            continue
        name = unit.name
        if library is not None:
            base = _SUFFIX.sub("", name)
            matched = match_fragment(library, result.fragments[name], base)
            name = matched if matched is not None else f"unmatched:{base}"
        counts[name] += 1
    return AssemblyFingerprint(
        nets=tuple(sorted(result.net_candidates)),
        blocks=_reduced(counts),
        fold=result.n_periodic_components,
    )


def from_framework(
    framework: Framework,
    nets: tuple[str, ...] | None = None,
    fold: int = 1,
) -> AssemblyFingerprint:
    """Fingerprint of a built framework, from its provenance.

    Blocks come straight from the per-slot ``sbu`` provenance - no
    deconstruction runs - so the names are the names of the library
    the framework was built from. ``nets`` defaults to the framework's
    name, which the builder sets to the blueprint topology; pass it
    explicitly for renamed or edited frameworks. ``fold`` must be
    given for frameworks catenated after building (interpenetrate does
    not change provenance names, only slot ids).
    """
    counts = Counter(framework.slots.values())
    return AssemblyFingerprint(
        nets=tuple(sorted(nets)) if nets is not None else (framework.name,),
        blocks=_reduced(counts),
        fold=fold,
    )
