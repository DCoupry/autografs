"""Corpus resolution shared by every benchmark driver.

This exists because the same ``_collect`` was copied into three drivers
and then had to be fixed three times. Manifest support was added to
``roundtrip.py``, then to ``embedding.py``, and ``rodtrip.py`` silently
processed one structure and reported a single deconstruction failure --
which reads like a corpus problem rather than a driver one. One
implementation, imported everywhere.

A corpus spec may be:

- a directory                -> its ``*.cif`` files, sorted
- a glob                     -> matching files, sorted
- a ``.txt`` manifest        -> one CIF path per line
- a single CIF               -> itself

The manifest form is not a convenience. CoRE MOF 2025 ships several
solvent-removal variants of most materials, so the analysable unit has
to be chosen upstream and handed to the driver as an explicit list;
globbing a directory would count most materials twice.
"""

from __future__ import annotations

from pathlib import Path


def collect(spec: str) -> list[Path]:
    """Resolve a corpus spec to a sorted list of structure paths."""
    path = Path(spec)
    if path.is_dir():
        return sorted(path.glob("*.cif"))
    if path.suffix.lower() == ".txt" and path.exists():
        return sorted(
            Path(line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if any(char in spec for char in "*?["):
        return sorted(Path(spec).parent.glob(path.name))
    return [path]
