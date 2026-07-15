#!/usr/bin/env python3
"""
Verify plane-group expansion against every 2D net in an RCSR CGD file.

For each 2D entry, every EDGE endpoint must coincide (mod 1) with an
image of some NODE under the entry's plane group: the second endpoint
is usually a symmetry image outside the asymmetric unit, so a wrong
operator table makes endpoints miss the node orbit immediately. This
checks each of the plane groups actually used by RCSR against every
net that uses it - the "verify each group against a known net" check,
automated.

Run from the repository root:

    python scripts/verify_plane_groups.py path/to/RCSRnets.cgd
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from autografs import plane_groups  # noqa: E402

TOLERANCE = 2e-4  # RCSR coordinates carry 5 decimals


def check_entry(entry: str) -> tuple[str, str, int, int] | None:
    """Return (name, group, n_endpoints, n_misses) for a 2D entry."""
    name, group = None, None
    nodes, endpoints = [], []
    for raw_line in entry.splitlines():
        tokens = raw_line[2:].split()
        if not tokens:
            continue
        key = tokens[0].upper()
        if key.startswith("NAME"):
            name = tokens[1]
        elif key.startswith("GROUP"):
            group = tokens[1]
        elif key.startswith("NODE") and len(tokens) == 5:
            nodes.append([float(tokens[3]), float(tokens[4])])
        elif key.startswith("EDGE_CENTER"):
            continue
        elif key.startswith("EDGE") and len(tokens) == 5:
            endpoints.append([float(tokens[1]), float(tokens[2])])
            endpoints.append([float(tokens[3]), float(tokens[4])])
    if group not in plane_groups.PLANE_GROUPS or not nodes or not endpoints:
        return None
    orbit = np.concatenate([plane_groups.expand_orbit(group, node) for node in nodes])
    misses = 0
    for endpoint in endpoints:
        delta = np.abs(orbit - np.mod(endpoint, 1.0))
        # periodic distance in fractional space
        delta = np.minimum(delta, 1.0 - delta)
        if delta.max(axis=1).min() > TOLERANCE:
            misses += 1
    return name, group, len(endpoints), misses


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    text = Path(sys.argv[1]).read_bytes().decode("utf8")
    per_group: dict[str, list[str]] = defaultdict(list)
    failures: dict[str, list[str]] = defaultdict(list)
    for entry in text.split("END"):
        result = check_entry(entry)
        if result is None:
            continue
        name, group, _, misses = result
        per_group[group].append(name)
        if misses:
            failures[group].append(name)
    print(f"{'group':8s} {'nets':>5s} {'failed':>7s}")
    for group in sorted(per_group, key=lambda g: plane_groups.PLANE_GROUPS[g].number):
        bad = failures.get(group, [])
        marker = f"  <-- {', '.join(bad)}" if bad else ""
        print(f"{group:8s} {len(per_group[group]):5d} {len(bad):7d}{marker}")
    total = sum(len(v) for v in per_group.values())
    total_bad = sum(len(v) for v in failures.values())
    print(f"\n{total} 2D nets checked, {total_bad} with unmatched edge endpoints")
    if total_bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
