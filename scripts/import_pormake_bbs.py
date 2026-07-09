"""
Convert the PORMAKE building-block library into an AuToGraFS SBU file.

PORMAKE (https://github.com/Sangwon91/PORMAKE, MIT license) ships 800+
node and edge building blocks curated from the ToBaCCo and CoRE MOF
databases. Its .xyz format is:

    line 1: atom count
    line 2: indices of the connection-point atoms
    atoms:  symbol x y z  (connection points use the symbol X)
    bonds:  i j type      (ignored - AuToGraFS re-derives bonding)

which maps directly onto the AuToGraFS convention (dummy atoms with
symbol X mark connection points; placement is direction-driven, so
PORMAKE's variable X distances are fine). Names are prefixed with
``pormake_`` to keep the provenance visible and avoid collisions with
the curated defaults.

Usage::

    git clone --depth 1 https://github.com/Sangwon91/PORMAKE
    python scripts/import_pormake_bbs.py \
        -i PORMAKE/src/pormake/database/bbs \
        -o src/autografs/data/pormake.xyz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("import_pormake_bbs")

# slots above this connectivity do not exist in the topology library
MAX_CONNECTIVITY = 24
# a connection point's anchor (nearest real atom) further away than
# this is a malformed entry, not a convention difference
MAX_ANCHOR_DISTANCE = 2.0


def parse_bb(path: Path) -> tuple[list[str], np.ndarray] | None:
    """Parse one PORMAKE building block; None (with a log) if unusable."""
    lines = path.read_text().splitlines()
    natoms = int(lines[0])
    symbols: list[str] = []
    coords = np.empty((natoms, 3))
    for i, line in enumerate(lines[2 : 2 + natoms]):
        parts = line.split()
        symbols.append(parts[0])
        coords[i] = [float(v) for v in parts[1:4]]
    if not np.isfinite(coords).all():
        logger.warning(f"{path.name}: non-finite coordinates; skipped.")
        return None
    dummies = [i for i, s in enumerate(symbols) if s == "X"]
    real = [i for i, s in enumerate(symbols) if s != "X"]
    if not 2 <= len(dummies) <= MAX_CONNECTIVITY:
        logger.warning(f"{path.name}: {len(dummies)} connection points; skipped.")
        return None
    if not real:
        logger.warning(f"{path.name}: no real atoms; skipped.")
        return None
    for i in dummies:
        nearest = np.linalg.norm(coords[real] - coords[i], axis=1).min()
        if nearest > MAX_ANCHOR_DISTANCE:
            logger.warning(
                f"{path.name}: connection point {i} is {nearest:.2f} A "
                "from the nearest atom; skipped."
            )
            return None
    return symbols, coords


def convert(input_dir: Path, output_file: Path) -> None:
    blocks: list[str] = []
    kept = 0
    skipped = 0
    for path in sorted(input_dir.glob("*.xyz")):
        parsed = parse_bb(path)
        if parsed is None:
            skipped += 1
            continue
        symbols, coords = parsed
        name = f"pormake_{path.stem}"
        lines = [str(len(symbols)), f"name={name} source=PORMAKE(MIT)"]
        for symbol, (x, y, z) in zip(symbols, coords, strict=True):
            lines.append(f"{symbol} {x:.6f} {y:.6f} {z:.6f}")
        blocks.append("\n".join(lines))
        kept += 1
    output_file.write_text("\n".join(blocks) + "\n")
    logger.info(f"Wrote {kept} building blocks to {output_file} ({skipped} skipped).")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="PORMAKE bbs directory (src/pormake/database/bbs)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="output multi-structure XYZ file",
    )
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
