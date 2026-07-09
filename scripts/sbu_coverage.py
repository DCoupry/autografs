"""
Report SBU coverage of the topology library.

A topology is *covered* when every one of its slot types has at least
one compatible SBU, i.e. it can be built at all. This is the metric
SBU-library growth should move.

Usage::

    python scripts/sbu_coverage.py                      # full library
    python scripts/sbu_coverage.py --exclude pormake_   # without a prefix
"""

from __future__ import annotations

import argparse
import logging
import time

from autografs import Autografs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude",
        default=None,
        help="exclude SBUs whose name starts with this prefix",
    )
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)

    gen = Autografs()
    subset = None
    if args.exclude:
        subset = [n for n in gen.sbu if not n.startswith(args.exclude)]
    n_sbu = len(subset) if subset is not None else len(gen.sbu)

    covered = 0
    total = 0
    t0 = time.time()
    for name in gen.list_topologies():
        total += 1
        topology = gen.topologies[name]
        options = gen.list_building_units(sieve=name, subset=subset)
        if len(options) == len(topology.mappings) and all(options.values()):
            covered += 1
    print(
        f"{covered}/{total} topologies covered "
        f"({covered / total:.1%}) with {n_sbu} SBUs "
        f"[{time.time() - t0:.0f} s]"
    )


if __name__ == "__main__":
    main()
