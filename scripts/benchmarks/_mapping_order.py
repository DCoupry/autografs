"""Order in which fragment-to-slot assignments are tried.

Both closure drivers (``roundtrip.py``, ``embedding.py``) enumerate one
compatible fragment per slot type and try the first N combinations,
because a multi-orbit net with several interchangeable fragments
explodes combinatorially and a benchmark has to terminate.

``itertools.product`` is the wrong enumeration for that budget. It
advances the **last** iterable fastest, so the first N combinations all
share the same fragment on every slot type but the last: a 3-orbit net
with 4 fitting fragments each has 64 combinations, and a budget of 8
spends every one of them on ``(f0, f0, *)``. The first slot type is
never varied at all.

That matters precisely because the point of trying several mappings is
that a net with interchangeable slot types can take a fragment on the
wrong orbit. A structure whose correct assignment differs in an early
slot type is then recorded as a failure when a mapping inside the
budget would have worked.

``graded_indices`` enumerates by **ascending total index** instead: the
all-first assignment, then every one-step variation (one per slot type,
in turn), then every two-step one, and so on. The budget is spent on
the combinations nearest the reference assignment in *every* direction
rather than in one. The first combination is unchanged, so a driver
that only ever needed one attempt behaves exactly as before.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterator, Sequence

__all__ = ["graded_indices", "n_combinations"]


def n_combinations(sizes: Sequence[int]) -> int:
    """Total number of assignments, for reporting what was dropped."""
    return math.prod(sizes) if sizes else 0


def graded_indices(sizes: Sequence[int], limit: int) -> Iterator[tuple[int, ...]]:
    """Index tuples in ascending total-index order, at most ``limit``.

    Parameters
    ----------
    sizes : sequence of int
        Number of options available for each slot type, in order.
    limit : int
        Maximum number of tuples to yield.

    Yields
    ------
    tuple[int, ...]
        One index per slot type. The first yield is always all-zero;
        ties at equal total index break lexicographically, so the
        enumeration is deterministic.

    Notes
    -----
    Grown lazily from the all-zero tuple through single-step
    increments, so the full product is never materialized: a net with
    a combinatorially large assignment space costs the budget, not the
    product.
    """
    if not sizes or any(size <= 0 for size in sizes) or limit <= 0:
        return
    start = (0,) * len(sizes)
    heap: list[tuple[int, tuple[int, ...]]] = [(0, start)]
    seen: set[tuple[int, ...]] = {start}
    emitted = 0
    while heap and emitted < limit:
        _rank, current = heapq.heappop(heap)
        yield current
        emitted += 1
        for axis, size in enumerate(sizes):
            if current[axis] + 1 >= size:
                continue
            nxt = current[:axis] + (current[axis] + 1,) + current[axis + 1 :]
            if nxt in seen:
                continue
            seen.add(nxt)
            heapq.heappush(heap, (sum(nxt), nxt))
