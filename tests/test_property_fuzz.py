"""Property tests: periodic-graph algorithms vs. brute-force oracles.

Two hand-rolled algorithms get oracle-fuzzed (issue #111):

- ``porosity._percolates`` (wrap-detecting union-find with path
  compression) against an independent BFS *potential* oracle: assign
  every open cell a wrap-to-root potential; any bond whose wrap
  disagrees with the recorded potentials closes a periodically
  wrapping cycle - the definition of percolation, with no union-find
  and no path compression to share bugs with.
- ``net.coordination_sequences`` (BFS over (vertex, image) states on
  the quotient graph) against plain shortest-path counts on an
  explicitly unfolded finite piece of the periodic graph. Voltages are
  drawn from {-1, 0, 1}, so a k-step walk strays at most k cells from
  the origin and a (2*shells+1)^3 box unfolds far enough that open
  boundaries cannot bias any distance <= shells.
- ``net.net_signature`` invariance under the two operations
  ``_prune_and_contract`` exists to absorb: subdividing an edge with a
  2-coordinated vertex, and hanging a 1-coordinated cap.
"""

from collections import Counter, deque

import networkx
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from autografs.net import coordination_sequences, net_signature
from autografs.porosity import _percolates

ORACLE_SHELLS = 4

voltages = st.tuples(st.integers(-1, 1), st.integers(-1, 1), st.integers(-1, 1))


def quotient_graphs(max_vertices: int = 3, max_edges: int = 6):
    """Random small labeled quotient graphs as edge multisets."""
    return st.integers(1, max_vertices).flatmap(
        lambda n: st.lists(
            st.tuples(st.integers(0, n - 1), st.integers(0, n - 1), voltages),
            min_size=1,
            max_size=max_edges,
        ).map(Counter)
    )


# ----------------------------------------------------------------------
# percolation
# ----------------------------------------------------------------------


def _oracle_percolates(open_grid: np.ndarray) -> bool:
    """BFS potential assignment; a wrap-inconsistent bond = percolation."""
    dims = open_grid.shape
    open_cells = list(
        zip(*(idx.tolist() for idx in np.nonzero(open_grid)), strict=True)
    )
    open_set = set(open_cells)
    potential: dict[tuple, tuple] = {}
    for start in open_cells:
        if start in potential:
            continue
        potential[start] = (0, 0, 0)
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            p = potential[cell]
            for axis in range(3):
                for sign in (1, -1):
                    stepped = list(cell)
                    wrap = [0, 0, 0]
                    stepped[axis] += sign
                    if stepped[axis] == dims[axis]:
                        stepped[axis] = 0
                        wrap[axis] = 1
                    elif stepped[axis] < 0:
                        stepped[axis] = dims[axis] - 1
                        wrap[axis] = -1
                    neighbor = tuple(stepped)
                    if neighbor not in open_set:
                        continue
                    q = (p[0] + wrap[0], p[1] + wrap[1], p[2] + wrap[2])
                    if neighbor not in potential:
                        potential[neighbor] = q
                        queue.append(neighbor)
                    elif potential[neighbor] != q:
                        return True
    return False


class TestPercolation:
    @settings(deadline=None)
    @given(
        grid=hnp.arrays(
            np.bool_,
            hnp.array_shapes(min_dims=3, max_dims=3, min_side=1, max_side=4),
        )
    )
    def test_matches_potential_oracle(self, grid):
        assert _percolates(grid.reshape(-1), grid.shape) == _oracle_percolates(grid)

    def test_straight_channel_percolates(self):
        grid = np.zeros((3, 3, 3), dtype=bool)
        grid[:, 1, 1] = True
        assert _percolates(grid.reshape(-1), grid.shape)

    def test_blocked_channel_does_not(self):
        grid = np.zeros((3, 3, 3), dtype=bool)
        grid[:, 1, 1] = True
        grid[1, 1, 1] = False
        assert not _percolates(grid.reshape(-1), grid.shape)


# ----------------------------------------------------------------------
# coordination sequences
# ----------------------------------------------------------------------


def _unfolded_sequence(edges: Counter, source: int, shells: int) -> tuple[int, ...]:
    """Coordination sequence by shortest paths on an explicit unfold."""
    box = range(-shells, shells + 1)
    graph = networkx.Graph()
    for a, b, voltage in edges:
        for cx in box:
            for cy in box:
                for cz in box:
                    target = (cx + voltage[0], cy + voltage[1], cz + voltage[2])
                    if all(-shells <= t <= shells for t in target):
                        graph.add_edge((a, (cx, cy, cz)), (b, target))
    lengths = networkx.single_source_shortest_path_length(
        graph, (source, (0, 0, 0)), cutoff=shells
    )
    counts = [0] * shells
    for distance in lengths.values():
        if 1 <= distance <= shells:
            counts[distance - 1] += 1
    return tuple(counts)


class TestCoordinationSequences:
    @settings(deadline=None, max_examples=60)
    @given(edges=quotient_graphs())
    def test_matches_explicit_unfold(self, edges):
        sequences = coordination_sequences(edges, shells=ORACLE_SHELLS)
        for source, sequence in sequences.items():
            assert sequence == _unfolded_sequence(edges, source, ORACLE_SHELLS)


# ----------------------------------------------------------------------
# signature invariance (the properties _prune_and_contract must grant)
# ----------------------------------------------------------------------


class TestSignatureInvariance:
    @settings(deadline=None)
    @given(edges=quotient_graphs(), data=st.data())
    def test_edge_subdivision_is_invisible(self, edges, data):
        """Splitting any edge with a 2-coordinated vertex must not
        change the contracted signature: an edge center and the edge
        it decorates are the same edge of the underlying net."""
        a, b, voltage = data.draw(st.sampled_from(sorted(edges)))
        new = max(n for e in edges for n in e[:2]) + 1
        subdivided = Counter(edges)
        subdivided[(a, b, voltage)] -= 1
        if not subdivided[(a, b, voltage)]:
            del subdivided[(a, b, voltage)]
        subdivided[(a, new, (0, 0, 0))] += 1
        subdivided[(new, b, voltage)] += 1
        assert net_signature(subdivided, shells=ORACLE_SHELLS) == net_signature(
            edges, shells=ORACLE_SHELLS
        )

    @settings(deadline=None)
    @given(edges=quotient_graphs(), data=st.data())
    def test_caps_are_invisible(self, edges, data):
        """A dangling 1-coordinated vertex on any node must not change
        the signature: capping ligands carry no net information."""
        node = data.draw(st.sampled_from(sorted({n for e in edges for n in e[:2]})))
        voltage = data.draw(voltages)
        new = max(n for e in edges for n in e[:2]) + 1
        capped = Counter(edges)
        capped[(node, new, voltage)] += 1
        assert net_signature(capped, shells=ORACLE_SHELLS) == net_signature(
            edges, shells=ORACLE_SHELLS
        )
