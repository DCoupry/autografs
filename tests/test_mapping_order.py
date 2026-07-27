"""Tests for the benchmark drivers' assignment enumeration order.

``itertools.product`` advances its last iterable fastest, so a capped
budget spent on it never varies the first slot type at all. The graded
enumeration spends the same budget on the combinations nearest the
reference assignment in *every* direction.
"""

import itertools
import os
import sys

import pytest

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts", "benchmarks")


def _load(name):
    sys.path.insert(0, SCRIPTS)
    try:
        return __import__(name)
    finally:
        sys.path.pop(0)


@pytest.fixture(scope="module")
def order():
    return _load("_mapping_order")


class TestGradedIndices:
    def test_first_yield_is_the_reference_assignment(self, order):
        """Unchanged behaviour for a driver that only needs one try."""
        assert next(iter(order.graded_indices([4, 4, 4], 8))) == (0, 0, 0)

    def test_every_axis_is_varied_within_the_budget(self, order):
        """The bug this replaces: on a 4x4x4 space a budget of 8 spent
        in product order leaves the first slot type pinned at its first
        fragment, so an assignment differing there is never tried."""
        product_order = list(itertools.islice(itertools.product(*[range(4)] * 3), 8))
        assert {combo[0] for combo in product_order} == {0}  # the old bias

        graded = list(order.graded_indices([4, 4, 4], 8))
        varied = {axis for combo in graded for axis, value in enumerate(combo) if value}
        assert varied == {0, 1, 2}

    def test_ascending_total_index(self, order):
        totals = [sum(combo) for combo in order.graded_indices([3, 4, 5], 30)]
        assert totals == sorted(totals)

    def test_enumeration_is_exhaustive_and_unique(self, order):
        sizes = [3, 2, 4]
        everything = list(order.graded_indices(sizes, 10_000))
        assert len(everything) == len(set(everything))
        assert set(everything) == set(itertools.product(*(range(s) for s in sizes)))
        assert len(everything) == order.n_combinations(sizes)

    def test_deterministic(self, order):
        assert list(order.graded_indices([3, 3, 3], 12)) == list(
            order.graded_indices([3, 3, 3], 12)
        )

    def test_budget_is_respected(self, order):
        assert len(list(order.graded_indices([9, 9, 9], 5))) == 5

    @pytest.mark.parametrize(
        "sizes,limit", [([], 5), ([3, 0, 2], 5), ([3, 3], 0), ([2, 2], -1)]
    )
    def test_degenerate_inputs_yield_nothing(self, order, sizes, limit):
        assert list(order.graded_indices(sizes, limit)) == []

    def test_single_slot_type(self, order):
        assert list(order.graded_indices([3], 10)) == [(0,), (1,), (2,)]

    def test_n_combinations(self, order):
        assert order.n_combinations([3, 4, 5]) == 60
        assert order.n_combinations([]) == 0


class TestDriversUseIt:
    """Both closure drivers must go through the shared enumeration."""

    @pytest.mark.parametrize("driver", ["roundtrip", "embedding"])
    def test_driver_imports_the_shared_order(self, driver):
        module = _load(driver)
        assert module.graded_indices is _load("_mapping_order").graded_indices
