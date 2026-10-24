"""Tests for one-dimensional bin packing."""

from __future__ import annotations

import pytest

from combinatorial import binpacking, instances

HEURISTICS = [
    binpacking.first_fit,
    binpacking.best_fit,
    binpacking.first_fit_decreasing,
    binpacking.best_fit_decreasing,
]


def minimum_bins(sizes, capacity):
    """Fewest bins, by exhaustive search over assignments."""
    order = sorted(sizes, reverse=True)
    best = len(order)

    def search(position, loads):
        nonlocal best
        if len(loads) >= best:
            return
        if position == len(order):
            best = min(best, len(loads))
            return
        tried = set()
        for index, load in enumerate(loads):
            if load + order[position] <= capacity and load not in tried:
                tried.add(load)
                loads[index] = load + order[position]
                search(position + 1, loads)
                loads[index] = load
        loads.append(order[position])
        search(position + 1, loads)
        loads.pop()

    search(0, [])
    return best


def assert_valid_packing(bins, sizes, capacity):
    packed = sorted(index for b in bins for index in b)
    assert packed == list(range(len(sizes)))
    for weight in binpacking.bin_weights(bins, sizes):
        assert weight <= capacity


@pytest.mark.parametrize("heuristic", HEURISTICS)
def test_heuristics_produce_valid_packings(heuristic):
    for seed in range(10):
        sizes, capacity = instances.random_bin_packing(25, seed=seed)
        bins = heuristic(sizes, capacity)
        assert_valid_packing(bins, sizes, capacity)


@pytest.mark.parametrize("heuristic", HEURISTICS)
def test_heuristics_are_within_the_published_bounds(heuristic):
    for seed in range(20):
        sizes, capacity = instances.random_bin_packing(12, seed=seed, capacity=100)
        best = minimum_bins(sizes, capacity)
        count = len(heuristic(sizes, capacity))
        assert count >= best
        assert count <= 17 / 10 * best + 2  # first fit / best fit, Johnson et al. 1974


def test_first_fit_decreasing_meets_the_eleven_ninths_bound():
    for seed in range(30):
        sizes, capacity = instances.random_bin_packing(14, seed=seed, capacity=100)
        best = minimum_bins(sizes, capacity)
        count = len(binpacking.first_fit_decreasing(sizes, capacity))
        assert count <= 11 / 9 * best + 4


def test_decreasing_variants_are_not_worse_than_the_plain_ones():
    better_or_equal = 0
    for seed in range(20):
        sizes, capacity = instances.random_bin_packing(30, seed=seed)
        plain = len(binpacking.first_fit(sizes, capacity))
        sorted_ = len(binpacking.first_fit_decreasing(sizes, capacity))
        assert sorted_ <= plain or sorted_ == plain
        better_or_equal += sorted_ <= plain
    assert better_or_equal == 20


def test_lower_bound_is_never_above_the_optimum():
    for seed in range(20):
        sizes, capacity = instances.random_bin_packing(12, seed=seed, capacity=100)
        bound = binpacking.lower_bound(sizes, capacity)
        best = minimum_bins(sizes, capacity)
        assert bound <= best


def test_lower_bound_counts_volume_and_oversized_items():
    assert binpacking.lower_bound([60, 60, 60], 100) == 3  # three items over half a bin
    assert binpacking.lower_bound([51, 51], 100) == 2  # volume: 102/100, and both are big
    assert binpacking.lower_bound([], 100) == 0


def test_exact_finds_the_minimum():
    for seed in range(25):
        sizes, capacity = instances.random_bin_packing(10, seed=seed, capacity=100)
        bins = binpacking.exact_bins(sizes, capacity)
        assert_valid_packing(bins, sizes, capacity)
        assert len(bins) == minimum_bins(sizes, capacity)


def test_exact_is_never_worse_than_the_heuristics():
    for seed in range(25):
        sizes, capacity = instances.random_bin_packing(12, seed=seed, capacity=100)
        exact = len(binpacking.exact_bins(sizes, capacity))
        for heuristic in HEURISTICS:
            assert exact <= len(heuristic(sizes, capacity))


def test_exact_beats_the_decreasing_heuristics_on_a_known_instance():
    """A case where both decreasing heuristics use 9 bins but 8 suffice.

    The volume lower bound is 8 and the search attains it, so this checks that
    the exact solver reaches the bound rather than merely beating the heuristics.
    """
    sizes = [29, 36, 60, 26, 40, 56, 30, 21, 46, 46, 24, 26, 28, 40, 50, 57, 48, 46, 33, 32]
    capacity = 100
    assert binpacking.lower_bound(sizes, capacity) == 8
    assert len(binpacking.first_fit_decreasing(sizes, capacity)) == 9
    assert len(binpacking.best_fit_decreasing(sizes, capacity)) == 9
    exact = binpacking.exact_bins(sizes, capacity)
    assert_valid_packing(exact, sizes, capacity)
    assert len(exact) == 8
    assert len(exact) == minimum_bins(sizes, capacity)


def test_exact_handles_an_empty_list():
    assert binpacking.exact_bins([], 10) == []


def test_items_larger_than_the_bin_are_rejected():
    for heuristic in HEURISTICS + [binpacking.lower_bound, binpacking.exact_bins]:
        with pytest.raises(ValueError):
            heuristic([11], 10)


def test_non_positive_capacity_is_rejected():
    with pytest.raises(ValueError):
        binpacking.first_fit([1], 0)
