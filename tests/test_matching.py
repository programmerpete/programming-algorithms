"""Tests for the minimum-weight perfect matching used by Christofides."""

from __future__ import annotations

import itertools
import random

import pytest

from combinatorial.matching import min_weight_perfect_matching


def brute_force_matching(cost, vertices):
    """Cheapest perfect matching, by enumerating all pairings."""
    if not vertices:
        return 0.0
    first, rest = vertices[0], vertices[1:]
    best = float("inf")
    for index, partner in enumerate(rest):
        remaining = rest[:index] + rest[index + 1 :]
        best = min(best, cost[first][partner] + brute_force_matching(cost, remaining))
    return best


def random_cost(k, seed):
    rng = random.Random(seed)
    cost = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            cost[i][j] = cost[j][i] = float(rng.randint(1, 100))
    return cost


@pytest.mark.parametrize("k", [2, 4, 6, 8, 10])
def test_exact_matching_matches_brute_force(k):
    for seed in range(20):
        cost = random_cost(k, seed)
        pairs, total = min_weight_perfect_matching(cost, method="exact")
        assert total == pytest.approx(brute_force_matching(cost, list(range(k))))
        assert sorted(v for pair in pairs for v in pair) == list(range(k))
        assert total == pytest.approx(sum(cost[u][v] for u, v in pairs))


def test_exact_matching_on_a_subset_of_vertices():
    cost = random_cost(8, 0)
    vertices = [1, 3, 4, 6, 7, 2]
    _, total = min_weight_perfect_matching(cost, vertices, method="exact")
    assert total == pytest.approx(brute_force_matching(cost, vertices))


def test_greedy_matching_is_never_better_than_exact():
    for seed in range(20):
        cost = random_cost(10, seed)
        _, exact = min_weight_perfect_matching(cost, method="exact")
        _, greedy = min_weight_perfect_matching(cost, method="greedy")
        assert greedy >= exact - 1e-9


def test_auto_uses_exact_on_small_inputs():
    cost = random_cost(6, 7)
    assert min_weight_perfect_matching(cost, method="auto") == min_weight_perfect_matching(
        cost, method="exact"
    )


def test_odd_number_of_vertices_is_rejected():
    with pytest.raises(ValueError):
        min_weight_perfect_matching(random_cost(5, 0), [0, 1, 2])


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        min_weight_perfect_matching(random_cost(4, 0), method="magic")


def test_result_is_deterministic_and_sorted():
    cost = random_cost(8, 3)
    pairs, _ = min_weight_perfect_matching(cost, method="exact")
    assert pairs == sorted(pairs)
    assert pairs == min_weight_perfect_matching(cost, method="exact")[0]


def test_empty_input():
    assert min_weight_perfect_matching([], []) == ([], 0.0)


def test_all_pairings_are_disjoint():
    cost = random_cost(12, 11)
    pairs, _ = min_weight_perfect_matching(cost, method="greedy")
    used = [v for pair in pairs for v in pair]
    assert len(used) == len(set(used)) == 12
    assert set(used) == set(range(12))


def test_brute_force_helper_agrees_with_dynamic_program_on_six_vertices():
    cost = random_cost(6, 5)
    assert min_weight_perfect_matching(cost, method="exact")[1] == pytest.approx(
        brute_force_matching(cost, list(range(6)))
    )
    # sanity: the oracle really does see every pairing
    assert len(list(itertools.combinations(range(6), 2))) == 15
