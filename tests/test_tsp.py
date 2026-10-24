"""Tests for the travelling salesman solvers.

The exact solvers are checked against brute force, and the approximation
algorithms are checked against their published worst-case guarantees, on
instances where those guarantees apply (Euclidean distances are metric).
"""

from __future__ import annotations

import random

import pytest

from combinatorial import instances, tsp


def optimum(distance):
    """Optimal tour length: brute force where it is affordable, else Held-Karp."""
    if len(distance) <= 10:
        return tsp.tour_length(distance, tsp.brute_force_tour(distance))
    return tsp.tour_length(distance, tsp.held_karp(distance))


# --------------------------------------------------------------------------- exact


@pytest.mark.parametrize("n", [4, 5, 6, 7, 8])
def test_held_karp_matches_brute_force_on_random_instances(n):
    for seed in range(5):
        distance = instances.random_symmetric_tsp(n, seed=seed)
        tour = tsp.held_karp(distance)
        assert tsp.tour_length(distance, tour) == pytest.approx(optimum(distance))


@pytest.mark.parametrize("n", [4, 6, 8])
def test_held_karp_matches_brute_force_on_asymmetric_instances(n):
    for seed in range(5):
        distance = instances.random_asymmetric_tsp(n, seed=seed)
        tour = tsp.held_karp(distance)
        assert tsp.tour_length(distance, tour) == pytest.approx(optimum(distance))


@pytest.mark.parametrize("n", [5, 6, 7, 8, 9, 10])
def test_branch_and_bound_matches_brute_force(n):
    for seed in range(4):
        distance = instances.random_euclidean_tsp(n, seed=seed)
        tour = tsp.branch_and_bound(distance)
        tsp.validate_tour(distance, tour)
        assert tsp.tour_length(distance, tour) == pytest.approx(optimum(distance))


def test_branch_and_bound_accepts_a_warm_start():
    distance = instances.random_euclidean_tsp(9, seed=42)
    tour = tsp.branch_and_bound(distance, upper_bound=tsp.nearest_neighbour(distance))
    assert tsp.tour_length(distance, tour) == pytest.approx(optimum(distance))


def test_exact_solvers_handle_degenerate_sizes():
    for n in (1, 2, 3):
        distance = instances.random_symmetric_tsp(n, seed=n)
        assert sorted(tsp.held_karp(distance)) == list(range(n))
        assert sorted(tsp.branch_and_bound(distance)) == list(range(n))


def test_brute_force_refuses_large_instances():
    with pytest.raises(ValueError):
        tsp.brute_force_tour(instances.random_symmetric_tsp(12, seed=0))


# ------------------------------------------------------------------- constructors

CONSTRUCTORS = [
    tsp.nearest_neighbour,
    tsp.nearest_insertion,
    tsp.greedy_edge,
    tsp.mst_double_tree,
    tsp.christofides,
]


@pytest.mark.parametrize("constructor", CONSTRUCTORS)
def test_constructors_return_valid_tours(constructor):
    for n in (2, 3, 5, 9, 14):
        distance = instances.random_euclidean_tsp(n, seed=n)
        tour = constructor(distance)
        tsp.validate_tour(distance, tour)


@pytest.mark.parametrize("constructor", CONSTRUCTORS)
def test_constructors_never_beat_the_optimum(constructor):
    for seed in range(10):
        distance = instances.random_euclidean_tsp(8, seed=seed)
        assert tsp.tour_length(distance, constructor(distance)) >= optimum(distance) - 1e-9


@pytest.mark.parametrize("seed", range(30))
def test_christofides_within_three_halves(seed):
    distance = instances.random_euclidean_tsp(10, seed=seed)
    length = tsp.tour_length(distance, tsp.christofides(distance))
    assert length <= 1.5 * optimum(distance) + 1e-9


@pytest.mark.parametrize("constructor", [tsp.mst_double_tree, tsp.nearest_insertion])
@pytest.mark.parametrize("seed", range(20))
def test_two_approximations_within_factor_two(constructor, seed):
    distance = instances.random_euclidean_tsp(10, seed=seed)
    length = tsp.tour_length(distance, constructor(distance))
    assert length <= 2.0 * optimum(distance) + 1e-9


def test_nearest_insertion_beats_double_tree_on_average():
    """Nearest insertion is the better of the two 2-approximations in practice."""
    insertion = double_tree = 0.0
    for seed in range(20):
        distance = instances.random_euclidean_tsp(20, seed=seed)
        insertion += tsp.tour_length(distance, tsp.nearest_insertion(distance))
        double_tree += tsp.tour_length(distance, tsp.mst_double_tree(distance))
    assert insertion < double_tree


def test_nearest_neighbour_respects_the_start_city():
    distance = instances.random_euclidean_tsp(10, seed=3)
    tour = tsp.nearest_neighbour(distance, start=4)
    assert tour[0] == 4
    tsp.validate_tour(distance, tour)


def test_greedy_edge_produces_a_hamiltonian_path_before_closing():
    distance = instances.random_euclidean_tsp(9, seed=8)
    tour = tsp.greedy_edge(distance)
    assert len(tour) == len(set(tour)) == 9


# ------------------------------------------------------------------ local search


@pytest.mark.parametrize("start", [tsp.nearest_neighbour, tsp.christofides])
@pytest.mark.parametrize("seed", range(10))
def test_two_opt_never_worsens_and_stays_valid(start, seed):
    distance = instances.random_euclidean_tsp(15, seed=seed)
    before = start(distance)
    after = tsp.two_opt(distance, before)
    tsp.validate_tour(distance, after)
    assert tsp.tour_length(distance, after) <= tsp.tour_length(distance, before) + 1e-9


@pytest.mark.parametrize("seed", range(10))
def test_two_opt_improves_on_a_random_tour(seed):
    distance = instances.random_euclidean_tsp(12, seed=seed)
    random_order = random.Random(seed).sample(range(12), 12)
    improved = tsp.two_opt(distance, random_order)
    assert tsp.tour_length(distance, improved) < tsp.tour_length(distance, random_order)


def test_two_opt_reaches_a_local_optimum():
    """No 2-opt move may improve the returned tour, by definition of local optimum."""
    distance = instances.random_euclidean_tsp(12, seed=4)
    tour = tsp.two_opt(distance, tsp.nearest_neighbour(distance))
    n = len(tour)
    for i in range(n - 1):
        for j in range(i + 2, n):
            a, b, c, d = tour[i], tour[i + 1], tour[j], tour[(j + 1) % n]
            if a == c or b == d:
                continue
            delta = distance[a][c] + distance[b][d] - distance[a][b] - distance[c][d]
            assert delta >= -1e-9


def test_or_opt_never_worsens():
    distance = instances.random_euclidean_tsp(14, seed=6)
    before = tsp.two_opt(distance, tsp.christofides(distance))
    after = tsp.or_opt(distance, before)
    tsp.validate_tour(distance, after)
    assert tsp.tour_length(distance, after) <= tsp.tour_length(distance, before) + 1e-9


def test_local_search_chain_gets_close_to_the_optimum():
    """2-opt followed by Or-opt is the classical first improvement step."""
    distance = instances.random_euclidean_tsp(14, seed=9)
    best = tsp.or_opt(distance, tsp.two_opt(distance, tsp.christofides(distance)))
    assert tsp.tour_length(distance, best) <= 1.1 * optimum(distance)


# ------------------------------------------------------------------------- bounds


@pytest.mark.parametrize("n", [5, 8, 12])
def test_one_tree_is_a_lower_bound(n):
    for seed in range(10):
        distance = instances.random_euclidean_tsp(n, seed=seed)
        weight, _ = tsp.one_tree(distance)
        assert weight <= optimum(distance) + 1e-9


@pytest.mark.parametrize("n", [6, 10, 13])
def test_held_karp_bound_is_a_lower_bound(n):
    for seed in range(10):
        distance = instances.random_euclidean_tsp(n, seed=seed)
        bound, _ = tsp.held_karp_bound(distance, iterations=50)
        assert bound <= optimum(distance) + 1e-9


def test_penalties_strengthen_the_bound():
    """The subgradient ascent must not be worse than the plain 1-tree bound."""
    improved = 0
    for seed in range(15):
        distance = instances.random_euclidean_tsp(10, seed=seed)
        plain, _ = tsp.one_tree(distance)
        penalised, _ = tsp.held_karp_bound(distance, iterations=60)
        assert penalised >= plain - 1e-9
        if penalised > plain + 1e-9:
            improved += 1
    assert improved >= 10


def test_held_karp_bound_is_tight_on_a_small_instance():
    distance = instances.random_euclidean_tsp(7, seed=2)
    bound, _ = tsp.held_karp_bound(distance, iterations=200)
    assert bound == pytest.approx(optimum(distance), rel=0.02)


def test_bound_proves_optimality_when_it_meets_a_tour():
    """A bound equal to the incumbent tour length certifies that tour as optimal."""
    distance = instances.random_euclidean_tsp(8, seed=1)
    best = tsp.or_opt(distance, tsp.two_opt(distance, tsp.christofides(distance)))
    length = tsp.tour_length(distance, best)
    bound, _ = tsp.held_karp_bound(distance, iterations=300, upper_bound=length)
    assert bound <= length + 1e-9
    if bound == pytest.approx(length, rel=1e-9):
        assert length == pytest.approx(optimum(distance))


def test_one_tree_with_arbitrary_penalties_stays_below_the_optimum():
    """The Lagrangian inequality, with penalties that were not fitted at all.

    For *any* penalty vector, ``w(1-tree) + sum_i pi_i (deg_i - 2) <= OPT`` holds,
    because a tour is a 1-tree with every degree equal to 2.  Choosing the two
    edges at city 0 by true rather than penalised cost breaks exactly this.
    """
    for n in (6, 8, 10):
        for seed in range(6):
            distance = instances.random_symmetric_tsp(n, seed=seed * 17 + n)
            best = optimum(distance)
            for trial in range(3):
                rng = random.Random(seed * 100 + trial)
                pi = [rng.uniform(-80, 80) for _ in range(n)]
                weight, degree = tsp.one_tree(distance, pi)
                value = weight + sum(pi[i] * (degree[i] - 2) for i in range(n))
                assert value <= best + 1e-9


def test_one_tree_degrees_sum_to_twice_the_edges():
    distance = instances.random_euclidean_tsp(9, seed=0)
    weight, degree = tsp.one_tree(distance)
    assert sum(degree) == 2 * len(distance)
    assert min(degree) >= 1
    assert weight > 0
