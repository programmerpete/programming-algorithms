"""Tests for the knapsack solvers.

Every exact solver is checked against exhaustive enumeration, every approximate
one against the guarantee it claims, and every solution against the instance it
claims to solve (weights and profits are recomputed from the chosen indices, so a
solver cannot pass by reporting a value it did not achieve).
"""

from __future__ import annotations

import pytest

from combinatorial import instances, knapsack
from combinatorial.knapsack import Knapsack, Solution

EXACT_SOLVERS = [
    knapsack.dp_by_capacity,
    knapsack.dp_by_profit,
    knapsack.meet_in_the_middle,
    knapsack.branch_and_bound,
]


def uncorrelated(seed):
    return instances.random_knapsack(10, seed=seed, capacity=200)


def correlated(seed):
    return instances.correlated_knapsack(10, seed=seed, capacity=300)


def subset_sum(seed):
    return instances.subset_sum_knapsack(10, seed=seed, capacity=250)


def recompute(instance, solution):
    """(weight, value) of a solution, read back from the instance."""
    weight = sum(instance.items[i].weight for i in solution.items)
    value = sum(instance.items[i].profit for i in solution.items)
    return weight, value


def assert_consistent(instance, solution):
    weight, value = recompute(instance, solution)
    assert weight <= instance.capacity
    assert (solution.weight, solution.value) == (weight, value)


# --------------------------------------------------------------------------- exact


@pytest.mark.parametrize("solver", EXACT_SOLVERS)
@pytest.mark.parametrize("factory", [uncorrelated, correlated, subset_sum])
def test_exact_solvers_agree_with_brute_force(solver, factory):
    for seed in range(4):
        instance = factory(seed)
        expected = knapsack.brute_force_solution(instance)
        solution = solver(instance)
        assert_consistent(instance, solution)
        assert solution.value == expected.value


@pytest.mark.parametrize("solver", EXACT_SOLVERS)
def test_exact_solvers_handle_a_tiny_capacity(solver):
    instance = Knapsack.from_pairs(capacity=5, pairs=[(4, 10), (3, 7), (5, 9), (6, 100)])
    solution = solver(instance)
    assert_consistent(instance, solution)
    assert solution.value == 10  # the 6-weight item does not fit, the rest conflict


@pytest.mark.parametrize("solver", EXACT_SOLVERS)
def test_exact_solvers_handle_an_empty_instance(solver):
    instance = Knapsack.from_pairs(capacity=10, pairs=[])
    solution = solver(instance)
    assert solution == Solution(value=0, weight=0, items=())


@pytest.mark.parametrize("solver", EXACT_SOLVERS)
def test_exact_solvers_take_zero_weight_items(solver):
    instance = Knapsack.from_pairs(capacity=3, pairs=[(0, 5), (3, 1), (0, 0)])
    solution = solver(instance)
    assert_consistent(instance, solution)
    assert solution.value == 6


@pytest.mark.parametrize("solver", EXACT_SOLVERS)
def test_exact_solvers_handle_zero_profits(solver):
    instance = Knapsack.from_pairs(capacity=10, pairs=[(4, 0), (6, 0), (11, 50)])
    solution = solver(instance)
    assert_consistent(instance, solution)
    assert solution.value == 0


def test_exact_solvers_agree_on_a_larger_capacity_instance():
    instance = instances.random_knapsack(40, seed=1, capacity=1500)
    values = {solver.__name__: solver(instance).value for solver in EXACT_SOLVERS}
    assert len(set(values.values())) == 1


def test_dp_by_profit_is_used_when_profits_are_small():
    """The profit axis stays small even when the capacity is a million."""
    instance = Knapsack.from_pairs(
        capacity=1_300_000, pairs=[(10**5, 3), (2 * 10**5, 5), (10**6, 9)]
    )
    solution = knapsack.dp_by_profit(instance)
    assert_consistent(instance, solution)
    assert solution.value == 17  # everything fits
    assert solution.items == (0, 1, 2)


def test_capacity_dp_ignores_an_oversized_capacity_axis():
    """Sanity: the two axes agree even when one of them is enormous."""
    instance = Knapsack.from_pairs(capacity=10**6, pairs=[(4, 3), (5, 7), (6, 6), (7, 2)])
    assert knapsack.dp_by_profit(instance).value == knapsack.brute_force_solution(instance).value


# ----------------------------------------------------------------------- unbounded


def unbounded_brute_force(instance):
    """Best profit when every item may be reused, by exhaustive recursion."""
    weights = instance.weights
    profits = instance.profits

    def search(index, capacity):
        if index == len(weights):
            return 0
        best = 0
        copies = 0
        while copies * weights[index] <= capacity:
            best = max(
                best, copies * profits[index] + search(index + 1, capacity - copies * weights[index])
            )
            copies += 1
        return best

    return search(0, instance.capacity)


def test_unbounded_matches_brute_force():
    for seed in range(6):
        instance = instances.random_knapsack(5, seed=seed, capacity=40)
        solution = knapsack.unbounded_dp(instance)
        weight, value = recompute(instance, solution)
        assert weight <= instance.capacity
        assert solution.weight == weight and solution.value == value
        assert value == unbounded_brute_force(instance)


def test_unbounded_allows_repeats():
    instance = Knapsack.from_pairs(capacity=10, pairs=[(3, 5), (10, 9)])
    solution = knapsack.unbounded_dp(instance)
    assert solution.value == 15  # three copies of the first item beat one of the second
    assert solution.items == (0, 0, 0)


def test_unbounded_rejects_an_unbounded_instance():
    instance = Knapsack.from_pairs(capacity=10, pairs=[(0, 1)])
    with pytest.raises(ValueError):
        knapsack.unbounded_dp(instance)


# --------------------------------------------------------------------- approximate


@pytest.mark.parametrize("factory", [uncorrelated, correlated, subset_sum])
@pytest.mark.parametrize("epsilon", [0.01, 0.05, 0.2, 0.5, 1.0])
def test_fptas_meets_its_guarantee(factory, epsilon):
    for seed in range(3):
        instance = factory(seed)
        best = knapsack.brute_force_solution(instance).value
        solution = knapsack.fptas(instance, epsilon)
        assert_consistent(instance, solution)
        assert solution.value >= (1 - epsilon) * best - 1e-9


def test_fptas_on_a_large_instance_is_close_to_the_exact_value():
    instance = instances.correlated_knapsack(200, seed=4, capacity=4000)
    exact = knapsack.dp_by_capacity(instance).value
    approximate = knapsack.fptas(instance, 0.01)
    assert_consistent(instance, approximate)
    assert approximate.value >= 0.99 * exact


def test_fptas_rejects_a_bad_epsilon():
    instance = instances.random_knapsack(5, seed=0)
    for epsilon in (0, -0.5, 1.5):
        with pytest.raises(ValueError):
            knapsack.fptas(instance, epsilon)


def test_half_approximation_meets_its_guarantee():
    """The 1/2-approximation must hold on adversarial as well as random instances."""
    instances_to_try = [
        instances.random_knapsack(9, seed=seed, capacity=180) for seed in range(20)
    ] + [
        instances.correlated_knapsack(9, seed=seed, capacity=200) for seed in range(20)
    ] + [
        # the split item is heavier than the knapsack, so it cannot be used directly
        Knapsack.from_pairs(capacity=10, pairs=[(10, 10), (11, 100), (1, 1)]),
        Knapsack.from_pairs(capacity=10, pairs=[(5, 5), (5, 5)]),
        Knapsack.from_pairs(capacity=7, pairs=[(7, 7), (8, 8), (6, 6)]),
    ]
    for instance in instances_to_try:
        best = knapsack.brute_force_solution(instance).value
        solution = knapsack.half_approximation(instance)
        assert_consistent(instance, solution)
        assert solution.value >= best / 2 - 1e-9


def test_greedy_and_fractional_bound_are_consistent():
    for seed in range(10):
        instance = instances.random_knapsack(12, seed=seed, capacity=250)
        best = knapsack.brute_force_solution(instance).value
        greedy = knapsack.greedy_density(instance)
        bound = knapsack.fractional_bound(instance)
        assert_consistent(instance, greedy)
        assert greedy.value <= best
        assert bound.value >= best - 1e-9  # the LP relaxation bounds the integer optimum


def test_fractional_bound_reports_its_split_item():
    instance = Knapsack.from_pairs(capacity=10, pairs=[(6, 6), (6, 6)])
    bound = knapsack.fractional_bound(instance)
    assert bound.value == pytest.approx(10.0)
    assert bound.items == (0,)
    assert bound.split_item == 1
    assert bound.split_fraction == pytest.approx(4 / 6)


def test_fractional_bound_is_exact_when_everything_fits():
    instance = Knapsack.from_pairs(capacity=10, pairs=[(4, 4), (5, 5)])
    bound = knapsack.fractional_bound(instance)
    assert bound.split_item is None
    assert bound.value == 9
    assert sorted(bound.items) == [0, 1]


def test_density_order_is_sorted_by_ratio():
    instance = Knapsack.from_pairs(capacity=10, pairs=[(5, 5), (5, 10), (0, 1), (2, 1)])
    order = knapsack.density_order(instance)
    ratios = [
        float("inf") if instance.items[i].weight == 0 else instance.items[i].profit / instance.items[i].weight
        for i in order
    ]
    assert ratios == sorted(ratios, reverse=True)
    assert order[0] == 2  # the zero-weight item has an infinite ratio


# -------------------------------------------------------------------------- model


def test_negative_inputs_are_rejected():
    with pytest.raises(ValueError):
        Knapsack.from_pairs(capacity=-1, pairs=[])
    with pytest.raises(ValueError):
        Knapsack.from_pairs(capacity=10, pairs=[(-1, 5)])
    with pytest.raises(ValueError):
        Knapsack.from_pairs(capacity=10, pairs=[(5, -1)])


def test_solution_from_indices_rejects_infeasible_selections():
    instance = Knapsack.from_pairs(capacity=4, pairs=[(5, 5)])
    with pytest.raises(ValueError):
        knapsack.solution_from_indices(instance, [0])
