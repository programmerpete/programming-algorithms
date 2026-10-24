"""Exact algorithms for the 0/1 knapsack problem.

Four exact solvers with complementary costs, so the right one can be picked from
the shape of the instance:

============================  ==================  ==================================
Solver                        Cost                Best when
============================  ==================  ==================================
``dp_by_capacity``            O(n c)              the capacity is moderate
``dp_by_profit``              O(n P)              profits are small
``meet_in_the_middle``        O(2^(n/2))          few items, huge capacity
``branch_and_bound``          exponential         large n, weak correlation
============================  ==================  ==================================

``P`` is the sum of all profits, ``c`` the capacity.  The first three are exact
in the worst case; branch and bound is exponential in theory but is the only one
of the four that solves the large, weakly correlated instances used as
benchmarks (Pisinger 2005).

References
----------
Bellman, R. (1957), as in :mod:`combinatorial.knapsack.dp`.
Dantzig, G. B. (1957), for the fractional bound used by branch and bound.
Horowitz, E. & Sahni, S. (1974). Computing partitions with applications to the
    knapsack problem. *Journal of the ACM* 21(2), 277-292.
    (Meet in the middle, and the dominance pruning of the half-enumerations.)
Martello, S. & Toth, P. (1990). *Knapsack Problems: Algorithms and Computer
    Implementations*. Wiley.  (The branch-and-bound framework and its bounds.)
Martello, S., Pisinger, D. & Toth, P. (1999). Dynamic programming and strong
    bounds for the 0-1 knapsack problem. *Management Science* 45(3), 414-424.
Pisinger, D. (1997). A minimal algorithm for the 0-1 knapsack problem.
    *Operations Research* 45(5), 758-767.
"""

from __future__ import annotations

from bisect import bisect_right

from .approx import half_approximation
from .dp import capacity_dp_solution, profit_dp_solution
from .models import Knapsack, Solution, density_order, solution_from_indices


def dp_by_capacity(knapsack: Knapsack) -> Solution:
    """Bellman's dynamic program over the capacity axis.

    Pseudo-polynomial: ``O(n c)`` time and ``O(c)`` space, plus ``O(n c log n)``
    for recovering the items without storing the decision table.  This is the
    algorithm to use when the capacity is small; it is also what makes the
    problem only *weakly* NP-hard.
    """
    return capacity_dp_solution(knapsack)


def dp_by_profit(knapsack: Knapsack) -> Solution:
    """Dynamic program over the profit axis: ``O(n P)`` time and ``O(P)`` space.

    The mirror image of :func:`dp_by_capacity`, and the better choice when the
    profits are small but the capacity is not.  Scaling the profits down is what
    turns this into the FPTAS (:func:`combinatorial.knapsack.approx.fptas`).
    """
    return profit_dp_solution(knapsack)


def meet_in_the_middle(knapsack: Knapsack) -> Solution:
    """Exact solution by enumerating both halves and combining their frontiers.

    Split the items into two halves.  For each half, enumerate the feasible
    subsets and keep only its Pareto frontier: a subset is discarded when another
    is both lighter and at least as profitable.  Then, for every state of the left
    half, binary-search the right frontier for the most profitable state that
    still fits.

    ``O(2^(n/2))`` time and space in the worst case (Horowitz & Sahni 1974); the
    dominance pruning is what keeps it small on correlated instances.  In Python
    it is practical to roughly 40 items, and it is the only exact method here
    that does not care how large the capacity is.
    """
    free = knapsack.free()
    indices = list(knapsack.usable())
    capacity = knapsack.capacity
    if not indices:
        return solution_from_indices(knapsack, free)

    middle = len(indices) // 2
    left_indices, right_indices = indices[:middle], indices[middle:]
    left = _pareto_frontier(knapsack, left_indices, capacity)
    right = _pareto_frontier(knapsack, right_indices, capacity)

    best_value = -1
    best_left_mask = best_right_mask = 0
    for weight, profit, mask in zip(*left):
        position = bisect_right(right[0], capacity - weight) - 1
        if position < 0:
            continue
        total = profit + right[1][position]
        if total > best_value:
            best_value = total
            best_left_mask, best_right_mask = mask, right[2][position]

    chosen = (
        list(free)
        + _expand(left_indices, best_left_mask)
        + _expand(right_indices, best_right_mask)
    )
    return solution_from_indices(knapsack, chosen)


def branch_and_bound(knapsack: Knapsack, initial: Solution | None = None) -> Solution:
    """Depth-first branch and bound with the Dantzig (fractional) bound.

    Items are ordered by non-increasing profit/weight ratio, so the fractional
    fill of the items not yet decided is a valid upper bound at every node::

        U(i, r) = value + (profits of items i..k-1, which all fit)
                         + (r - weight of those) * p_k / w_k

    where ``k`` is the first item that no longer fits and ``r`` the capacity left.
    Prefix sums make each bound ``O(log n)``.  The search prunes whenever the
    bound cannot beat the incumbent, takes all remaining items at once when they
    fit, and starts from the 1/2-approximation so that pruning bites immediately.
    """
    free = knapsack.free()
    indices = list(knapsack.usable())
    capacity = knapsack.capacity
    if not indices:
        return solution_from_indices(knapsack, free)

    order = density_order(knapsack, indices)
    weights = [knapsack.items[i].weight for i in order]
    profits = [knapsack.items[i].profit for i in order]
    n = len(order)

    prefix_weight = [0] * (n + 1)
    prefix_profit = [0] * (n + 1)
    for k in range(n):
        prefix_weight[k + 1] = prefix_weight[k] + weights[k]
        prefix_profit[k + 1] = prefix_profit[k] + profits[k]

    incumbent = initial if initial is not None else half_approximation(knapsack)
    best_value = incumbent.value
    best_items = list(incumbent.items)

    def bound(index: int, remaining: int, value: int) -> float:
        limit = prefix_weight[index] + remaining
        k = bisect_right(prefix_weight, limit, lo=index) - 1
        total = value + prefix_profit[k] - prefix_profit[index]
        if k < n:
            total += (limit - prefix_weight[k]) * profits[k] / weights[k]
        return total

    taken = [False] * n

    def search(index: int, remaining: int, value: int) -> None:
        nonlocal best_value, best_items

        if index == n:
            if value > best_value:
                best_value = value
                best_items = list(free) + [order[i] for i in range(n) if taken[i]]
            return
        if value + (prefix_profit[n] - prefix_profit[index]) <= best_value:
            return  # even taking every remaining item cannot beat the incumbent
        if bound(index, remaining, value) <= best_value:
            return
        if prefix_weight[n] - prefix_weight[index] <= remaining:
            total = value + prefix_profit[n] - prefix_profit[index]
            best_value = total
            best_items = (
                list(free)
                + [order[i] for i in range(index, n)]
                + [order[i] for i in range(index) if taken[i]]
            )
            return

        # Explore "take" first so that good solutions are found early.
        if weights[index] <= remaining:
            taken[index] = True
            search(index + 1, remaining - weights[index], value + profits[index])
            taken[index] = False
        search(index + 1, remaining, value)

    search(0, capacity, 0)
    return solution_from_indices(knapsack, best_items)


def unbounded_dp(knapsack: Knapsack) -> Solution:
    """Unbounded knapsack: every item may be used any number of times.

    ``f[j] = max(f[j-1], max over items with w_i <= j of f[j - w_i] + p_i)``, an
    ``O(n c)`` dynamic program over capacities, with the chosen item stored per
    capacity so the selection can be read back directly.  A zero-weight item with
    positive profit makes the problem unbounded and is rejected.
    """
    if any(item.weight == 0 and item.profit > 0 for item in knapsack.items):
        raise ValueError(
            "a zero-weight item with positive profit can be taken infinitely often"
        )

    capacity = knapsack.capacity
    values = [0] * (capacity + 1)
    choice = [-1] * (capacity + 1)
    for j in range(1, capacity + 1):
        best, pick = values[j - 1], -1
        for i, item in enumerate(knapsack.items):
            if item.weight <= j:
                candidate = values[j - item.weight] + item.profit
                if candidate > best:
                    best, pick = candidate, i
        values[j], choice[j] = best, pick

    chosen: list[int] = []
    j = capacity
    while j > 0:
        i = choice[j]
        if i < 0:
            j -= 1
        else:
            chosen.append(i)
            j -= knapsack.items[i].weight
    return solution_from_indices(knapsack, chosen)


def _pareto_frontier(
    knapsack: Knapsack, indices: list[int], capacity: int
) -> tuple[list[int], list[int], list[int]]:
    """Pareto frontier of the feasible subsets of ``indices``.

    Returns parallel lists ``(weights, profits, masks)`` sorted by weight, with
    strictly increasing profits, so that the most profitable state that fits a
    given capacity is found by binary search.
    """
    frontier = [(0, 0, 0)]  # weight, profit, bitmask over indices
    for bit, index in enumerate(indices):
        item = knapsack.items[index]
        extended = [
            (weight + item.weight, profit + item.profit, mask | 1 << bit)
            for weight, profit, mask in frontier
            if weight + item.weight <= capacity
        ]
        frontier = _prune(frontier + extended)
    return (
        [state[0] for state in frontier],
        [state[1] for state in frontier],
        [state[2] for state in frontier],
    )


def _prune(states: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Drop every state dominated by a lighter one that is at least as profitable."""
    states.sort(key=lambda state: (state[0], -state[1]))
    kept: list[tuple[int, int, int]] = []
    best_profit = -1
    for state in states:
        if state[1] > best_profit:
            kept.append(state)
            best_profit = state[1]
    return kept


def _expand(indices: list[int], mask: int) -> list[int]:
    """Indices selected by a bitmask over ``indices``."""
    return [index for bit, index in enumerate(indices) if mask >> bit & 1]
