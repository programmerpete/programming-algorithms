"""Dynamic programming for the 0/1 knapsack problem.

Two value functions and one reconstruction routine.

The value functions are the textbook recurrences, each stated in the form that
matches the dimension it is indexed by:

* :func:`max_profit_values` indexes by capacity and stores the best profit,
  ``O(n c)`` time and ``O(c)`` space;
* :func:`min_weight_values` indexes by profit and stores the least weight,
  ``O(n P)`` time and ``O(P)`` space, which pays off when profits are small.

Neither returns the selected items.  The usual way to recover them is to keep an
``n x c`` table of decisions, which costs as much memory as the whole value
function; instead :func:`_reconstruct` recovers an optimal set by divide and
conquer over the index dimension, using ``O(c)`` additional memory and a
``log n`` factor in time.  That is the same trick as Hirschberg's linear-space
sequence alignment.

References
----------
Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
Dantzig, G. B. (1957). Discrete-variable extremum problems. *Operations
    Research* 5(2), 266-288.
Kellerer, Pferschy & Pisinger (2004), chapter 5, for both recurrences and their
    complexity in terms of ``c`` and ``P = sum_j p_j``.
Hirschberg, D. S. (1975). A linear space algorithm for computing maximal common
    subsequences. *Communications of the ACM* 18(6), 341-343.
"""

from __future__ import annotations

from typing import Callable, Sequence

from .models import Knapsack, Solution, solution_from_indices


def max_profit_values(
    weights: Sequence[int], profits: Sequence[int], capacity: int
) -> list[int]:
    """``f[j]`` = best profit achievable with total weight at most ``j``.

    The item loop runs downwards so that each item is used at most once::

        f[j] = max(f[j], f[j - w_i] + p_i)      for j = capacity .. w_i
    """
    values = [0] * (capacity + 1)
    for weight, profit in zip(weights, profits):
        for j in range(capacity, weight - 1, -1):
            candidate = values[j - weight] + profit
            if candidate > values[j]:
                values[j] = candidate
    return values


def min_weight_values(
    weights: Sequence[int], profits: Sequence[int], total_profit: int
) -> list[float]:
    """``g[p]`` = least weight of a subset with profit exactly ``p``, else infinity."""
    infinity = float("inf")
    values = [infinity] * (total_profit + 1)
    values[0] = 0.0
    for weight, profit in zip(weights, profits):
        for p in range(total_profit, profit - 1, -1):
            candidate = values[p - profit] + weight
            if candidate < values[p]:
                values[p] = candidate
    return values


def _reconstruct(
    indices: Sequence[int],
    axis: int,
    values: Callable[[Sequence[int], int], list],
    minimize: bool,
) -> list[int]:
    """Recover an optimal subset for ``(indices, axis)`` without a decision table.

    ``values(subset, axis)`` returns the value function of the given subset, so
    the same routine serves both recurrences.  Optimality splits over any
    bipartition of the items::

        OPT(items, axis) = best over a of combine(values(left, a), values(right, axis - a))

    because the weight (or profit) of a solution splits between the two halves.
    Recurse on the two halves with the capacity each one received; the axis
    shrinks geometrically, so the total work is ``O(n axis log n)`` and the
    memory ``O(axis)``.
    """
    infinity = float("inf")
    chosen: list[int] = []
    stack: list[tuple[list[int], int]] = [(list(indices), axis)]

    while stack:
        subset, ax = stack.pop()
        if not subset or ax <= 0:
            continue

        if len(subset) == 1:
            objective = values(subset, ax)[ax]
            # Take the item only if it beats taking nothing.
            empty = 0.0 if (not minimize or ax == 0) else infinity
            improves = objective < empty if minimize else objective > empty
            if improves:
                chosen.append(subset[0])
            continue

        middle = len(subset) // 2
        left, right = subset[:middle], subset[middle:]
        left_values = values(left, ax)
        right_values = values(right, ax)
        best = infinity if minimize else float("-inf")
        best_split = -1
        for a in range(ax + 1):
            candidate = left_values[a] + right_values[ax - a]
            improves = candidate < best if minimize else candidate > best
            if improves:
                best, best_split = candidate, a
        if best_split < 0:
            continue  # no feasible split: this subproblem is infeasible

        stack.append((left, best_split))
        stack.append((right, ax - best_split))

    return chosen


def capacity_dp_solution(knapsack: Knapsack, profits: Sequence[int] | None = None) -> Solution:
    """Optimal solution of the capacity-indexed dynamic program.

    ``profits`` may be *scaled* profits, which is what the FPTAS uses; the
    returned solution always reports the true profit of the selected items.

    Zero-weight items with positive profit are set aside and added back at the
    end: they are always taken, and leaving them in would break the assumption
    that a subproblem with capacity 0 is worth nothing.
    """
    axis_profits = list(knapsack.profits if profits is None else profits)
    free = knapsack.free()
    excluded = set(free)
    indices = [i for i in knapsack.fitting() if i not in excluded]
    if not indices:
        return solution_from_indices(knapsack, free)

    weights = knapsack.weights
    capacity = knapsack.capacity

    def values(subset: Sequence[int], axis: int) -> list[int]:
        return max_profit_values(
            [weights[i] for i in subset], [axis_profits[i] for i in subset], axis
        )

    chosen = _reconstruct(indices, capacity, values, False)
    return solution_from_indices(knapsack, list(free) + chosen)


def profit_dp_solution(knapsack: Knapsack) -> Solution:
    """Optimal solution of the profit-indexed dynamic program.

    The best profit is the largest ``p`` whose least weight fits in the knapsack;
    the corresponding subset is then recovered as above.  Zero-weight items with
    positive profit are set aside, as in :func:`capacity_dp_solution`.
    """
    free = knapsack.free()
    excluded = set(free)
    indices = [i for i in knapsack.fitting() if i not in excluded]
    if not indices:
        return solution_from_indices(knapsack, free)

    weights = knapsack.weights
    profits = knapsack.profits
    capacity = knapsack.capacity

    def values(subset: Sequence[int], axis: int) -> list[float]:
        return min_weight_values(
            [weights[i] for i in subset], [profits[i] for i in subset], axis
        )

    total_profit = sum(profits[i] for i in indices)
    reachable = values(indices, total_profit)
    best_profit = max(p for p in range(total_profit + 1) if reachable[p] <= capacity)
    if best_profit == 0:
        return solution_from_indices(knapsack, free)

    chosen = _reconstruct(indices, best_profit, values, True)
    return solution_from_indices(knapsack, list(free) + chosen)
