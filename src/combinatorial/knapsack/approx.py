"""Approximation algorithms and bounds for the 0/1 knapsack problem.

The 0/1 knapsack problem is only *weakly* NP-hard, so unlike the TSP it admits a
fully polynomial time approximation scheme: for any ``eps > 0`` a solution of
value at least ``(1 - eps) OPT`` can be computed in time polynomial in ``n`` and
``1/eps``.  That scheme, the simple 1/2-approximation, and the fractional (LP)
bound that everything else is measured against are collected here.

References
----------
Dantzig, G. B. (1957). Discrete-variable extremum problems. *Operations
    Research* 5(2), 266-288.  (The fractional bound and the greedy fill.)
Sahni, S. (1975). Approximate algorithms for the 0/1 knapsack problem.
    *Journal of the ACM* 22(1), 115-124.  (Polynomial-time approximation.)
Ibarra, O. H. & Kim, C. E. (1975). Fast approximation algorithms for the
    knapsack and sum of subset problems. *Journal of the ACM* 22(4), 463-468.
    (The first fully polynomial scheme, by scaling the profits.)
Lawler, E. L. (1979). Fast approximation algorithms for knapsack problems.
    *Mathematics of Operations Research* 4(4), 339-356.  (The form used here.)
Kellerer, Pferschy & Pisinger (2004), chapters 2 and 5.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dp import capacity_dp_solution
from .models import Knapsack, Solution, density_order, solution_from_indices


@dataclass(frozen=True)
class FractionalSolution:
    """Optimal solution of the LP relaxation: whole items plus one split item.

    ``value`` is the relaxation's objective, an upper bound on the integer
    optimum; ``split_item`` is the item the fill stopped on, taken with
    ``split_fraction`` of its profit, or ``None`` if everything fitted.
    """

    value: float
    items: tuple[int, ...]
    split_item: int | None
    split_fraction: float


def fractional_bound(knapsack: Knapsack) -> FractionalSolution:
    """Dantzig's fractional bound: fill by density, splitting the item that does not fit.

    Allowing one item to be taken fractionally turns the knapsack into a linear
    program whose optimum is found by a single greedy pass, and whose value is an
    upper bound on OPT.  Items heavier than the capacity are dropped first (they
    belong to no feasible solution, and keeping them would only weaken the bound),
    so the split item always fits on its own.  This is the bound the
    branch-and-bound search evaluates at every node.
    """
    order = density_order(knapsack, list(knapsack.free()) + list(knapsack.usable()))
    remaining = knapsack.capacity
    value = 0.0
    taken: list[int] = []
    split_item: int | None = None
    split_fraction = 0.0

    for index in order:
        item = knapsack.items[index]
        if item.weight <= remaining:
            taken.append(index)
            value += item.profit
            remaining -= item.weight
        else:
            split_item = index
            split_fraction = remaining / item.weight
            value += split_fraction * item.profit
            break

    return FractionalSolution(value, tuple(taken), split_item, split_fraction)


def greedy_density(knapsack: Knapsack) -> Solution:
    """Take items in decreasing profit/weight ratio whenever they still fit.

    The classical greedy fill, with no guarantee of its own; it is the first half
    of :func:`half_approximation` and the incumbent that starts the
    branch-and-bound search.
    """
    remaining = knapsack.capacity
    taken: list[int] = []
    for index in density_order(knapsack):
        item = knapsack.items[index]
        if item.weight <= remaining:
            taken.append(index)
            remaining -= item.weight
    return solution_from_indices(knapsack, taken)


def half_approximation(knapsack: Knapsack) -> Solution:
    """The better of the greedy fill and the most profitable item that fits: >= OPT/2.

    Write ``g`` for the greedy value and ``s`` for the item on which the
    fractional fill gets stuck, so that ``OPT <= g + p_s`` because the LP
    relaxation is an upper bound.  Items heavier than the capacity are dropped
    first, so ``s`` fits on its own and the best single item ``p_max`` satisfies
    ``p_max >= p_s``.  If ``p_s <= OPT/2`` then ``g >= OPT - p_s >= OPT/2``;
    otherwise the single item ``p_max >= p_s > OPT/2``.  Either way the better of
    the two candidates reaches ``OPT/2``.

    Runs in ``O(n log n)`` (the sort), and the bound is tight: an instance with two
    items of equal weight and profit ``OPT/2`` each cannot be improved on by any
    greedy rule.
    """
    greedy = greedy_density(knapsack)

    best_single = solution_from_indices(knapsack, [])
    for index in knapsack.usable():
        if knapsack.items[index].profit > best_single.value:
            best_single = solution_from_indices(knapsack, [index])

    return greedy if greedy.value >= best_single.value else best_single


def fptas(knapsack: Knapsack, epsilon: float) -> Solution:
    """Fully polynomial approximation scheme: value >= ``(1 - eps) OPT``.

    Ibarra and Kim's idea, in Lawler's form: scale the profits down by
    ``K = eps * p_max / n``, solve the scaled instance exactly with the
    profit-indexed dynamic program, and return that solution.  The dynamic program
    then has ``O(n^2 / eps)`` profit levels, and dropping the remainder of the
    division costs at most ``K`` per item::

        value  >=  K * (OPT / K - n)  =  OPT - eps * p_max  >=  (1 - eps) OPT

    because the optimal solution's scaled profit is at least ``OPT / K - n``, and
    ``p_max <= OPT``.  ``K`` is never taken below 1, which keeps the profit axis
    from growing when the profits are already small; in that case the scheme
    simply solves the instance exactly.
    """
    if not 0 < epsilon <= 1:
        raise ValueError("epsilon must lie in (0, 1]")

    indices = list(knapsack.fitting())
    if not indices:
        return solution_from_indices(knapsack, [])

    best_profit = max(knapsack.items[i].profit for i in indices)
    if best_profit == 0:
        return solution_from_indices(knapsack, [])

    scale = max(epsilon * best_profit / len(indices), 1.0)
    scaled = [int(item.profit // scale) for item in knapsack.items]
    return capacity_dp_solution(knapsack, profits=scaled)
