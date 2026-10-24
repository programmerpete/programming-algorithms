"""Knapsack problem solvers.

The 0/1 problem: choose a subset of items maximising total profit subject to a
total weight of at most the capacity.  It is NP-hard (Karp 1972) but only weakly
so, which is why there are pseudo-polynomial exact algorithms, an FPTAS, and
branch-and-bound codes that solve ten-thousand-item instances.

============================  ==================  ===============================
Solver                        Guarantee           Cost
============================  ==================  ===============================
``dp_by_capacity``            exact               O(n c) time, O(c) space
``dp_by_profit``              exact               O(n P) time, O(P) space
``meet_in_the_middle``        exact               O(2^(n/2)) time and space
``branch_and_bound``          exact               exponential worst case
``unbounded_dp``              exact (unbounded)   O(n c)
``fptas``                     >= (1 - eps) OPT    O(n^2 / eps) profit levels
``half_approximation``        >= OPT/2            O(n log n)
``greedy_density``            none                O(n log n)
``fractional_bound``          upper bound on OPT  O(n log n)
============================  ==================  ===============================

``c`` is the capacity and ``P`` the sum of all profits.  Instances and solutions
are the immutable objects in :mod:`combinatorial.knapsack.models`.

Typical use::

    from combinatorial import instances, knapsack

    problem = instances.random_knapsack(50, seed=1)
    print(knapsack.branch_and_bound(problem))
    print(knapsack.fptas(problem, 0.05))
"""

from .approx import FractionalSolution, fptas, fractional_bound, greedy_density, half_approximation
from .dp import capacity_dp_solution, max_profit_values, min_weight_values, profit_dp_solution
from .exact import branch_and_bound, dp_by_capacity, dp_by_profit, meet_in_the_middle, unbounded_dp
from .models import (
    Item,
    Knapsack,
    Solution,
    brute_force_solution,
    density_order,
    solution_from_indices,
)

__all__ = [
    "FractionalSolution",
    "Item",
    "Knapsack",
    "Solution",
    "branch_and_bound",
    "brute_force_solution",
    "capacity_dp_solution",
    "density_order",
    "dp_by_capacity",
    "dp_by_profit",
    "fptas",
    "fractional_bound",
    "greedy_density",
    "half_approximation",
    "max_profit_values",
    "meet_in_the_middle",
    "min_weight_values",
    "profit_dp_solution",
    "solution_from_indices",
    "unbounded_dp",
]
