"""Travelling salesman problem solvers.

The problem: given a complete graph with edge costs ``d(i, j)``, find a
Hamiltonian cycle of minimum total cost.  Deciding whether a tour shorter than a
given bound exists is NP-complete (Karp 1972), so the package offers both exact
solvers for small instances and the classical approximation and local search
algorithms for large ones.

===========================  ==========================  =========================
Solver                       Guarantee (metric TSP)      Cost
===========================  ==========================  =========================
``held_karp``                exact                       O(n^2 2^n) time
``branch_and_bound``         exact                       exponential worst case
``christofides``             <= 3/2 OPT                  O(n^2 log n)
``mst_double_tree``          <= 2 OPT                    O(n^2)
``nearest_insertion``        <= 2 OPT                    O(n^3)
``greedy_edge``              <= 0.5 (ceil(log2 n) + 1)   O(n^2 log n)
``nearest_neighbour``        Theta(log n) OPT (tight)    O(n^2)
``two_opt`` / ``or_opt``     none (local optimum)        O(n^2) per pass
``held_karp_bound``          lower bound on OPT          O(n^2) per iteration
===========================  ==========================  =========================

Typical use::

    from combinatorial import instances, tsp

    distance = instances.random_euclidean_tsp(50, seed=1)
    tour = tsp.two_opt(distance, tsp.christofides(distance))
    print(tsp.tour_length(distance, tour))
"""

from .bounds import held_karp_bound, one_tree
from .branch_bound import branch_and_bound
from .constructive import (
    christofides,
    greedy_edge,
    mst_double_tree,
    nearest_insertion,
    nearest_neighbour,
)
from .exact import brute_force_tour, held_karp
from .local_search import or_opt, two_opt
from .tours import tour_length, validate_tour

__all__ = [
    "branch_and_bound",
    "brute_force_tour",
    "christofides",
    "greedy_edge",
    "held_karp",
    "held_karp_bound",
    "mst_double_tree",
    "nearest_insertion",
    "nearest_neighbour",
    "one_tree",
    "or_opt",
    "tour_length",
    "two_opt",
    "validate_tour",
]
