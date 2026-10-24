"""Reference implementations of classical combinatorial optimisation algorithms.

The package is organised by problem family:

``combinatorial.tsp``
    Travelling salesman problem: exact dynamic programming and branch and
    bound, the classical approximation algorithms with their worst-case
    guarantees, local search, and the Held-Karp lower bound.

``combinatorial.knapsack``
    Knapsack problems: dynamic programming, meet in the middle, branch and
    bound, the fractional (LP) bound, and fully polynomial approximation.

``combinatorial.binpacking``
    One-dimensional bin packing: first/best fit, their decreasing variants,
    the classical lower bound, and an exact branch and bound.

Everything is pure Python with no third-party dependencies, so the algorithms
can be read as they are stated in the literature.  See ``README.md`` for the
bibliography each routine is drawn from.
"""

from . import binpacking, graphs, instances, knapsack, matching, tsp

__all__ = ["binpacking", "graphs", "instances", "knapsack", "matching", "tsp"]
