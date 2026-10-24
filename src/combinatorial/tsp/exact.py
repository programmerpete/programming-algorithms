"""Exact algorithms for the travelling salesman problem.

Two solvers, both exhaustive:

* :func:`brute_force_tour` enumerates all ``(n-1)!`` tours; it exists as the
  oracle for tests and as the baseline the dynamic program improves on.
* :func:`held_karp` is the Held-Karp dynamic program: it exploits the fact that
  an optimal tour through a subset that ends at a given city is an optimal tour
  through that subset, which reduces ``(n-1)!`` to ``n^2 2^n`` elementary steps.

References
----------
Held, M. & Karp, R. M. (1962). A dynamic programming approach to sequencing
    problems. *SIAM Journal on Applied Mathematics* 10(1), 196-210.
Bellman, R. (1962). Dynamic programming treatment of the travelling salesman
    problem. *Journal of the ACM* 9(1), 61-63.
Applegate, D. L., Bixby, R. E., Chvatal, V. & Cook, W. J. (2006). *The
    Traveling Salesman Problem: A Computational Study*. Princeton University
    Press.  Chapter 4 develops the dynamic program and its complexity.
"""

from __future__ import annotations

import itertools
from array import array
from typing import Sequence

from .tours import tour_length

# 10 cities is 9! = 362 880 tours; 11 would be 3.6 million, which is where
# brute force stops being a useful oracle.
_BRUTE_FORCE_LIMIT = 10


def brute_force_tour(distance: Sequence[Sequence[float]]) -> list[int]:
    """Optimal tour by enumerating every permutation, fixing city 0 as the start."""
    n = len(distance)
    if n > _BRUTE_FORCE_LIMIT:
        raise ValueError(
            f"brute force over {n} cities is {(n - 1)}! tours; "
            f"the limit is {_BRUTE_FORCE_LIMIT}"
        )
    if n <= 2:
        return list(range(n))

    best_length = float("inf")
    best_tour: list[int] = []
    for tail in itertools.permutations(range(1, n)):
        tour = (0,) + tail
        length = tour_length(distance, tour)
        if length < best_length:
            best_length, best_tour = length, list(tour)
    return best_tour


def held_karp(distance: Sequence[Sequence[float]]) -> list[int]:
    """Optimal tour by the Held-Karp dynamic program, in O(n^2 2^n) time.

    Let ``C(S, j)`` be the length of the shortest path that starts at city 0,
    visits exactly the cities of ``S`` (with ``j`` in ``S``) and ends at ``j``.
    Then::

        C(S, j) = min over i in S \\ {j} of C(S \\ {j}, i) + d(i, j)
        OPT     = min over j of C({1..n-1}, j) + d(j, 0)

    The table holds one value per (subset, endpoint) pair, so the memory is
    ``n 2^(n-1)`` doubles and the running time ``O(n^2 2^n)``.  Working on the
    symmetric matrix is what makes this the best known *general* exact algorithm
    for the TSP; it is still exponential, and it is the reason practical
    solvers branch on edges with linear-programming bounds instead.
    """
    n = len(distance)
    if n <= 2:
        return list(range(n))

    m = n - 1  # cities 1..n-1, with bit k standing for city k+1
    size = 1 << m
    infinity = float("inf")

    # table[mask][k]: cost of the best path 0 -> ... -> city k+1 over the cities of mask
    table = [array("d", [infinity]) * m for _ in range(size)]
    parent = [array("i", [-1]) * m for _ in range(size)]
    for k in range(m):
        table[1 << k][k] = distance[0][k + 1]

    for mask in range(1, size):
        row = table[mask]
        for k in range(m):
            if not mask >> k & 1:
                continue
            current = row[k]
            if current == infinity:
                continue
            for nxt in range(m):
                if mask >> nxt & 1:
                    continue
                candidate = current + distance[k + 1][nxt + 1]
                next_mask = mask | 1 << nxt
                if candidate < table[next_mask][nxt]:
                    table[next_mask][nxt] = candidate
                    parent[next_mask][nxt] = k

    full = size - 1
    best_length = infinity
    last = -1
    for k in range(m):
        candidate = table[full][k] + distance[k + 1][0]
        if candidate < best_length:
            best_length, last = candidate, k

    order: list[int] = []
    mask, k = full, last
    while k != -1:
        order.append(k + 1)
        previous = parent[mask][k]
        mask ^= 1 << k
        k = previous
    order.reverse()
    return [0] + order
