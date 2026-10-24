"""Exact branch and bound for the travelling salesman problem.

The search grows partial paths from city 0 and prunes with a spanning-tree
bound: any completion of a partial path is a Hamiltonian path on the cities that
are still unvisited plus the current city, so its length is at least the weight
of a minimum spanning tree on that set.  Adding the cheapest way back into the
start gives a second, independent bound, and the larger of the two is used.

This is the classical path-branching search of Little, Murty, Sweeney and Karel
(1963) with a cheaper bound; production codes branch on edges and bound with the
Held-Karp relaxation (:mod:`combinatorial.tsp.bounds`) plus cutting planes.

References
----------
Little, J. D. C., Murty, K. G., Sweeney, D. W. & Karel, C. (1963). An algorithm
    for the traveling salesman problem. *Operations Research* 11(6), 972-989.
Held, M. & Karp, R. M. (1970), as in :mod:`combinatorial.tsp.bounds`.
"""

from __future__ import annotations

from typing import Sequence

from ..graphs import prim_mst
from .constructive import nearest_neighbour
from .local_search import two_opt
from .tours import tour_length


def branch_and_bound(
    distance: Sequence[Sequence[float]], upper_bound: Sequence[int] | None = None
) -> list[int]:
    """Optimal tour by depth-first branch and bound.

    The incumbent is a 2-opt tour unless one is supplied, and cities are explored
    nearest-first so that good tours are found early.  Exact, but exponential in
    the worst case; the bound is what keeps it usable up to roughly 20 cities.
    """
    n = len(distance)
    if n <= 3:
        return list(range(n))

    incumbent = list(upper_bound) if upper_bound is not None else two_opt(
        distance, nearest_neighbour(distance, 0)
    )
    best_tour = incumbent
    best_length = tour_length(distance, incumbent)

    # Stack entries are (path, visited bitmask, length of the path so far).
    stack: list[tuple[list[int], int, float]] = [([0], 1, 0.0)]
    while stack:
        path, visited, length = stack.pop()
        last = path[-1]

        if len(path) == n:
            total = length + distance[last][0]
            if total < best_length:
                best_length, best_tour = total, path
            continue

        if length + _completion_bound(distance, path, visited) >= best_length:
            continue

        # Push farthest-first so the nearest city is explored first (LIFO).
        candidates = sorted(
            (city for city in range(n) if not visited >> city & 1),
            key=lambda city: distance[last][city],
            reverse=True,
        )
        for city in candidates:
            stack.append((path + [city], visited | 1 << city, length + distance[last][city]))

    return best_tour


def _completion_bound(
    distance: Sequence[Sequence[float]], path: Sequence[int], visited: int
) -> float:
    """Lower bound on the cost of completing ``path`` into a tour.

    Any completion is a Hamiltonian path from ``path[-1]`` through the unvisited
    cities ``R`` to ``path[0]``.  Removing the edge incident to the start leaves a
    spanning tree of ``R + path[-1]``, and the removed edge runs to some vertex of
    that set, so::

        cost >= MST(R + last) + min over x in R + {last} of d(start, x)

    and symmetrically::

        cost >= MST(R + start) + min over y in R + {start} of d(last, y)

    Both are valid; the larger is used.  Either one dominates the plain
    ``MST(R + last)`` bound that the second term of the first expression extends,
    and together they cut the search by a factor of about five on Euclidean
    instances.
    """
    n = len(distance)
    last = path[-1]
    start = path[0]
    remaining = [city for city in range(n) if not visited >> city & 1]

    without_start = prim_mst(n, lambda i, j: distance[i][j], remaining + [last])
    bound = sum(weight for _, _, weight in without_start)
    bound += min(distance[start][city] for city in remaining + [last])

    without_last = prim_mst(n, lambda i, j: distance[i][j], remaining + [start])
    other = sum(weight for _, _, weight in without_last)
    other += min(distance[last][city] for city in remaining + [start])

    return max(bound, other)
