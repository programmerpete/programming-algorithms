"""Constructive heuristics for the travelling salesman problem.

These build a tour directly from the distance matrix, in polynomial time, with
worst-case guarantees that are stated with each function.  They are the
classical reference points of the field: every one of them is superseded in
practice by local search, but they are what local search starts from and what
the approximation theory is about.

References
----------
Christofides, N. (1976). Worst-case analysis of a new heuristic for the
    travelling salesman problem. Technical Report 388, Graduate School of
    Industrial Administration, Carnegie Mellon University.  (Reprinted in
    *Operations Research Forum* 3, 20, 2022.)
Rosenkrantz, D. J., Stearns, R. E. & Lewis, P. M. (1977). An analysis of
    several heuristics for the traveling salesman problem. *SIAM Journal on
    Computing* 6(3), 563-581.  (Double tree, nearest insertion and their
    2-approximation proofs; the log n lower bound for nearest neighbour.)
Ong, H. L. & Moore, J. B. (1984). Worst-case analysis of two travelling
    salesman heuristics. *Operations Research Letters* 2(6), 273-277.
    (The log n bound for the greedy edge-matching heuristic.)
Johnson, D. S. & McGeoch, L. A. (1997). The traveling salesman problem: a case
    study in local optimization. In *Local Search in Combinatorial
    Optimization*, Wiley, 215-310.  (Empirical comparison of these heuristics.)
"""

from __future__ import annotations

from typing import Sequence

from ..graphs import DisjointSet, adjacency_lists, dfs_preorder, eulerian_circuit, prim_mst
from ..matching import min_weight_perfect_matching


def nearest_neighbour(distance: Sequence[Sequence[float]], start: int = 0) -> list[int]:
    """Walk to the nearest unvisited city until none are left.

    No constant factor is possible: Rosenkrantz, Stearns and Lewis (1977)
    construct instances where the tour is ``Theta(log n)`` times the optimum,
    and that is tight up to the constant.  Runs in O(n^2).
    """
    n = len(distance)
    unvisited = set(range(n))
    unvisited.discard(start)
    tour = [start]
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda city: distance[current][city])
        unvisited.discard(nxt)
        tour.append(nxt)
        current = nxt
    return tour


def nearest_insertion(distance: Sequence[Sequence[float]], start: int = 0) -> list[int]:
    """Grow a subtour by repeatedly inserting the city nearest to it.

    A 2-approximation for metric instances (Rosenkrantz, Stearns & Lewis 1977),
    and the bound is tight.  Runs in O(n^3) in this straightforward form.
    """
    n = len(distance)
    if n <= 2:
        return list(range(n))

    unvisited = set(range(n))
    unvisited.discard(start)
    first = min(unvisited, key=lambda city: distance[start][city])
    tour = [start, first]
    unvisited.discard(first)

    while unvisited:
        nearest = min(unvisited, key=lambda city: min(distance[city][u] for u in tour))
        best_position, best_delta = 0, float("inf")
        for position in range(len(tour)):
            a, b = tour[position], tour[(position + 1) % len(tour)]
            delta = distance[a][nearest] + distance[nearest][b] - distance[a][b]
            if delta < best_delta:
                best_position, best_delta = position, delta
        tour.insert(best_position + 1, nearest)
        unvisited.discard(nearest)
    return tour


def greedy_edge(distance: Sequence[Sequence[float]]) -> list[int]:
    """Cheapest-link construction: take the cheapest edge that keeps the degree
    at most two and does not close a cycle prematurely, until a Hamiltonian path
    remains.

    ``O(n^2 log n)``; the tour is at most ``0.5 (ceil(log2 n) + 1)`` times the
    optimum on metric instances (Ong & Moore 1984).
    """
    n = len(distance)
    if n <= 2:
        return list(range(n))

    edges = sorted(
        (distance[i][j], i, j) for i in range(n) for j in range(i + 1, n)
    )
    forest = DisjointSet(n)
    degree = [0] * n
    adjacency: list[list[int]] = [[] for _ in range(n)]
    taken = 0

    for _, i, j in edges:
        if degree[i] >= 2 or degree[j] >= 2:
            continue
        if forest.find(i) == forest.find(j):
            continue  # would close a cycle before the path is complete
        forest.union(i, j)
        degree[i] += 1
        degree[j] += 1
        adjacency[i].append(j)
        adjacency[j].append(i)
        taken += 1
        if taken == n - 1:
            break

    start = next(city for city in range(n) if degree[city] == 1)
    tour = [start]
    previous, current = -1, start
    while True:
        following = next((city for city in adjacency[current] if city != previous), None)
        if following is None:
            break
        tour.append(following)
        previous, current = current, following
    return tour


def mst_double_tree(distance: Sequence[Sequence[float]]) -> list[int]:
    """Double-tree tour: traverse a minimum spanning tree in preorder.

    A 2-approximation for metric instances: the tree costs at most OPT, walking
    it twice costs ``2 * MST <= 2 OPT``, and shortcutting repeated cities only
    shortens the walk (triangle inequality).
    """
    n = len(distance)
    edges = prim_mst(n, lambda i, j: distance[i][j])
    return dfs_preorder(adjacency_lists(n, edges), 0)


def christofides(distance: Sequence[Sequence[float]], matching: str = "auto") -> list[int]:
    """Christofides' heuristic: spanning tree + minimum matching + shortcutting.

    1. Build a minimum spanning tree ``T``.
    2. Let ``O`` be the odd-degree vertices of ``T``; find a minimum-weight
       perfect matching ``M`` on ``O`` under the given distances.
    3. ``T + M`` is connected with all degrees even, hence Eulerian; shortcut
       the Eulerian circuit to a Hamiltonian cycle.

    For metric instances the tour is at most 3/2 times the optimum: the Eulerian
    circuit costs ``w(T) + w(M) <= OPT + OPT/2``, where ``w(M) <= OPT/2`` holds
    because an optimal tour, restricted to ``O``, is a pair of perfect matchings.
    The exact worst-case ratio is slightly below 3/2 (Cornuejols & Nemhauser
    1978).  ``matching="greedy"`` still yields a 2-approximation and is the
    practical choice beyond ~20 odd vertices.
    """
    n = len(distance)
    if n <= 3:
        return list(range(n))

    tree_edges = prim_mst(n, lambda i, j: distance[i][j])
    odd = [city for city in range(n) if len(adjacency_lists(n, tree_edges)[city]) % 2]
    pairs, _ = min_weight_perfect_matching(distance, odd, method=matching)

    multigraph = [(u, v) for u, v, _ in tree_edges] + list(pairs)
    circuit = eulerian_circuit(n, multigraph)

    seen = set()
    tour = []
    for city in circuit:
        if city not in seen:
            seen.add(city)
            tour.append(city)
    return tour
