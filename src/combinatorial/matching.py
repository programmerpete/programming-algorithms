"""Minimum-weight perfect matching on a complete graph.

Christofides' algorithm needs a minimum-weight perfect matching on the odd
degree vertices of a spanning tree, which is a *general* (non-bipartite) graph,
so the assignment problem / Hungarian algorithm does not apply; the classical
polynomial algorithm is Edmonds' blossom algorithm.

This module implements the matching exactly by dynamic programming over subsets,
which is exponential in the number of vertices to match but has no large
constant and is completely transparent.  It is exact for the sizes that
Christofides' construction is practical for (up to ~20 odd vertices), and falls
back to the classical greedy matching for larger inputs, at the price of the
3/2 guarantee (a 2-approximate matching yields a 2-approximate tour).

References
----------
Edmonds, J. (1965). Paths, trees, and flowers. Canadian Journal of
    Mathematics 17, 449-467.
"""

from __future__ import annotations

from array import array
from typing import Sequence

# Above these sizes the subset dynamic program becomes the bottleneck rather
# than the surrounding algorithm; see the module docstring.
_EXACT_AUTO_LIMIT = 20
_EXACT_HARD_LIMIT = 24


def min_weight_perfect_matching(
    cost: Sequence[Sequence[float]],
    vertices: Sequence[int] | None = None,
    method: str = "auto",
) -> tuple[list[tuple[int, int]], float]:
    """Minimum-weight perfect matching of ``vertices`` under ``cost``.

    Parameters
    ----------
    cost:
        Symmetric cost matrix (``cost[u][v]``).
    vertices:
        Vertices to match; must have even cardinality.  Defaults to all.
    method:
        ``"exact"`` (subset dynamic program), ``"greedy"`` (sort edges, take
        disjoint pairs) or ``"auto"`` (exact up to ``_EXACT_AUTO_LIMIT``
        vertices, greedy beyond).

    Returns
    -------
    ``(pairs, total_cost)`` with each pair an ordered ``(u, v)`` tuple and the
    pairs sorted by their first vertex, so the output is deterministic.
    """
    verts = list(range(len(cost))) if vertices is None else list(vertices)
    if len(verts) % 2 != 0:
        raise ValueError("a perfect matching needs an even number of vertices")
    if not verts:
        return [], 0.0

    if method == "auto":
        method = "exact" if len(verts) <= _EXACT_AUTO_LIMIT else "greedy"
    if method == "exact":
        if len(verts) > _EXACT_HARD_LIMIT:
            raise ValueError(
                f"exact matching of {len(verts)} vertices needs "
                f"2^{len(verts)} table entries; use method='greedy' instead"
            )
        pairs = _exact_matching(cost, verts)
    elif method == "greedy":
        pairs = _greedy_matching(cost, verts)
    else:
        raise ValueError(f"unknown matching method: {method!r}")

    total = sum(cost[u][v] for u, v in pairs)
    return pairs, total


def _exact_matching(
    cost: Sequence[Sequence[float]], verts: Sequence[int]
) -> list[tuple[int, int]]:
    """Minimum-weight perfect matching by dynamic programming over subsets.

    ``table[mask]`` is the cost of optimally matching the vertices of ``mask``.
    Every even mask is settled by pairing its lowest-indexed vertex with one of
    the others, which enumerates each matching exactly once::

        table[mask] = min over j in mask, j != i of cost[i][j] + table[mask \\ {i, j}]

    where i is the lowest-indexed vertex of ``mask``.
    """
    k = len(verts)
    size = 1 << k
    infinity = float("inf")
    table = array("d", [infinity]) * size
    partner = array("i", [-1]) * size
    table[0] = 0.0

    for mask in range(2, size):
        if bin(mask).count("1") % 2:
            continue
        lowest = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << lowest)
        best = infinity
        best_j = -1
        remaining = rest
        while remaining:
            j = (remaining & -remaining).bit_length() - 1
            remaining ^= 1 << j
            candidate = table[rest ^ (1 << j)] + cost[verts[lowest]][verts[j]]
            if candidate < best:
                best, best_j = candidate, j
        table[mask] = best
        partner[mask] = best_j

    pairs: list[tuple[int, int]] = []
    mask = size - 1
    while mask:
        lowest = (mask & -mask).bit_length() - 1
        j = partner[mask]
        pairs.append((verts[lowest], verts[j]))
        mask ^= (1 << lowest) | (1 << j)
    pairs.sort()
    return pairs


def _greedy_matching(
    cost: Sequence[Sequence[float]], verts: Sequence[int]
) -> list[tuple[int, int]]:
    """Greedy matching: repeatedly take the cheapest edge between free vertices."""
    free = set(verts)
    candidates = sorted(
        (cost[u][v], u, v)
        for index, u in enumerate(verts)
        for v in verts[index + 1 :]
    )
    pairs: list[tuple[int, int]] = []
    for _, u, v in candidates:
        if u in free and v in free:
            free.discard(u)
            free.discard(v)
            pairs.append((u, v))
    pairs.sort()
    return pairs
