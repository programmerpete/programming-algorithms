"""Graph primitives shared by the combinatorial solvers.

Small, dependency-free building blocks: a minimum spanning tree, a disjoint-set
forest, and an Eulerian circuit.  All of them are stated in the form used by the
algorithms that build on them, and all are O(n^2) or better, which is the right
trade-off for the dense complete graphs that the TSP solvers work with.

References
----------
Prim, R. C. (1957). Shortest connection networks and some generalizations.
    Bell System Technical Journal 36(6), 1389-1401.
    (Independently Jarnik 1930 and Dijkstra 1959.)
Tarjan, R. E. (1975). Efficiency of a good but not linear set union algorithm.
    Journal of the ACM 22(2), 215-225.
Hierholzer, C. (1873). Ueber die Moeglichkeit, einen Linienzug ohne
    Wiederholung und ohne Unterbrechung zu umfahren.
    Mathematische Annalen 6(1), 30-32.
"""

from __future__ import annotations

from typing import Callable, Sequence

Cost = Callable[[int, int], float]


def prim_mst(
    n: int, cost: Cost, vertices: Sequence[int] | None = None
) -> list[tuple[int, int, float]]:
    """Minimum spanning tree of the subgraph induced by ``vertices``.

    Parameters
    ----------
    n:
        Number of vertices; vertex labels are ``0 .. n-1``.
    cost:
        Symmetric cost function ``cost(u, v) -> float``.  Passing a closure is
        what lets the Held-Karp bound run Prim on *penalised* edge costs.
    vertices:
        Vertices to span.  Defaults to all of them.

    Returns
    -------
    list of ``(u, v, weight)`` edges, ``len(vertices) - 1`` of them.

    The implementation is the textbook O(k^2) array form of Prim's algorithm,
    with k = len(vertices).  Ties are broken by vertex index, which keeps the
    result deterministic and therefore reproducible in tests.
    """
    verts = list(range(n)) if vertices is None else list(vertices)
    k = len(verts)
    if k <= 1:
        return []

    infinity = float("inf")
    best = [infinity] * k  # cheapest known edge from the tree to each vertex
    parent = [-1] * k
    in_tree = [False] * k
    best[0] = 0.0
    edges: list[tuple[int, int, float]] = []

    for _ in range(k):
        u = -1
        u_best = infinity
        for i in range(k):
            if not in_tree[i] and best[i] < u_best:
                u, u_best = i, best[i]
        in_tree[u] = True
        if parent[u] != -1:
            edges.append((verts[parent[u]], verts[u], u_best))
        for i in range(k):
            if not in_tree[i]:
                c = cost(verts[u], verts[i])
                if c < best[i]:
                    best[i] = c
                    parent[i] = u
    return edges


def adjacency_lists(n: int, edges: Sequence[tuple[int, int, float]]) -> list[list[int]]:
    """Undirected adjacency lists for an edge list."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v, _ in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def dfs_preorder(adjacency: Sequence[Sequence[int]], start: int = 0) -> list[int]:
    """Depth-first preorder traversal of a connected graph."""
    seen = [False] * len(adjacency)
    order: list[int] = []
    stack = [start]
    while stack:
        u = stack.pop()
        if seen[u]:
            continue
        seen[u] = True
        order.append(u)
        for v in reversed(adjacency[u]):
            if not seen[v]:
                stack.append(v)
    return order


def eulerian_circuit(
    n: int, edges: Sequence[tuple[int, int]], start: int = 0
) -> list[int]:
    """Eulerian circuit of a connected multigraph whose degrees are all even.

    Hierholzer's algorithm, iterative form: walk until stuck, splice the
    resulting cycles together.  Returns the sequence of vertices visited, of
    length ``len(edges) + 1``.
    """
    incident: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for index, (u, v) in enumerate(edges):
        incident[u].append((v, index))
        incident[v].append((u, index))

    used = [False] * len(edges)
    circuit: list[int] = []
    stack = [start]
    while stack:
        u = stack[-1]
        while incident[u] and used[incident[u][-1][1]]:
            incident[u].pop()
        if incident[u]:
            v, index = incident[u].pop()
            used[index] = True
            stack.append(v)
        else:
            circuit.append(stack.pop())
    return circuit


class DisjointSet:
    """Union-find with path compression and union by rank."""

    __slots__ = ("parent", "rank")

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        """Merge the sets of ``a`` and ``b``; False if they were already equal."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True
