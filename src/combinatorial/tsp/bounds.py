"""Lower bounds for the travelling salesman problem.

The bound implemented here is the one that made large instances tractable: the
minimum 1-tree relaxed by vertex penalties, maximised by subgradient ascent.  It
is both a measure of how good a heuristic tour is and the pruning rule used by
branch-and-bound codes; Concorde's cutting-plane algorithm starts from the same
relaxation.

References
----------
Held, M. & Karp, R. M. (1970). The traveling-salesman problem and minimum
    spanning trees. *Operations Research* 18(6), 1138-1162.
Held, M. & Karp, R. M. (1971). The traveling-salesman problem and minimum
    spanning trees: part II. *Mathematical Programming* 1(1), 6-25.
    (The subgradient ascent and its convergence.)
Volgenant, A. & Jonker, R. (1982). A branch and bound algorithm for the
    symmetric traveling salesman problem based on the 1-tree relaxation.
    *European Journal of Operational Research* 9(1), 83-89.
    (The step-size schedule used by practical implementations.)
Applegate, Bixby, Chvatal & Cook (2006), chapter 5: the relationship between
    the Held-Karp bound and the subtour linear program.
"""

from __future__ import annotations

from typing import Sequence

from ..graphs import prim_mst
from .constructive import nearest_neighbour
from .local_search import two_opt
from .tours import tour_length


def one_tree(
    distance: Sequence[Sequence[float]], pi: Sequence[float] | None = None
) -> tuple[float, list[int]]:
    """Cheapest 1-tree under vertex penalties ``pi``.

    A 1-tree is a spanning tree on cities ``1..n-1`` plus the two cheapest edges
    incident to city 0.  Every tour is a 1-tree (drop one edge of the cycle, and
    city 0 keeps its two incident edges), so the cheapest 1-tree costs at most
    OPT.

    Returns the tree's true weight and the degree of every city.  With penalties
    ``pi`` the tree is computed on the modified costs ``d(i,j) + pi_i + pi_j``,
    and the caller recovers the Lagrangian value as
    ``weight + sum_i pi_i (deg_i - 2)``.
    """
    n = len(distance)
    if n == 0:
        return 0.0, []
    if n == 1:
        return 0.0, [0]
    if n == 2:
        return distance[0][1], [1, 1]

    pi = [0.0] * n if pi is None else list(pi)
    edges = prim_mst(n, lambda i, j: distance[i][j] + pi[i] + pi[j], list(range(1, n)))

    weight = sum(distance[i][j] for i, j, _ in edges)
    degree = [0] * n
    for i, j, _ in edges:
        degree[i] += 1
        degree[j] += 1

    # The two edges at city 0 must be the cheapest ones under the *penalised*
    # costs, otherwise this is not the minimum 1-tree and its value is not a
    # lower bound on OPT.
    nearest_two = sorted(range(1, n), key=lambda city: distance[0][city] + pi[city])[:2]
    for city in nearest_two:
        weight += distance[0][city]
        degree[0] += 1
        degree[city] += 1
    return weight, degree


def held_karp_bound(
    distance: Sequence[Sequence[float]],
    iterations: int = 100,
    upper_bound: float | None = None,
    lambda0: float = 2.0,
) -> tuple[float, list[float]]:
    """Maximise the 1-tree bound by subgradient ascent; returns ``(bound, pi)``.

    For any penalties ``pi``::

        L(pi) = w(1-tree with costs d(i,j) + pi_i + pi_j) + sum_i pi_i (deg_i - 2)

    is a lower bound on OPT, since every tour is a 1-tree with ``deg_i = 2``.
    The subgradient is ``deg_i - 2``, and the step is Polyak's
    ``lambda * (UB - L) / ||g||^2`` with ``lambda`` halved every few non-improving
    iterations (Held & Karp 1971; Volgenant & Jonker 1982).

    ``upper_bound`` defaults to a 2-opt tour, which is only used for the step
    size, not for validity.  Two situations make the ascent stop early, and both
    prove optimality: a 1-tree with all degrees equal to 2 is itself a tour, and
    a bound that reaches the incumbent tour length means no tour is shorter.
    """
    n = len(distance)
    if n < 3:
        return tour_length(distance, list(range(n))), [0.0] * n

    if upper_bound is None:
        upper_bound = tour_length(distance, two_opt(distance, nearest_neighbour(distance)))

    pi = [0.0] * n
    best = float("-inf")
    best_pi = pi[:]
    step_scale = lambda0
    period = max(5, iterations // 10)
    stalled = 0

    for _ in range(iterations):
        weight, degree = one_tree(distance, pi)
        bound = weight + sum(pi[i] * (degree[i] - 2) for i in range(n))
        if bound > best + 1e-9:
            best, best_pi = bound, pi[:]
            stalled = 0
        else:
            stalled += 1

        if all(d == 2 for d in degree):
            break  # the 1-tree is a tour, so the bound is attained and optimal
        gradient = [d - 2 for d in degree]
        norm = sum(g * g for g in gradient)
        if norm == 0:
            break
        step = step_scale * (upper_bound - bound) / norm
        if step <= 0:
            break  # the bound reached the incumbent: that tour is optimal
        for i in range(n):
            pi[i] += step * gradient[i]

        if stalled >= period:
            step_scale *= 0.5
            stalled = 0

    return best, best_pi
