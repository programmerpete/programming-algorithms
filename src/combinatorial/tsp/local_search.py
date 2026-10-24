"""Local search for the travelling salesman problem.

Both routines improve an existing tour to a local optimum, so they compose with
the constructive heuristics: ``two_opt(nearest_neighbour(d))`` is the classical
first improvement step, and the same 2-opt move sits inside Lin-Kernighan and
the chained local search that solves the largest instances in practice.

These moves reverse or relocate parts of a tour, so they need symmetric
distances to evaluate a move in O(1); that is stated again in each docstring.

References
----------
Croes, G. A. (1958). A method for solving traveling-salesman problems.
    *Operations Research* 6(6), 791-812.  (The 2-opt move.)
Flood, M. M. (1956). The traveling-salesman problem. *Operations Research*
    4(1), 61-75.  (Independent introduction of the same exchange.)
Or, I. (1976). *Traveling Salesman-Type Combinatorial Problems and their
    Relation to the Logistics of Regional Blood Banking*. PhD thesis,
    Northwestern University.  (Or-opt segment moves.)
Lin, S. & Kernighan, B. W. (1973). An effective heuristic algorithm for the
    traveling-salesman problem. *Operations Research* 21(2), 498-516.
"""

from __future__ import annotations

from typing import Sequence

_EPSILON = 1e-9


def two_opt(distance: Sequence[Sequence[float]], tour: Sequence[int]) -> list[int]:
    """Improve ``tour`` by undoing crossing edges until no 2-opt move helps.

    For each pair of edges ``(a, b)`` and ``(c, d)`` the move replaces them with
    ``(a, c)`` and ``(b, d)``, which is exactly reversing the segment between
    them::

        delta = d(a, c) + d(b, d) - d(a, b) - d(c, d)

    First-improvement, repeated until a local optimum is reached; O(n^2) per
    pass.  Requires symmetric distances.
    """
    tour = list(tour)
    n = len(tour)
    if n < 4:
        return tour

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            a, b = tour[i], tour[i + 1]
            for j in range(i + 1, n):
                c = tour[j]
                d = tour[(j + 1) % n]
                if a == c or b == d:
                    continue
                delta = distance[a][c] + distance[b][d] - distance[a][b] - distance[c][d]
                if delta < -_EPSILON:
                    tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                    improved = True
                    break
            if improved:
                break
    return tour


def or_opt(distance: Sequence[Sequence[float]], tour: Sequence[int], max_segment: int = 3) -> list[int]:
    """Improve ``tour`` by relocating segments of up to ``max_segment`` cities.

    The 2-opt move reverses a segment; Or-opt moves one elsewhere, which is what
    repairs tours whose cities are in the right order but at the wrong place.
    First-improvement, repeated until no move helps.  Requires symmetric
    distances.
    """
    tour = list(tour)
    n = len(tour)
    if n < 4:
        return tour

    improved = True
    while improved:
        improved = False
        for length in range(1, min(max_segment, n - 2) + 1):
            for i in range(n):
                segment = [tour[(i + k) % n] for k in range(length)]
                before = tour[(i - 1) % n]
                after = tour[(i + length) % n]
                if before in segment or after in segment:
                    continue
                moved = set(segment)
                removal = (
                    distance[before][after]
                    - distance[before][segment[0]]
                    - distance[segment[-1]][after]
                )
                rest = [city for city in tour if city not in moved]
                best_position, best_delta = None, -_EPSILON
                for position in range(len(rest)):
                    a = rest[position]
                    b = rest[(position + 1) % len(rest)]
                    delta = (
                        removal
                        + distance[a][segment[0]]
                        + distance[segment[-1]][b]
                        - distance[a][b]
                    )
                    if delta < best_delta:
                        best_position, best_delta = position, delta
                if best_position is not None:
                    rest[best_position + 1 : best_position + 1] = segment
                    tour = rest
                    improved = True
                    break
            if improved:
                break
    return tour
