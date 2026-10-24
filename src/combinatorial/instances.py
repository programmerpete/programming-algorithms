"""Instance generators used by the tests, the examples and the benchmarks.

Every generator takes a ``seed`` so that the tests are reproducible.  The knapsack
generators follow the data classes of Pisinger (2005), which is what the
literature uses to separate easy instances from hard ones: uncorrelated items are
solved by any method, while strongly correlated items (``p_j = w_j + 10``) defeat
the greedy rules and subset-sum-like items (``p_j = w_j``) defeat dynamic
programming.

References
----------
Pisinger, D. (2005). Where are the hard knapsack problems? *Computers &
    Operations Research* 32(9), 2271-2284.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from .knapsack.models import Knapsack

__all__ = [
    "correlated_knapsack",
    "euclidean_matrix",
    "random_asymmetric_tsp",
    "random_bin_packing",
    "random_euclidean_tsp",
    "random_knapsack",
    "random_points",
    "random_symmetric_tsp",
    "subset_sum_knapsack",
]


def euclidean_matrix(points: Sequence[tuple[float, float]]) -> list[list[float]]:
    """Distance matrix of points in the plane.

    Euclidean distances satisfy the triangle inequality exactly, which is the
    assumption behind the 3/2 and 2-approximation guarantees.
    """
    return [
        [math.dist(p, q) for q in points]
        for p in points
    ]


def random_points(
    n: int, seed: int | None = None, scale: float = 1000.0
) -> list[tuple[float, float]]:
    """``n`` points uniform in the square ``[0, scale]^2``."""
    rng = random.Random(seed)
    return [(rng.uniform(0, scale), rng.uniform(0, scale)) for _ in range(n)]


def random_euclidean_tsp(
    n: int, seed: int | None = None, scale: float = 1000.0
) -> list[list[float]]:
    """Metric TSP instance: distances between random points in the plane."""
    return euclidean_matrix(random_points(n, seed=seed, scale=scale))


def random_symmetric_tsp(
    n: int, seed: int | None = None, low: int = 1, high: int = 100
) -> list[list[float]]:
    """Symmetric instance with random integer distances.

    Not metric in general: this is the harder setting where the approximation
    guarantees do not apply, and it is what the exact solvers are tested on.
    """
    rng = random.Random(seed)
    distance = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            weight = float(rng.randint(low, high))
            distance[i][j] = distance[j][i] = weight
    return distance


def random_asymmetric_tsp(
    n: int, seed: int | None = None, low: int = 1, high: int = 100
) -> list[list[float]]:
    """Instance with independent directed distances (no symmetry, no metric)."""
    rng = random.Random(seed)
    distance = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                distance[i][j] = float(rng.randint(low, high))
    return distance


def random_knapsack(
    n: int,
    seed: int | None = None,
    capacity: int | None = None,
    weight_range: tuple[int, int] = (1, 100),
    profit_range: tuple[int, int] = (1, 100),
) -> Knapsack:
    """Uncorrelated instance: weights and profits drawn independently.

    The default capacity is half the total weight, which makes the instance
    non-trivial without being pathological.
    """
    rng = random.Random(seed)
    items = [
        (rng.randint(*weight_range), rng.randint(*profit_range)) for _ in range(n)
    ]
    if capacity is None:
        capacity = max(1, sum(weight for weight, _ in items) // 2)
    return Knapsack.from_pairs(capacity, items)


def correlated_knapsack(
    n: int, seed: int | None = None, capacity: int | None = None, spread: int = 10
) -> Knapsack:
    """Strongly correlated instance: ``p_j = w_j + spread``.

    Greedy rules do badly here, since every item has nearly the same density, and
    the fractional bound is close to the integer optimum, so branch and bound has
    little to prune with.
    """
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        weight = rng.randint(1, 100)
        items.append((weight, weight + spread))
    if capacity is None:
        capacity = max(1, sum(weight for weight, _ in items) // 2)
    return Knapsack.from_pairs(capacity, items)


def subset_sum_knapsack(
    n: int, seed: int | None = None, capacity: int | None = None
) -> Knapsack:
    """Subset-sum instance: ``p_j = w_j``, so profit is weight.

    The capacity-indexed dynamic program has to fill its whole table here, which
    makes this the standard stress test for pseudo-polynomial algorithms.
    """
    rng = random.Random(seed)
    items = [(rng.randint(1, 100), 0) for _ in range(n)]
    items = [(weight, weight) for weight, _ in items]
    if capacity is None:
        capacity = max(1, sum(weight for weight, _ in items) // 2)
    return Knapsack.from_pairs(capacity, items)


def random_bin_packing(
    n: int, seed: int | None = None, capacity: int = 100, size_range: tuple[int, int] = (10, 70)
) -> tuple[list[int], int]:
    """Item sizes for one-dimensional bin packing, with the bin capacity."""
    rng = random.Random(seed)
    sizes = [rng.randint(*size_range) for _ in range(n)]
    return sizes, capacity
