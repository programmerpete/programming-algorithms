"""Data model for knapsack problems.

Instances and solutions are immutable value objects so that solvers can be
composed and compared without aliasing surprises::

    >>> instance = Knapsack.from_pairs(capacity=10, pairs=[(5, 4), (6, 7)])
    >>> solution = dp_by_capacity(instance)
    >>> solution.value, solution.weight, solution.items
    (7, 6, (1,))

References
----------
Kellerer, H., Pferschy, U. & Pisinger, D. (2004). *Knapsack Problems*.
    Springer.  Chapter 1 states the problem family and its notation; we use
    ``p_j`` for profits and ``w_j`` for weights, and ``c`` for the capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Item:
    """A single item with an integer weight and an integer profit."""

    weight: int
    profit: int

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("item weights must be non-negative")
        if self.profit < 0:
            raise ValueError("item profits must be non-negative")


@dataclass(frozen=True)
class Knapsack:
    """A 0/1 knapsack instance: maximise total profit subject to total weight."""

    capacity: int
    items: tuple[Item, ...]

    def __post_init__(self) -> None:
        if self.capacity < 0:
            raise ValueError("capacity must be non-negative")

    @classmethod
    def from_pairs(cls, capacity: int, pairs: Iterable[tuple[int, int]]) -> "Knapsack":
        """Build an instance from ``(weight, profit)`` pairs."""
        return cls(capacity=capacity, items=tuple(Item(w, p) for w, p in pairs))

    @property
    def n(self) -> int:
        return len(self.items)

    @property
    def weights(self) -> list[int]:
        return [item.weight for item in self.items]

    @property
    def profits(self) -> list[int]:
        return [item.profit for item in self.items]

    @property
    def total_weight(self) -> int:
        return sum(item.weight for item in self.items)

    @property
    def total_profit(self) -> int:
        return sum(item.profit for item in self.items)

    def fitting(self) -> tuple[int, ...]:
        """Indices of the items that fit on their own, i.e. ``w_j <= c``.

        Items outside this set belong to no feasible solution and every exact
        algorithm can discard them; the density-greedy bound argument also
        needs them gone, since the split item must fit by itself.
        """
        return tuple(i for i, item in enumerate(self.items) if item.weight <= self.capacity)

    def free(self) -> tuple[int, ...]:
        """Indices of zero-weight items with positive profit, which are always taken."""
        return tuple(
            i for i, item in enumerate(self.items) if item.weight == 0 and item.profit > 0
        )

    def usable(self) -> tuple[int, ...]:
        """Indices of the items that fit and weigh something.

        This is the set the density-ordered algorithms work on: items heavier
        than the capacity belong to no feasible solution, and zero-weight items
        are either always taken (:meth:`free`) or worthless, while a zero weight
        would make the profit/weight ratio in the branch-and-bound bound
        undefined.
        """
        return tuple(
            i for i, item in enumerate(self.items) if 0 < item.weight <= self.capacity
        )


def density_order(knapsack: Knapsack, indices: Iterable[int] | None = None) -> list[int]:
    """Item indices sorted by non-increasing profit/weight ratio.

    Zero-weight items sort first, since their ratio is infinite.  Ties are broken
    by index so that every solver explores the same order and results are
    reproducible.
    """
    idx = list(range(knapsack.n)) if indices is None else list(indices)

    def key(i: int) -> tuple[float, int]:
        item = knapsack.items[i]
        density = float("inf") if item.weight == 0 else item.profit / item.weight
        return (-density, i)

    return sorted(idx, key=key)


@dataclass(frozen=True)
class Solution:
    """A feasible selection of items together with its value and weight.

    ``items`` holds the chosen indices in ascending order.  An index appears as
    many times as the item is used, which matters only for the unbounded problem;
    the 0/1 solvers never repeat an index.
    """

    value: int
    weight: int
    items: tuple[int, ...]

    def __bool__(self) -> bool:
        return bool(self.items)


def solution_from_indices(knapsack: Knapsack, indices: Iterable[int]) -> Solution:
    """Assemble a :class:`Solution` from item indices, which may repeat."""
    chosen = sorted(indices)
    value = sum(knapsack.items[i].profit for i in chosen)
    weight = sum(knapsack.items[i].weight for i in chosen)
    if weight > knapsack.capacity:
        raise ValueError("selected items exceed the capacity")
    return Solution(value=value, weight=weight, items=tuple(chosen))


def brute_force_solution(knapsack: Knapsack) -> Solution:
    """Optimal solution by exhaustive enumeration; the oracle used in tests."""
    best: tuple[int, int, tuple[int, ...]] = (0, 0, ())
    for mask in range(1 << knapsack.n):
        weight = value = 0
        indices = []
        for i, item in enumerate(knapsack.items):
            if mask >> i & 1:
                weight += item.weight
                value += item.profit
                indices.append(i)
        if weight <= knapsack.capacity and value > best[0]:
            best = (value, weight, tuple(indices))
    return Solution(*best)
