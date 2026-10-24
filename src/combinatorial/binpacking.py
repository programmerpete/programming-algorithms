"""One-dimensional bin packing.

Given item sizes and a bin capacity, pack every item using as few bins as
possible.  This is the counting version of the "how many boxes fit" question:
the knapsack solvers answer how much *value* fits, these answer how many
*containers* are needed.  Bin packing is NP-hard in the strong sense (Garey &
Johnson 1975), so unlike the knapsack problem it has no FPTAS, and the classical
results are constant-factor approximations and their additive terms.

The four heuristics are the classical ones, all ``O(n log n)`` with the sort:

* first fit: put each item in the first bin that can take it;
* best fit: put it in the fullest bin that can take it;
* the decreasing variants sort the items first, which is what buys the 11/9 bound.

References
----------
Johnson, D. S., Demers, A., Ullman, J. D., Garey, M. R. & Graham, R. L. (1974).
    Worst-case performance bounds for simple one-dimensional packing algorithms.
    *SIAM Journal on Computing* 3(4), 299-325.  (First fit and best fit use at
    most 17/10 OPT + 2 bins; in decreasing order, 11/9 OPT + 4, and later work
    reduced the additive constant to 6/9.)
Garey, M. R. & Johnson, D. S. (1975). Complexity results for multiprocessor
    scheduling under resource constraints. *SIAM Journal on Computing* 4(4),
    397-411.  (Strong NP-hardness.)
Martello, S. & Toth, P. (1990), chapter 8, for the lower bounds and the exact
    branch-and-bound scheme.
"""

from __future__ import annotations

import math
from typing import Sequence

__all__ = [
    "best_fit",
    "best_fit_decreasing",
    "bin_weights",
    "exact_bins",
    "first_fit",
    "first_fit_decreasing",
    "lower_bound",
]


def _validate(sizes: Sequence[int], capacity: int) -> None:
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    for size in sizes:
        if size > capacity:
            raise ValueError(f"item of size {size} does not fit in a bin of {capacity}")


def first_fit(sizes: Sequence[int], capacity: int) -> list[list[int]]:
    """Place each item in the first bin it fits in, opening a new bin if needed."""
    _validate(sizes, capacity)
    bins: list[list[int]] = []
    remaining: list[int] = []
    for index, size in enumerate(sizes):
        for b in range(len(bins)):
            if remaining[b] >= size:
                bins[b].append(index)
                remaining[b] -= size
                break
        else:
            bins.append([index])
            remaining.append(capacity - size)
    return bins


def best_fit(sizes: Sequence[int], capacity: int) -> list[list[int]]:
    """Place each item in the fullest bin that still has room for it."""
    _validate(sizes, capacity)
    bins: list[list[int]] = []
    remaining: list[int] = []
    for index, size in enumerate(sizes):
        choice, least_room = -1, None
        for b in range(len(bins)):
            room = remaining[b] - size
            if room >= 0 and (least_room is None or room < least_room):
                choice, least_room = b, room
        if choice < 0:
            bins.append([index])
            remaining.append(capacity - size)
        else:
            bins[choice].append(index)
            remaining[choice] = least_room
    return bins


def first_fit_decreasing(sizes: Sequence[int], capacity: int) -> list[list[int]]:
    """First fit over the items sorted by decreasing size: at most 11/9 OPT + 4 bins."""
    order = sorted(range(len(sizes)), key=lambda index: -sizes[index])
    packed = first_fit([sizes[i] for i in order], capacity)
    return [[order[i] for i in b] for b in packed]


def best_fit_decreasing(sizes: Sequence[int], capacity: int) -> list[list[int]]:
    """Best fit over the items sorted by decreasing size."""
    order = sorted(range(len(sizes)), key=lambda index: -sizes[index])
    packed = best_fit([sizes[i] for i in order], capacity)
    return [[order[i] for i in b] for b in packed]


def lower_bound(sizes: Sequence[int], capacity: int) -> int:
    """``max(ceil(total / capacity), number of items larger than half a bin)``.

    The classical bound of Johnson et al. (1974): the first term counts the volume,
    the second the items that pairwise cannot share a bin.  Both terms are valid
    for every packing, so the maximum is too.
    """
    _validate(sizes, capacity)
    total = sum(sizes)
    volume = math.ceil(total / capacity)
    big = sum(1 for size in sizes if 2 * size > capacity)
    return max(volume, big)


def bin_weights(bins: Sequence[Sequence[int]], sizes: Sequence[int]) -> list[int]:
    """Total size of the items in each bin."""
    return [sum(sizes[index] for index in b) for b in bins]


def exact_bins(
    sizes: Sequence[int], capacity: int, node_limit: int = 200_000
) -> list[list[int]]:
    """Fewest bins, by depth-first branch and bound over the items.

    Items are placed largest first; at each step the item goes into one of the
    existing bins that can take it, or opens a new one.  The search prunes as soon
    as the number of open bins reaches the incumbent, skips bins with equal room
    (they are interchangeable for the items still to come), and starts from the
    first-fit-decreasing packing with the volume bound as the target.

    Exponential in the worst case; the node limit keeps it responsive, and if the
    limit is reached the best packing found so far is returned, which may not be
    optimal.  For instances of the size where that matters, use the decreasing
    heuristics.
    """
    _validate(sizes, capacity)
    if not sizes:
        return []

    order = sorted(range(len(sizes)), key=lambda index: -sizes[index])
    best = first_fit_decreasing(sizes, capacity)
    target = lower_bound(sizes, capacity)
    if len(best) <= target:
        return best

    bins: list[list[int]] = []
    remaining: list[int] = []
    nodes = 0

    def search(position: int) -> None:
        nonlocal best, nodes
        if nodes >= node_limit:
            return
        nodes += 1

        if position == len(order):
            if len(bins) < len(best):
                best = [list(b) for b in bins]
            return
        if len(bins) >= len(best):
            return  # cannot improve on the incumbent

        item = order[position]
        size = sizes[item]
        tried: set[int] = set()
        for b in range(len(bins)):
            room = remaining[b]
            if room >= size and room not in tried:
                tried.add(room)
                bins[b].append(item)
                remaining[b] = room - size
                search(position + 1)
                bins[b].pop()
                remaining[b] = room

        bins.append([item])
        remaining.append(capacity - size)
        search(position + 1)
        bins.pop()
        remaining.pop()

    search(0)
    return best
