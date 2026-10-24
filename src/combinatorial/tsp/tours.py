"""Tour helpers.

A tour is a list of city indices in visit order; the edge from the last city
back to the first is implicit, so a tour of ``n`` cities has ``n`` edges.  Every
solver in this package returns a tour in that form.
"""

from __future__ import annotations

from typing import Sequence


def tour_length(distance: Sequence[Sequence[float]], tour: Sequence[int]) -> float:
    """Length of the closed tour, including the edge back to the start."""
    n = len(tour)
    return sum(distance[tour[i]][tour[(i + 1) % n]] for i in range(n))


def validate_tour(distance: Sequence[Sequence[float]], tour: Sequence[int]) -> None:
    """Raise ``ValueError`` unless ``tour`` visits every city exactly once."""
    n = len(distance)
    if len(tour) != n:
        raise ValueError(f"tour visits {len(tour)} of {n} cities")
    if sorted(tour) != list(range(n)):
        raise ValueError("tour is not a permutation of the cities")
