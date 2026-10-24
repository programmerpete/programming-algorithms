"""Worked examples: run every solver, print a report, draw the figures.

    python examples/demo.py --out figures

The travelling salesman instance is small enough that the exact solvers can
certify the optimum, which is what makes the ratios in the table meaningful.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from combinatorial import binpacking, instances, knapsack, tsp


def rule(title: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}")


def print_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))
    ]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def draw_tour(ax, points, tour, title: str, colour: str) -> None:
    order = list(tour) + [tour[0]]
    ax.plot([points[i][0] for i in order], [points[i][1] for i in order], color=colour, lw=1.4)
    ax.scatter([p[0] for p in points], [p[1] for p in points], s=28, color="#222222", zorder=3)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def travelling_salesman(out_dir: Path) -> None:
    rule("Travelling salesman: 18 cities, Euclidean distances")
    points = instances.random_points(18, seed=10)
    distance = instances.euclidean_matrix(points)

    optimal = tsp.held_karp(distance)
    optimal_length = tsp.tour_length(distance, optimal)
    bound, _ = tsp.held_karp_bound(distance, iterations=200)

    print(f"exact optimum (Held-Karp dynamic program)  {optimal_length:9.2f}")
    print(f"Held-Karp 1-tree lower bound               {bound:9.2f}")
    print(f"certified gap                              {100 * (optimal_length - bound) / optimal_length:9.2f}%")

    improved = tsp.or_opt(distance, tsp.two_opt(distance, tsp.christofides(distance)))
    tours = [
        ("nearest neighbour", tsp.nearest_neighbour(distance)),
        ("cheapest link", tsp.greedy_edge(distance)),
        ("nearest insertion", tsp.nearest_insertion(distance)),
        ("double tree", tsp.mst_double_tree(distance)),
        ("Christofides", tsp.christofides(distance)),
        ("Christofides + 2-opt", tsp.two_opt(distance, tsp.christofides(distance))),
        ("... + Or-opt", improved),
        ("branch and bound", tsp.branch_and_bound(distance, upper_bound=improved)),
    ]

    rows = []
    for name, tour in tours:
        length = tsp.tour_length(distance, tour)
        rows.append(
            (
                name,
                f"{length:.2f}",
                f"{length / optimal_length:.3f}",
                f"{100 * (length - bound) / bound:.2f}%",
            )
        )
    print_table(("method", "length", "vs optimum", "above HK bound"), rows)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    draw_tour(axes[0], points, tours[0][1], f"nearest neighbour ({rows[0][1]})", "#c0392b")
    draw_tour(axes[1], points, tours[4][1], f"Christofides ({rows[4][1]})", "#2471a3")
    draw_tour(axes[2], points, optimal, f"optimal ({optimal_length:.2f})", "#1e8449")
    figure.suptitle("18-city instance: heuristic tours against the exact optimum")
    figure.tight_layout()
    figure.savefig(out_dir / "tsp_tours.png", dpi=140)
    plt.close(figure)

    counts = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    bounds = [tsp.held_karp_bound(distance, iterations=k)[0] for k in counts]
    figure, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(counts, bounds, marker="o", label="Held-Karp bound")
    ax.axhline(optimal_length, color="#1e8449", ls="--", label="optimum")
    ax.axhline(
        tsp.tour_length(distance, improved), color="#2471a3", ls=":", label="2-opt + Or-opt tour"
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("subgradient iterations")
    ax.set_ylabel("tour length")
    ax.set_title("The 1-tree bound climbs towards the optimum as penalties are fitted")
    ax.legend()
    ax.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_dir / "tsp_bound.png", dpi=140)
    plt.close(figure)


def knapsack_problem() -> knapsack.Knapsack:
    """60 strongly correlated items with profits in the thousands."""
    rng = random.Random(5)
    pairs = []
    for _ in range(60):
        weight = rng.randint(100, 1000)
        pairs.append((weight, weight + 10))
    capacity = sum(weight for weight, _ in pairs) // 2
    return knapsack.Knapsack.from_pairs(capacity, pairs)


def knapsack_report(out_dir: Path) -> None:
    problem = knapsack_problem()
    rule(
        f"Knapsack: {problem.n} strongly correlated items (p = w + 10), "
        f"capacity {problem.capacity}"
    )

    start = time.perf_counter()
    exact = knapsack.branch_and_bound(problem)
    exact_seconds = time.perf_counter() - start
    bound = knapsack.fractional_bound(problem)
    print(f"exact optimum (branch and bound)  {exact.value:8d}  in {exact_seconds * 1000:.0f} ms")
    print(f"LP relaxation upper bound         {bound.value:8.1f}")
    print(f"optimal packing weight            {exact.weight:8d} of {problem.capacity}")

    rows = []
    for name, solution, seconds in [
        ("fractional bound (LP)", None, None),
        ("density greedy", knapsack.greedy_density(problem), None),
        ("1/2-approximation", knapsack.half_approximation(problem), None),
    ]:
        if solution is None:
            rows.append((name, f"{bound.value:.1f}", f"{bound.value / exact.value:.4f}", "-"))
        else:
            rows.append((name, str(solution.value), f"{solution.value / exact.value:.4f}", "-"))

    epsilons = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    fptas_rows = []
    quality = []
    for epsilon in epsilons:
        start = time.perf_counter()
        solution = knapsack.fptas(problem, epsilon)
        seconds = time.perf_counter() - start
        ratio = solution.value / exact.value
        quality.append(ratio)
        fptas_rows.append(
            (
                f"fptas eps={epsilon}",
                str(solution.value),
                f"{ratio:.4f}",
                f"{seconds * 1000:.0f} ms",
            )
        )
    rows.extend(fptas_rows)
    print_table(("method", "value", "vs optimum", "time"), rows)

    figure, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(epsilons, quality, marker="o", label="achieved")
    ax.plot(epsilons, [1 - epsilon for epsilon in epsilons], ls="--", label="guarantee (1 - eps)")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(0.98, 1.002)
    ax.set_xlabel("epsilon (approximation parameter)")
    ax.set_ylabel("value / optimum")
    ax.set_title("FPTAS: the guarantee is met with room to spare on this instance")
    ax.legend()
    ax.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_dir / "knapsack_fptas.png", dpi=140)
    plt.close(figure)


def bin_packing(out_dir: Path) -> None:
    rule("Bin packing: an instance where the decreasing heuristics are not optimal")
    sizes = [29, 36, 60, 26, 40, 56, 30, 21, 46, 46, 24, 26, 28, 40, 50, 57, 48, 46, 33, 32]
    capacity = 100
    first_fit_decreasing = binpacking.first_fit_decreasing(sizes, capacity)
    exact = binpacking.exact_bins(sizes, capacity)
    print(f"total size {sum(sizes)} -> volume lower bound {binpacking.lower_bound(sizes, capacity)} bins")
    print(f"first fit decreasing  {len(first_fit_decreasing)} bins")
    print(f"exact search          {len(exact)} bins")

    figure, axes = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
    for ax, bins, title in [
        (axes[0], first_fit_decreasing, f"first fit decreasing: {len(first_fit_decreasing)} bins"),
        (axes[1], exact, f"exact search: {len(exact)} bins"),
    ]:
        for row, contents in enumerate(bins):
            left = 0
            for index in contents:
                ax.barh(row, sizes[index], left=left, height=0.7, edgecolor="white", linewidth=1)
                ax.text(left + sizes[index] / 2, row, str(sizes[index]), ha="center", va="center", fontsize=7)
                left += sizes[index]
            ax.barh(row, capacity - left, left=left, height=0.7, color="#eeeeee", edgecolor="white")
        ax.axvline(capacity, color="#c0392b", lw=1.2)
        ax.set_xlim(0, capacity * 1.06)
        ax.set_yticks(range(len(bins)))
        ax.set_ylabel("bin")
        ax.set_title(title, fontsize=10)
    axes[1].set_xlabel("size (bin capacity 100)")
    figure.tight_layout()
    figure.savefig(out_dir / "binpacking.png", dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("figures"), help="figure directory")
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)

    travelling_salesman(arguments.out)
    knapsack_report(arguments.out)
    bin_packing(arguments.out)

    print(f"\nfigures written to {arguments.out}/")


if __name__ == "__main__":
    main()
