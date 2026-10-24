# programming-algorithms

useful algorithms for software solutions

Reference implementations of the classical algorithms for three of the most
studied problems in combinatorial optimisation: the **travelling salesman
problem**, the **knapsack problem**, and **bin packing**.

Pure Python, no dependencies, no build step.  Every routine states the algorithm
it implements, the guarantee it carries, and where the guarantee comes from, so
the code can be read alongside the literature it belongs to.

## Where to start reading

If you want one book per problem, these are the definitive references, and the
implementations here follow them closely:

| Problem | Reference | Why this one |
| --- | --- | --- |
| Travelling salesman | Applegate, Bixby, Chvátal & Cook (2006), *The Traveling Salesman Problem: A Computational Study*, Princeton University Press. ISBN 978-0-691-12993-8. | The definitive computational study, by the authors of Concorde, the code that has solved every TSP instance ever posed up to 85,900 cities. 2007 INFORMS Lanchester Prize. |
| Knapsack | Kellerer, Pferschy & Pisinger (2004), *Knapsack Problems*, Springer. ISBN 978-3-540-40286-2. | The standard monograph: every variant, its exact algorithms, bounds, and approximation schemes, with proofs. |
| Both, as theory | Korte & Vygen, *Combinatorial Optimization: Theory and Algorithms*, Springer (6th ed. 2018). | The graduate text that connects these problems to polyhedral theory, LP duality and approximation. |
| Both, as classics | Papadimitriou & Steiglitz (1982), *Combinatorial Optimization: Algorithms and Complexity*, Prentice-Hall. | The book that taught a generation the branch-and-bound and dynamic-programming machinery used here. |
| Why they are hard | Garey & Johnson (1979), *Computers and Intractability: A Guide to the Theory of NP-Completeness*, Freeman. | The reference for the NP-hardness reductions these algorithms work around. |

The complete bibliography, paper by paper, is at the end of this file, and each
module repeats the citations relevant to it in its docstring.

## Quickstart

```python
from combinatorial import instances, knapsack, tsp

# --- travelling salesman -------------------------------------------------
distance = instances.random_euclidean_tsp(50, seed=1)

tour = tsp.christofides(distance)          # <= 3/2 OPT on metric instances
tour = tsp.two_opt(distance, tour)         # local search to a 2-opt optimum
tour = tsp.or_opt(distance, tour)          # relocate segments of up to 3 cities
print(tsp.tour_length(distance, tour))

lower, _ = tsp.held_karp_bound(distance, iterations=100)
print("within", tsp.tour_length(distance, tour) / lower, "of the Held-Karp bound")

# --- knapsack ------------------------------------------------------------
problem = instances.correlated_knapsack(60, seed=1, capacity=1500)

print(knapsack.branch_and_bound(problem))  # exact
print(knapsack.fptas(problem, 0.05))       # >= (1 - 0.05) OPT
print(knapsack.half_approximation(problem))  # >= OPT / 2
print(knapsack.fractional_bound(problem).value)  # upper bound on OPT
```

## What is implemented

### Travelling salesman (`combinatorial.tsp`)

A tour is a list of city indices in visit order; the edge back to the first city
is implicit.  ``n`` is the number of cities.

| Function | Guarantee (metric instances) | Cost |
| --- | --- | --- |
| `held_karp` | exact | `O(n^2 2^n)` time, `O(n 2^n)` memory |
| `branch_and_bound` | exact | exponential worst case, `O(n)` memory |
| `christofides` | at most `3/2 OPT` | `O(n^2 log n)` |
| `mst_double_tree` | at most `2 OPT` | `O(n^2)` |
| `nearest_insertion` | at most `2 OPT` | `O(n^3)` |
| `greedy_edge` | at most `0.5 (ceil(log2 n) + 1) OPT` | `O(n^2 log n)` |
| `nearest_neighbour` | `Theta(log n) OPT`, and that is tight | `O(n^2)` |
| `two_opt`, `or_opt` | none; reach a local optimum | `O(n^2)` per pass |
| `held_karp_bound` | a lower bound on OPT | `O(n^2)` per iteration |
| `brute_force_tour` | exact (oracle for tests, `n <= 10`) | `O(n!)` |

`held_karp` is the dynamic program that reduces `(n-1)!` tours to `n^2 2^n`
states.  `branch_and_bound` grows partial paths and prunes with a spanning-tree
bound.  `christofides` is the best classical guarantee; `held_karp_bound` is the
1-tree relaxation with vertex penalties, which is both a quality measure for the
heuristics and the pruning rule inside production solvers.

### Knapsack (`combinatorial.knapsack`)

Instances and solutions are immutable value objects: `Knapsack(capacity, items)`
and `Solution(value, weight, items)`.  `c` is the capacity, `P` the sum of all
profits, `n` the number of items.

| Function | Guarantee | Cost |
| --- | --- | --- |
| `dp_by_capacity` | exact | `O(n c)` time, `O(c)` memory |
| `dp_by_profit` | exact | `O(n P)` time, `O(P)` memory |
| `meet_in_the_middle` | exact | `O(2^(n/2))` time and memory |
| `branch_and_bound` | exact | exponential worst case, `O(n)` memory |
| `unbounded_dp` | exact, items reusable | `O(n c)` |
| `fptas` | at least `(1 - eps) OPT` | `O(n^2 / eps)` profit levels |
| `half_approximation` | at least `OPT / 2` | `O(n log n)` |
| `greedy_density` | none | `O(n log n)` |
| `fractional_bound` | an upper bound on OPT (LP relaxation) | `O(n log n)` |

The dynamic programs are the textbook recurrences; the selected items are
recovered by divide and conquer over the items rather than by storing an
`n x c` table of decisions, which keeps the memory at `O(c)` for a `log n`
factor in time (the same trick as Hirschberg's linear-space alignment).

### Bin packing (`combinatorial.binpacking`)

The counting version of "how many boxes fit": pack every item using as few bins
as possible.  This problem is NP-hard in the strong sense, so unlike knapsack it
has no FPTAS.

| Function | Guarantee | Cost |
| --- | --- | --- |
| `first_fit`, `best_fit` | at most `17/10 OPT + 2` bins | `O(n^2)` |
| `first_fit_decreasing`, `best_fit_decreasing` | at most `11/9 OPT + 4` bins | `O(n log n)` |
| `lower_bound` | at most OPT, and attained on many instances | `O(n)` |
| `exact_bins` | exact (branch and bound, small instances) | exponential |

## Running the tests and the examples

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest          # 234 tests, ~45 s
.venv/bin/python examples/demo.py --out figures
```

The tests are not smoke tests: the exact solvers are checked against exhaustive
enumeration, and the approximate ones against the guarantee they claim, on
instances where that guarantee applies.  The suite includes, among others,

* `held_karp` and `branch_and_bound` against brute force, on symmetric and
  asymmetric instances;
* `christofides <= 1.5 OPT`, `mst_double_tree <= 2 OPT`,
  `nearest_insertion <= 2 OPT` over many random Euclidean instances;
* `two_opt` returning a tour in which no 2-opt move improves (the definition of
  a local optimum);
* `held_karp_bound <= OPT` — the lower-bound property, which is easy to break;
* every knapsack solver against exhaustive enumeration, on uncorrelated,
  strongly correlated and subset-sum instances;
* `fptas >= (1 - eps) OPT` and `half_approximation >= OPT/2`, including on
  adversarial instances where the fractional split item does not fit;
* bin packing against an exhaustive search for the minimum number of bins,
  including an instance where first-fit-decreasing provably uses 9 bins while 8
  suffice.

## Layout

```
src/combinatorial/
  graphs.py           Prim's MST, union-find, Hierholzer's Eulerian circuit
  matching.py         minimum-weight perfect matching (subset DP + greedy)
  instances.py        instance generators (Euclidean, correlated, subset-sum)
  tsp/
    exact.py          brute force, Held-Karp dynamic program
    constructive.py   nearest neighbour, insertion, cheapest link, double tree, Christofides
    local_search.py   2-opt and Or-opt
    bounds.py         1-tree and the Held-Karp bound by subgradient ascent
    branch_bound.py   exact search with a spanning-tree bound
    tours.py          tour length and validation
  knapsack/
    models.py         Item, Knapsack, Solution
    dp.py             the two value functions and linear-space reconstruction
    exact.py          capacity DP, profit DP, meet in the middle, branch and bound, unbounded
    approx.py         fractional bound, greedy, 1/2-approximation, FPTAS
  binpacking.py       first fit, best fit, decreasing variants, lower bound, exact search
tests/                the suite described above
examples/demo.py      worked examples that print a report and draw the figures
```

## Deliberately out of scope

These are the algorithms a production solver would use; they are named here so
the gap is explicit rather than accidental.

* **Cutting planes (Dantzig-Fulkerson-Johnson) and Concorde.** The exact TSP
  solvers here are dynamic programming and branch and bound.  The method that
  solves 85,900-city instances adds subtour and comb inequalities to a linear
  program; it needs an LP solver and a separation oracle, and it is the subject
  of the Applegate-Bixby-Chvátal-Cook book.
* **Edmonds' blossom algorithm.** Matching is solved here exactly by a subset
  dynamic program, which is fine for the odd-degree sets Christofides produces
  (up to ~20 vertices) but does not scale like the blossom algorithm.
* **Lin-Kernighan and chained local search.** `two_opt` and `or_opt` are the
  moves inside Lin-Kernighan; the variable-depth version and its chained
  variants are what actually close the last few percent on large instances.
* **Martello-Toth style core problems and combo/minknap.** The branch-and-bound
  solver here uses the Dantzig bound with no core reduction; the algorithms in
  Kellerer, Pferschy & Pisinger, chapter 5, are faster on large instances.

## Bibliography

Travelling salesman

* Dantzig, G., Fulkerson, R. & Johnson, S. (1954). Solution of a large-scale
  traveling-salesman problem. *Operations Research* 2(4), 393-410.
* Held, M. & Karp, R. M. (1962). A dynamic programming approach to sequencing
  problems. *SIAM Journal on Applied Mathematics* 10(1), 196-210.
* Bellman, R. (1962). Dynamic programming treatment of the travelling salesman
  problem. *Journal of the ACM* 9(1), 61-63.
* Little, J. D. C., Murty, K. G., Sweeney, D. W. & Karel, C. (1963). An
  algorithm for the traveling salesman problem. *Operations Research* 11(6),
  972-989.
* Held, M. & Karp, R. M. (1970). The traveling-salesman problem and minimum
  spanning trees. *Operations Research* 18(6), 1138-1162.  And part II (1971),
  *Mathematical Programming* 1(1), 6-25.
* Karp, R. M. (1972). Reducibility among combinatorial problems. In *Complexity
  of Computer Computations*, Plenum, 85-103.
* Lin, S. & Kernighan, B. W. (1973). An effective heuristic algorithm for the
  traveling-salesman problem. *Operations Research* 21(2), 498-516.
* Christofides, N. (1976). Worst-case analysis of a new heuristic for the
  travelling salesman problem. Technical Report 388, GSIA, Carnegie Mellon
  University.  (Reprinted in *Operations Research Forum* 3, 20, 2022.)
* Rosenkrantz, D. J., Stearns, R. E. & Lewis, P. M. (1977). An analysis of
  several heuristics for the traveling salesman problem. *SIAM Journal on
  Computing* 6(3), 563-581.
* Cornuéjols, G. & Nemhauser, G. L. (1978). Tight bounds for Christofides'
  traveling salesman heuristic. *Mathematical Programming* 14(1), 116-121.
* Volgenant, A. & Jonker, R. (1982). A branch and bound algorithm for the
  symmetric traveling salesman problem based on the 1-tree relaxation.
  *European Journal of Operational Research* 9(1), 83-89.
* Ong, H. L. & Moore, J. B. (1984). Worst-case analysis of two travelling
  salesman heuristics. *Operations Research Letters* 2(6), 273-277.
* Johnson, D. S. & McGeoch, L. A. (1997). The traveling salesman problem: a case
  study in local optimization. In *Local Search in Combinatorial Optimization*,
  Wiley, 215-310.
* Applegate, D. L., Bixby, R. E., Chvátal, V. & Cook, W. J. (2006). *The
  Traveling Salesman Problem: A Computational Study*. Princeton University
  Press.

Knapsack

* Dantzig, G. B. (1957). Discrete-variable extremum problems. *Operations
  Research* 5(2), 266-288.
* Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
* Nemhauser, G. L. & Ullmann, Z. (1969). Discrete dynamic programming and
  capital allocation. *Management Science* 15(9), 494-505.
* Horowitz, E. & Sahni, S. (1974). Computing partitions with applications to the
  knapsack problem. *Journal of the ACM* 21(2), 277-292.
* Sahni, S. (1975). Approximate algorithms for the 0/1 knapsack problem.
  *Journal of the ACM* 22(1), 115-124.
* Ibarra, O. H. & Kim, C. E. (1975). Fast approximation algorithms for the
  knapsack and sum of subset problems. *Journal of the ACM* 22(4), 463-468.
* Lawler, E. L. (1979). Fast approximation algorithms for knapsack problems.
  *Mathematics of Operations Research* 4(4), 339-356.
* Martello, S. & Toth, P. (1990). *Knapsack Problems: Algorithms and Computer
  Implementations*. Wiley.
* Pisinger, D. (1997). A minimal algorithm for the 0-1 knapsack problem.
  *Operations Research* 45(5), 758-767.
* Martello, S., Pisinger, D. & Toth, P. (1999). Dynamic programming and strong
  bounds for the 0-1 knapsack problem. *Management Science* 45(3), 414-424.
* Kellerer, H., Pferschy, U. & Pisinger, D. (2004). *Knapsack Problems*.
  Springer.
* Pisinger, D. (2005). Where are the hard knapsack problems? *Computers &
  Operations Research* 32(9), 2271-2284.

Bin packing

* Johnson, D. S., Demers, A., Ullman, J. D., Garey, M. R. & Graham, R. L.
  (1974). Worst-case performance bounds for simple one-dimensional packing
  algorithms. *SIAM Journal on Computing* 3(4), 299-325.
* Garey, M. R. & Johnson, D. S. (1975). Complexity results for multiprocessor
  scheduling under resource constraints. *SIAM Journal on Computing* 4(4),
  397-411.
* Dósa, G. (2007). The tight bound of first fit decreasing bin-packing algorithm
  is `FFD(I) <= 11/9 OPT(I) + 6/9`. In *Combinatorics, Algorithms, Probabilistic
  and Experimental Methodologies*, Springer, 1-11.

General

* Karp, R. M. (1972), as above; Garey & Johnson (1979), as above.
* Papadimitriou, C. H. & Steiglitz, K. (1982). *Combinatorial Optimization:
  Algorithms and Complexity*. Prentice-Hall.
* Hirschberg, D. S. (1975). A linear space algorithm for computing maximal
  common subsequences. *Communications of the ACM* 18(6), 341-343.
* Korte, B. & Vygen, J. (2018). *Combinatorial Optimization: Theory and
  Algorithms* (6th ed.). Springer.
