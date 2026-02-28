"""
classical_solver.py — Classical Benchmarks for Reinsurance Allocation
=======================================================================
Implements three classical approaches to the binary knapsack / reinsurance
allocation problem:

  1. Brute-Force  — guaranteed optimal (exponential time, feasible for n ≤ 20)
  2. Simulated Annealing (SA) — metaheuristic, near-optimal for any n
  3. Greedy Heuristic — fast but not optimal

All solvers share a common SolverResult interface for easy benchmarking.
"""

import time
import math
import random
import numpy as np
from dataclasses import dataclass
from itertools import product
from typing import Optional

from dataset import ReinsuranceDataset, generate_dataset, print_dataset


# ---------------------------------------------------------------------------
# Shared result container
# ---------------------------------------------------------------------------

@dataclass
class SolverResult:
    """Standardised output for every solver."""
    solver_name: str
    bitstring: str          # e.g. '1010110010'
    cost: float             # total cost of selected protections
    risk_reduction: float   # total risk reduction achieved
    runtime_s: float        # wall-clock time in seconds
    feasible: bool          # does solution satisfy budget constraint?

    def __str__(self) -> str:
        feas_str = "✓ FEASIBLE" if self.feasible else "✗ INFEASIBLE"
        return (
            f"[{self.solver_name}]\n"
            f"  Bitstring      : {self.bitstring}\n"
            f"  Risk reduction : {self.risk_reduction:.2f}\n"
            f"  Cost           : {self.cost:.2f}\n"
            f"  Status         : {feas_str}\n"
            f"  Runtime        : {self.runtime_s*1000:.2f} ms"
        )


def _make_result(
    name: str, bitstring: str, data: ReinsuranceDataset, t_start: float
) -> SolverResult:
    """Helper: build a SolverResult from a bitstring and start time."""
    cost = data.cost_of(bitstring)
    risk = data.risk_of(bitstring)
    return SolverResult(
        solver_name=name,
        bitstring=bitstring,
        cost=cost,
        risk_reduction=risk,
        runtime_s=time.perf_counter() - t_start,
        feasible=cost <= data.budget,
    )


# ---------------------------------------------------------------------------
# 1. Brute-Force Exact Solver
# ---------------------------------------------------------------------------

def brute_force(data: ReinsuranceDataset) -> SolverResult:
    """
    Exact solver via exhaustive enumeration.

    Iterates all 2^n binary strings and returns the one that:
      - satisfies the budget constraint  (cost ≤ B)
      - maximises total risk reduction

    Time complexity  : O(2^n · n)
    Space complexity : O(n)
    Feasible for     : n ≤ ~20  (2^20 ≈ 1 M  iterations ≈ <1 s)

    Returns
    -------
    SolverResult with the globally optimal solution.
    """
    t0 = time.perf_counter()

    n = data.n
    best_risk = -1.0
    best_bs   = "0" * n

    # Iterate every integer from 0 to 2^n − 1
    # Each integer encodes a subset via its binary representation
    for mask in range(2 ** n):
        # Decode mask into a numpy vector x ∈ {0,1}^n
        x = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)

        cost = float(data.costs @ x)
        if cost > data.budget:          # prune infeasible solutions
            continue

        risk = float(data.risks @ x)
        if risk > best_risk:
            best_risk = risk
            best_bs   = "".join(str(int(x[i])) for i in range(n))

    return _make_result("BruteForce", best_bs, data, t0)


# ---------------------------------------------------------------------------
# 2. Simulated Annealing
# ---------------------------------------------------------------------------

def simulated_annealing(
    data: ReinsuranceDataset,
    T_init: float   = 100.0,
    T_min: float    = 1e-3,
    cooling: float  = 0.95,
    max_iter: int   = 10_000,
    lambda_penalty: float = 5.0,
    seed: int = 123,
) -> SolverResult:
    """
    Simulated Annealing (SA) for the constrained binary knapsack.

    The constraint Σ c_i x_i ≤ B is enforced via a soft penalty term in
    the objective:

        f(x) = Σ r_i x_i  −  λ · max(0, Σ c_i x_i − B)²

    SA explores the solution space by randomly flipping single bits
    (neighbours) and accepts worse moves with probability exp(-Δ/T).
    The temperature T is cooled exponentially each iteration.

    Parameters
    ----------
    T_init         : initial temperature
    T_min          : stop when temperature drops below this value
    cooling        : multiplicative cooling factor (0 < cooling < 1)
    max_iter       : maximum number of SA steps
    lambda_penalty : penalty weight for constraint violation
    seed           : random seed for reproducibility

    Returns
    -------
    SolverResult with the best feasible solution found.
    """
    t0 = time.perf_counter()
    rng = random.Random(seed)
    n   = data.n

    # ── Penalised objective (higher = better) ─────────────────────────────
    def objective(x: np.ndarray) -> float:
        cost       = float(data.costs @ x)
        risk       = float(data.risks @ x)
        violation  = max(0.0, cost - data.budget)
        return risk - lambda_penalty * violation ** 2

    # ── Initialise at a random feasible point ─────────────────────────────
    x_cur = np.zeros(n, dtype=float)
    # Greedily add random items while feasible
    indices = list(range(n))
    rng.shuffle(indices)
    for i in indices:
        if data.costs[i] + float(data.costs @ x_cur) <= data.budget:
            x_cur[i] = 1.0

    f_cur = objective(x_cur)

    # Track the best *feasible* solution separately
    best_x   = x_cur.copy()
    best_f   = f_cur if data.cost_of("".join(map(str, x_cur.astype(int)))) <= data.budget else -np.inf

    T = T_init

    for _ in range(max_iter):
        if T < T_min:
            break

        # ── Generate neighbour by flipping a random bit ───────────────────
        flip_idx  = rng.randint(0, n - 1)
        x_new     = x_cur.copy()
        x_new[flip_idx] = 1.0 - x_new[flip_idx]   # flip 0↔1

        f_new = objective(x_new)
        delta = f_new - f_cur

        # ── Metropolis acceptance criterion ───────────────────────────────
        if delta > 0 or rng.random() < math.exp(delta / T):
            x_cur = x_new
            f_cur = f_new

            # Update best feasible if this state is better
            bs = "".join(map(str, x_cur.astype(int)))
            if data.is_feasible(bs) and f_cur > best_f:
                best_f = f_cur
                best_x = x_cur.copy()

        T *= cooling   # cool the temperature

    best_bs = "".join(map(str, best_x.astype(int)))
    return _make_result("SimulatedAnnealing", best_bs, data, t0)


# ---------------------------------------------------------------------------
# 3. Greedy Heuristic
# ---------------------------------------------------------------------------

def greedy(data: ReinsuranceDataset) -> SolverResult:
    """
    Greedy heuristic based on the risk-to-cost efficiency ratio.

    Algorithm:
      1. Sort protections in descending order of r_i / c_i
      2. Select each protection if it fits within the remaining budget

    This is the classical fractional-knapsack greedy, applied to the 0/1
    variant.  It runs in O(n log n) but is NOT guaranteed to be optimal.

    Returns
    -------
    SolverResult (feasible by construction, but possibly suboptimal).
    """
    t0 = time.perf_counter()
    n  = data.n

    # Compute efficiency ratios and sort
    ratios  = data.risks / data.costs               # r_i / c_i
    order   = np.argsort(-ratios)                   # descending

    x        = np.zeros(n, dtype=float)
    remaining = data.budget

    for i in order:
        if data.costs[i] <= remaining:
            x[i]      = 1.0
            remaining -= data.costs[i]

    best_bs = "".join(map(str, x.astype(int)))
    return _make_result("Greedy", best_bs, data, t0)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison(results: list[SolverResult], data: ReinsuranceDataset) -> None:
    """Print a side-by-side comparison table of multiple solver results."""
    print(f"\n{'═'*68}")
    print(f"  {'Solver':<22} {'Risk Reduc.':>12} {'Cost':>8} {'Feasible':>10} {'Time (ms)':>12}")
    print(f"{'─'*68}")

    # Find best feasible risk reduction to mark the winner
    best_risk = max(
        (r.risk_reduction for r in results if r.feasible), default=0.0
    )

    for r in results:
        marker = " ★" if r.feasible and r.risk_reduction >= best_risk else "  "
        feas   = "YES" if r.feasible else "NO"
        print(
            f"  {r.solver_name:<22} {r.risk_reduction:>12.2f} {r.cost:>8.2f} "
            f"{feas:>10} {r.runtime_s*1000:>11.2f} ms{marker}"
        )

    print(f"{'─'*68}")
    print(f"  Budget: {data.budget:.2f}   Max possible risk: {data.total_risk:.2f}")
    print(f"{'═'*68}\n")


def approx_ratio(result: SolverResult, optimal_risk: float) -> float:
    """Approximation ratio: result_risk / optimal_risk."""
    if optimal_risk <= 0:
        return 0.0
    return result.risk_reduction / optimal_risk


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 68)
    print("  CLASSICAL SOLVERS — DEMO")
    print("=" * 68)

    data = generate_dataset(n=10, seed=42)
    print_dataset(data)

    # Run all three solvers
    bf  = brute_force(data)
    sa  = simulated_annealing(data)
    gr  = greedy(data)

    # Print individual results
    for res in [bf, sa, gr]:
        print(res)
        print()

    # Print comparison table
    print_comparison([bf, sa, gr], data)

    # Approximation ratios relative to brute-force optimal
    print("Approximation ratios (vs. brute-force optimal):")
    for res in [sa, gr]:
        ratio = approx_ratio(res, bf.risk_reduction)
        print(f"  {res.solver_name:<22} {ratio:.4f}  ({ratio*100:.1f}%)")
    print()
