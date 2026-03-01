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

from dataset_dummy import ReinsuranceDataset, generate_dataset, print_dataset

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

import numpy as np
import time

def brute_force(N, r, premiums, total_budget):

    start = time.time()

    best_score = -np.inf
    best_x = None

    # iterate over all 2^N combinations
    for i in range(2 ** N):

        # convert integer to binary vector of length N
        x = np.array(list(np.binary_repr(i, width=N)), dtype=int)

        # compute total cost
        total_cost = np.dot(premiums, x)

        # check budget constraint
        if total_cost <= total_budget:

            score = np.dot(r, x)

            if score > best_score:
                best_score = score
                best_x = x

    runtime = time.time() - start

    return best_x, best_score, runtime

# ---------------------------------------------------------------------------
# 2. Simulated Annealing
# ---------------------------------------------------------------------------

def simulated_annealing(
    N,
    r,
    premiums,
    total_budget,
    T_init=100.0,
    T_min=1e-3,
    cooling=0.95,
    max_iter=10000,
    lambda_penalty=5.0,
    seed=123,
):
    """
    Simulated Annealing for:

    max r^T x
    s.t. premiums^T x ≤ total_budget

    Uses soft penalty:
    f(x) = r^T x − λ * max(0, premiums^T x − B)^2
    """

    r = np.array(r, dtype=float)
    premiums = np.array(premiums, dtype=float)

    rng = random.Random(seed)

    # Penalised objective (higher is better)
    def objective(x):
        cost = premiums @ x
        risk = r @ x
        violation = max(0.0, cost - total_budget)
        return risk - lambda_penalty * violation**2

    # --- Initial feasible solution ---
    x_cur = np.zeros(N)
    indices = list(range(N))
    rng.shuffle(indices)

    for i in indices:
        if premiums[i] + premiums @ x_cur <= total_budget:
            x_cur[i] = 1

    f_cur = objective(x_cur)

    # Track best feasible solution
    best_x = x_cur.copy()
    best_f = f_cur if premiums @ best_x <= total_budget else -np.inf

    T = T_init

    for _ in range(max_iter):

        if T < T_min:
            break

    # --- Flip random bit ---
        flip_idx = rng.randint(0, N - 1)
        x_new = x_cur.copy()
        x_new[flip_idx] = 1 - x_new[flip_idx]

        f_new = objective(x_new)
        delta = f_new - f_cur

        # --- Metropolis criterion ---
        if delta > 0 or rng.random() < math.exp(delta / T):
            x_cur = x_new
            f_cur = f_new

        # Update best feasible
            if premiums @ x_cur <= total_budget and f_cur > best_f:
                best_f = f_cur
                best_x = x_cur.copy()

        T *= cooling

    best_score = r @ best_x

    return best_x, best_score


# ---------------------------------------------------------------------------
# 3. Greedy Heuristic
# ---------------------------------------------------------------------------

def greedy(N, r, premiums, total_budget):
    """
    Returns
    x : np.ndarray
    Selected binary vector
    total_risk : float
    Achieved objective value
    """

    r = np.array(r, dtype=float)
    premiums = np.array(premiums, dtype=float)

    # Compute efficiency ratio r_i / c_i
    ratios = r / premiums

    # Sort indices in descending order of ratio
    order = np.argsort(-ratios)

    x = np.zeros(N)
    remaining_budget = total_budget

    for i in order:
        if premiums[i] <= remaining_budget:
            x[i] = 1
            remaining_budget -= premiums[i]

    total_risk = r @ x

    return x, total_risk


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
