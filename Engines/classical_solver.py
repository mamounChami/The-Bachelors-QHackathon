"""
classical_solver.py — Classical Benchmarks for Reinsurance Allocation
=======================================================================
Implements three classical approaches to the binary knapsack / reinsurance
allocation problem:

  1. Brute-Force  — guaranteed optimal (exponential time, feasible for n ≤ 20)
  2. Simulated Annealing (SA) — metaheuristic, near-optimal for any n
  3. Greedy Heuristic — fast but not optimal
"""
import math
import random
import numpy as np

# ---------------------------------------------------------------------------
# 1. Brute-Force Exact Solver
# ---------------------------------------------------------------------------
def brute_force(r_values, premiums, total_budget):
    N = r_values.shape[1]
    
    # 1. Generate all 2^N - 1 combinations
    numbers = np.arange(1, 2**N, dtype=np.uint32)
    combinations = ((numbers[:, None] & (1 << np.arange(N)[::-1])) > 0).astype(int).T
    
    # 2. Calculate total profit and total cost for EVERY combination
    all_profits = np.dot(r_values, combinations).flatten()
    all_costs = np.dot(premiums, combinations).flatten()
    
    # 3. Filter combinations that are over budget
    valid_mask = all_costs <= total_budget
    
    if not np.any(valid_mask):
        return None, "No valid combination within budget", 0
    
    # 4. Find the combination with the highest profit
    best_relative_idx = np.argmax(all_profits[valid_mask])
    original_idx = np.where(valid_mask)[0][best_relative_idx]
    
    best_combination = combinations[:, original_idx]
    best_profit = all_profits[original_idx]
    best_cost = all_costs[original_idx]
    
    return best_combination, best_profit

# ---------------------------------------------------------------------------
# 2. Simulated Annealing
# ---------------------------------------------------------------------------

def simulated_annealing(
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

    Robust to r being shape (N,) or (1,N).
    """

    # ---- FIX: force 1D vectors ----
    r = np.asarray(r, dtype=float).reshape(-1)           # (N,)
    premiums = np.asarray(premiums, dtype=float).reshape(-1)  # (N,)
    N = r.size

    rng = random.Random(seed)

    # Penalised objective (higher is better)
    def objective(x):
        cost = float(np.dot(premiums, x))               # scalar
        score = float(np.dot(r, x))                     # scalar
        violation = max(0.0, cost - total_budget)
        return score - lambda_penalty * (violation ** 2)

    # --- Initial feasible solution ---
    x_cur = np.zeros(N, dtype=int)
    indices = list(range(N))
    rng.shuffle(indices)

    cur_cost = 0.0
    for i in indices:
        if cur_cost + premiums[i] <= total_budget:
            x_cur[i] = 1
            cur_cost += premiums[i]

    f_cur = float(objective(x_cur))

    # Track best feasible solution
    best_x = x_cur.copy()
    best_f = f_cur if float(np.dot(premiums, best_x)) <= total_budget else -np.inf

    T = T_init

    for _ in range(max_iter):
        if T < T_min:
            break

        # --- Flip random bit ---
        flip_idx = rng.randint(0, N - 1)
        x_new = x_cur.copy()
        x_new[flip_idx] = 1 - x_new[flip_idx]

        f_new = float(objective(x_new))
        delta = f_new - f_cur  # now guaranteed scalar

        # --- Metropolis criterion ---
        if delta > 0.0 or rng.random() < math.exp(delta / T):
            x_cur = x_new
            f_cur = f_new

            # Update best feasible
            if float(np.dot(premiums, x_cur)) <= total_budget and f_cur > best_f:
                best_f = f_cur
                best_x = x_cur.copy()

        T *= cooling

    best_score = float(np.dot(r, best_x))
    return best_x, best_score

# ---------------------------------------------------------------------------
# 3. Greedy Heuristic
# ---------------------------------------------------------------------------

import numpy as np

def greedy(r, premiums, total_budget):
    """
    Greedy for:
      max r^T x
      s.t. premiums^T x <= total_budget

    Robust to r being shape (N,) or (1,N).
    Returns:
      x (N,) binary vector
      total_score (float)
    """

    # ---- FIX: force 1D ----
    r = np.asarray(r, dtype=float).reshape(-1)              # (N,)
    premiums = np.asarray(premiums, dtype=float).reshape(-1) # (N,)
    N = r.size

    # Avoid division by zero (just in case)
    premiums_safe = np.where(premiums == 0.0, 1e-12, premiums)

    # Efficiency ratio
    ratios = r / premiums_safe                              # (N,)

    # Sort indices descending
    order = np.argsort(-ratios)                             # (N,)

    x = np.zeros(N, dtype=int)
    remaining_budget = float(total_budget)

    for i in order:
        if premiums[i] <= remaining_budget:
            x[i] = 1
            remaining_budget -= premiums[i]

    total_score = float(np.dot(r, x))
    return x, total_score
