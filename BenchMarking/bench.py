import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os

# Get the path of the parent directory (The-Bachelors-QHackathon)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from DataSet.generate_N import generate_N
from Engines.classical_solver import (
    brute_force, simulated_annealing, greedy,
)
from Engines.quantum_solver import run_qaoa

# ------------------------------------------------------
# BENCHMARK
# ------------------------------------------------------

Ns = np.arange(6, 21)   

mean_scores_brute = []
runtimes_brute = []

mean_scores_sa = []
runtimes_sa = []

mean_scores_greedy = []
runtimes_greedy = []

runtimes_qaoa = []
mean_scores_qaoa = []


for N in Ns:
    total_budget = 4e6*N/2
    # POV : insured, not reinsurer 
    # ----- Classical -----
    props = np.random.uniform(0,1, N)
    thresholds =  np.random.uniform(2e8, 5e8, N)
    premiums =  np.random.uniform(1.5e6, 2e6, N)
    r =  generate_N(N, props, thresholds, premiums)
    scale_r = np.max(np.abs(r))
    scale_c = np.max(np.abs(premiums))
    scale_B = total_budget

    r_s = r / scale_r
    c_s = premiums / scale_c
    B_s = total_budget / scale_c


    start = time.perf_counter()
    x, score = brute_force(r_s, c_s, B_s)
    end = time.perf_counter()
    runtimes_brute.append(end - start)
    mean_scores_brute.append(-score)


    start = time.perf_counter()
    x, score = simulated_annealing(r_s, c_s, B_s)
    end = time.perf_counter()

    runtimes_sa.append(end - start)
    mean_scores_sa.append(-score)

    start = time.perf_counter()
    x, score = greedy(r_s, c_s, B_s)
    end = time.perf_counter()

    runtimes_greedy.append(end - start)
    mean_scores_greedy.append(-score)


    # ----- QAOA -----
    start = time.perf_counter()
    x, score, _ = run_qaoa(r = r_s,
    premiums = c_s,
    total_budget = B_s,
    p = 1,
    lam = 1.0,
    shots = 2048,
    verbose= True)
    end = time.perf_counter()

    runtimes_qaoa.append(end - start)
    mean_scores_qaoa.append(-score)

print("Benchmark complete.")

# ---- Mean Score vs N ----
plt.figure()
plt.plot(Ns, mean_scores_brute, label="brute force")
plt.plot(Ns, mean_scores_sa, label="sa")
plt.plot(Ns, mean_scores_greedy, label="greedy")


plt.plot(Ns, mean_scores_qaoa, label="QAOA")
plt.xlabel("Number of Policies (N)")
plt.ylabel("Mean normalized r")
plt.title("Mean Score vs N")
plt.legend()
plt.show()


# ---- Runtime vs N ----
plt.figure()
plt.plot(Ns, runtimes_brute, label="brute force")
plt.plot(Ns, runtimes_sa, label="sa")
plt.plot(Ns, runtimes_greedy, label="greedy")


plt.plot(Ns, runtimes_qaoa, label="QAOA")
plt.xlabel("Number of Policies (N)")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime Comparison")
plt.legend()
plt.show()