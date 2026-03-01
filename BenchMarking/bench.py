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

# ------------------------------------------------------
# BENCHMARK
# ------------------------------------------------------

Ns = np.arange(6, 19)   

mean_scores_brute = []
runtimes_brute = []

mean_scores_sa = []
runtimes_sa = []

mean_scores_greedy = []
runtimes_greedy = []

runtimes_quantum = []
mean_scores_quantum = []
for N in Ns:
    total_budget = 4e6
    # POV : insured, not reinsurer 
    # ----- Classical -----
    props = np.random.uniform(0,1, N)
    thresholds =  np.random.uniform(2e8, 5e8, N)
    premiums =  np.random.uniform(1e6, 2e6, N)
    r =  generate_N(N, props, thresholds, premiums)

    start = time.perf_counter()
    x, score = brute_force(r, premiums, total_budget)
    end = time.perf_counter()

    runtimes_brute.append(end - start)
    mean_scores_brute.append(score)

    start = time.perf_counter()
    x, score = simulated_annealing(r, premiums, total_budget)
    end = time.perf_counter()

    runtimes_sa.append(end - start)
    mean_scores_sa.append(score)

    start = time.perf_counter()
    x, score = greedy(r, premiums, total_budget)
    end = time.perf_counter()

    runtimes_greedy.append(end - start)
    mean_scores_greedy.append(score)


    # ----- Quantum -----
    start = time.perf_counter()
    end = time.perf_counter()

    runtimes_quantum.append(end - start)
    mean_scores_quantum.append(0)

print("Benchmark complete.")

# ---- Mean Score vs N ----
plt.figure()
plt.plot(Ns, mean_scores_brute, label="brute force")
plt.plot(Ns, mean_scores_sa, label="sa")
plt.plot(Ns, mean_scores_greedy, label="greedy")


plt.plot(Ns, mean_scores_quantum, label="Quantum")
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


plt.plot(Ns, runtimes_quantum, label="Quantum")
plt.xlabel("Number of Policies (N)")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime Comparison")
plt.legend()
plt.show()

# Avoid zero runtimes
eps = 1e-12
runtimes_brute = np.array(runtimes_brute) + eps
runtimes_sa = np.array(runtimes_sa) + eps
runtimes_greedy = np.array(runtimes_greedy) + eps

runtimes_quantum = np.array(runtimes_quantum) + eps

logN = np.log(Ns)
logT_brute = np.log(runtimes_brute)
logT_sa = np.log(runtimes_sa)
logT_greedy = np.log(runtimes_greedy)

logT_quantum = np.log(runtimes_quantum)

# Linear regression slope
slope_brute = np.polyfit(logN, logT_brute, 1)[0]
slope_sa = np.polyfit(logN, logT_sa, 1)[0]
slope_greedy = np.polyfit(logN, logT_greedy, 1)[0]

slope_quantum = np.polyfit(logN, logT_quantum, 1)[0]

print("Estimated complexity:")
print(f"Classical ≈ O(N^{slope_brute:.2f})")
print(f"Classical ≈ O(N^{slope_sa:.2f})")
print(f"Classical ≈ O(N^{slope_greedy:.2f})")

print(f"Quantum   ≈ O(N^{slope_quantum:.2f})")