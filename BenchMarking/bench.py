import numpy as np
import time
import matplotlib.pyplot as plt

def compute_r_values(N):
    return 0

def compute_r_values_quantum(N):
    return 0

# ------------------------------------------------------
# BENCHMARK
# ------------------------------------------------------

Ns = np.arange(2, 51)   

mean_scores_classical = []
mean_scores_quantum = []

runtimes_classical = []
runtimes_quantum = []

for N in Ns:

    # ----- Classical -----
    start = time.perf_counter()
    r_classical = compute_r_values(N)
    end = time.perf_counter()

    runtimes_classical.append(end - start)
    mean_scores_classical.append(np.mean(r_classical))

    # ----- Quantum -----
    start = time.perf_counter()
    r_quantum = compute_r_values_quantum(N)
    end = time.perf_counter()

    runtimes_quantum.append(end - start)
    mean_scores_quantum.append(np.mean(r_quantum))

print("Benchmark complete.")

# ---- Mean Score vs N ----
plt.figure()
plt.plot(Ns, mean_scores_classical, label="Classical")
plt.plot(Ns, mean_scores_quantum, label="Quantum")
plt.xlabel("Number of Policies (N)")
plt.ylabel("Mean normalized r")
plt.title("Mean Score vs N")
plt.legend()
plt.show()


# ---- Runtime vs N ----
plt.figure()
plt.plot(Ns, runtimes_classical, label="Classical")
plt.plot(Ns, runtimes_quantum, label="Quantum")
plt.xlabel("Number of Policies (N)")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime Comparison")
plt.legend()
plt.show()

# Avoid zero runtimes
eps = 1e-12
runtimes_classical = np.array(runtimes_classical) + eps
runtimes_quantum = np.array(runtimes_quantum) + eps

logN = np.log(Ns)
logT_classical = np.log(runtimes_classical)
logT_quantum = np.log(runtimes_quantum)

# Linear regression slope
slope_classical = np.polyfit(logN, logT_classical, 1)[0]
slope_quantum = np.polyfit(logN, logT_quantum, 1)[0]

print("Estimated complexity:")
print(f"Classical ≈ O(N^{slope_classical:.2f})")
print(f"Quantum   ≈ O(N^{slope_quantum:.2f})")