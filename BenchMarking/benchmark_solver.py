"""
benchmark_solvers.py — Classical vs QAOA Benchmark (Dummy Dataset)
==================================================================

Runs 4 experiments and generates plots in BenchMarking/plots_solvers/ :

  1) solution_quality.png     — risk reduction + approx ratio
  2) runtime_comparison.png   — runtime comparison (log scale)
  3) scaling_experiment.png   — approx ratio + runtime vs n
  4) qaoa_depth_experiment.png — QAOA approx ratio + runtime vs p

Usage:
  python BenchMarking/benchmark_solvers.py
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ── IMPORTANT: use dummy dataset for speed ────────────────────────────────
from dataset_dummy import generate_dataset, print_dataset

from classical_solver import (
    brute_force, simulated_annealing, greedy,
    print_comparison, approx_ratio,
)
from quantum_solver import run_qaoa


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots_solvers")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
COLORS = {
    "BruteForce"        : "#4C9BE8",
    "SimulatedAnnealing": "#F4A261",
    "Greedy"            : "#2A9D8F",
    "QAOA"              : "#E76F51",
}

def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor"  : "#0F1117",
        "axes.facecolor"    : "#1A1D2A",
        "axes.edgecolor"    : "#444455",
        "axes.labelcolor"   : "#CCCCDD",
        "axes.titlecolor"   : "#EEEEFF",
        "xtick.color"       : "#AAAACC",
        "ytick.color"       : "#AAAACC",
        "grid.color"        : "#2A2D3A",
        "grid.linestyle"    : "--",
        "grid.alpha"        : 0.7,
        "text.color"        : "#EEEEFF",
        "font.family"       : "DejaVu Sans",
        "font.size"         : 11,
        "axes.titlesize"    : 13,
        "axes.labelsize"    : 11,
        "legend.facecolor"  : "#1A1D2A",
        "legend.edgecolor"  : "#444455",
        "legend.fontsize"   : 10,
        "figure.dpi"        : 150,
    })


# ---------------------------------------------------------------------------
# Experiment 1 — Solution quality on a fixed instance
# ---------------------------------------------------------------------------
def experiment_solution_quality(n: int = 10, seed: int = 42, budget_ratio: float = 0.40):
    print(f"\n{'='*70}")
    print(f"EXPERIMENT 1 — Solution Quality (n={n}, seed={seed}, budget={budget_ratio:.2f})")
    print(f"{'='*70}")

    data = generate_dataset(n=n, seed=seed, budget_ratio=budget_ratio)
    print_dataset(data)

    # Brute force only feasible for small n
    bf = brute_force(data) if n <= 20 else None
    sa = simulated_annealing(data, seed=seed, max_iter=6000)  # keep fast
    gr = greedy(data)

    # QAOA fast params
    qa = run_qaoa(data, p=1, shots=1024, verbose=False)

    results = {}
    if bf is not None:
        results["BruteForce"] = bf
    results["SimulatedAnnealing"] = sa
    results["Greedy"] = gr
    results["QAOA(p=1)"] = qa

    print_comparison(list(results.values()), data)
    return results, data


def plot_solution_quality(results: dict, data) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Solution Quality (n={data.n}, Budget={data.budget:.1f})",
        fontsize=14, y=1.02
    )

    names = list(results.keys())
    colors = []
    for nm in names:
        if nm == "BruteForce":
            colors.append(COLORS["BruteForce"])
        elif nm == "SimulatedAnnealing":
            colors.append(COLORS["SimulatedAnnealing"])
        elif nm == "Greedy":
            colors.append(COLORS["Greedy"])
        else:
            colors.append(COLORS["QAOA"])

    # Left: risk reduction
    ax = axes[0]
    risks = [results[n].risk_reduction for n in names]

    # optimal line: brute-force if present, else best feasible among results
    feasible_risks = [results[n].risk_reduction for n in names if results[n].feasible]
    best = max(feasible_risks) if feasible_risks else 0.0

    bars = ax.bar(names, risks, color=colors, alpha=0.85, edgecolor="#888", linewidth=0.8)
    ax.axhline(best, linestyle="--", linewidth=1.5, label=f"Best feasible = {best:.1f}")
    ax.set_ylabel("Risk Reduction")
    ax.set_title("Risk Reduction by Solver")
    ax.grid(axis="y")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend()

    for bar, nm in zip(bars, names):
        val = results[nm].risk_reduction
        feas = "✓" if results[nm].feasible else "✗"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f} {feas}", ha="center", va="bottom", fontsize=9, color="white")

    # Right: approximation ratio vs optimal
    ax = axes[1]
    if "BruteForce" in results:
        opt = results["BruteForce"].risk_reduction
        opt_label = "Optimal (BruteForce)"
    else:
        opt = best if best > 0 else 1.0
        opt_label = "Best feasible (no BF)"

    ratios = [approx_ratio(results[n], opt) for n in names]
    bars2 = ax.bar(names, [r*100 for r in ratios], color=colors, alpha=0.85, edgecolor="#888", linewidth=0.8)
    ax.axhline(100, linestyle="--", linewidth=1.5, label=opt_label)
    ax.set_ylabel("Approximation Ratio (%)")
    ax.set_title("Approximation Ratio")
    ax.set_ylim(0, 115)
    ax.grid(axis="y")
    ax.legend()

    for bar, r in zip(bars2, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{r*100:.1f}%", ha="center", va="bottom", fontsize=9, color="white")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "solution_quality.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out}")


# ---------------------------------------------------------------------------
# Experiment 2 — Runtime comparison
# ---------------------------------------------------------------------------
def plot_runtime_comparison(results: dict) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Runtime Comparison (log scale)", fontsize=14)

    names = list(results.keys())
    runtimes_ms = [results[n].runtime_s * 1000 for n in names]

    colors = []
    for nm in names:
        if nm == "BruteForce":
            colors.append(COLORS["BruteForce"])
        elif nm == "SimulatedAnnealing":
            colors.append(COLORS["SimulatedAnnealing"])
        elif nm == "Greedy":
            colors.append(COLORS["Greedy"])
        else:
            colors.append(COLORS["QAOA"])

    bars = ax.barh(names, runtimes_ms, color=colors, alpha=0.85, edgecolor="#888", linewidth=0.8)
    ax.set_xlabel("Runtime (ms)")
    ax.set_xscale("log")
    ax.grid(axis="x")
    ax.invert_yaxis()

    for bar, v in zip(bars, runtimes_ms):
        ax.text(v * 1.05, bar.get_y() + bar.get_height()/2, f"{v:.2f} ms",
                va="center", fontsize=9, color="white")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "runtime_comparison.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out}")

    # ---------------------------------------------------------------------------
# Experiment 3 — Scaling vs n
# ---------------------------------------------------------------------------
def experiment_scaling(
    sizes=None,
    seed: int = 42,
    budget_ratio: float = 0.40,
):
    if sizes is None:
        sizes = [4, 5, 6, 7, 8, 10]

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 3 — Scaling (sizes={sizes}, budget={budget_ratio:.2f})")
    print(f"{'='*70}")

    ratios = {"Greedy": [], "SimulatedAnnealing": [], "QAOA(p=1)": []}
    runtimes = {"Greedy": [], "SimulatedAnnealing": [], "QAOA(p=1)": []}
    opt_used = []

    for n in sizes:
        print(f"\n  n = {n}")
        data = generate_dataset(n=n, seed=seed, budget_ratio=budget_ratio)

        # classical
        gr = greedy(data)
        sa = simulated_annealing(data, seed=seed, max_iter=4000)

        # qaoa
        qa = run_qaoa(data, p=1, shots=512, verbose=False)

        # optimal reference if possible
        if n <= 20:
            bf = brute_force(data)
            opt = bf.risk_reduction if bf.risk_reduction > 0 else 1.0
            ref = "BruteForce"
        else:
            feas = [r.risk_reduction for r in [gr, sa, qa] if r.feasible]
            opt = max(feas) if feas else 1.0
            ref = "Best feasible"

        opt_used.append((n, ref))

        ratios["Greedy"].append(approx_ratio(gr, opt))
        ratios["SimulatedAnnealing"].append(approx_ratio(sa, opt))
        ratios["QAOA(p=1)"].append(approx_ratio(qa, opt))

        runtimes["Greedy"].append(gr.runtime_s)
        runtimes["SimulatedAnnealing"].append(sa.runtime_s)
        runtimes["QAOA(p=1)"].append(qa.runtime_s)

        print(f"    Reference: {ref} (opt≈{opt:.2f})")
        print(f"    Greedy: risk={gr.risk_reduction:.2f}, time={gr.runtime_s*1000:.2f}ms, ratio={ratios['Greedy'][-1]:.3f}")
        print(f"    SA    : risk={sa.risk_reduction:.2f}, time={sa.runtime_s*1000:.2f}ms, ratio={ratios['SimulatedAnnealing'][-1]:.3f}")
        print(f"    QAOA  : risk={qa.risk_reduction:.2f}, time={qa.runtime_s:.2f}s,  ratio={ratios['QAOA(p=1)'][-1]:.3f}")

    return sizes, ratios, runtimes, opt_used


def plot_scaling(sizes, ratios, runtimes) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Scaling Experiment — Performance vs Problem Size n", fontsize=14)

    # Left: approximation ratio
    ax = axes[0]
    ax.plot(sizes, [r*100 for r in ratios["Greedy"]],
            "s-", color=COLORS["Greedy"], label="Greedy", linewidth=2, markersize=6)
    ax.plot(sizes, [r*100 for r in ratios["SimulatedAnnealing"]],
            "o-", color=COLORS["SimulatedAnnealing"], label="Simulated Annealing", linewidth=2, markersize=6)
    ax.plot(sizes, [r*100 for r in ratios["QAOA(p=1)"]],
            "^-", color=COLORS["QAOA"], label="QAOA (p=1)", linewidth=2, markersize=6)

    ax.set_xlabel("Problem size n")
    ax.set_ylabel("Approximation Ratio (%)")
    ax.set_title("Approximation Ratio vs n")
    ax.set_ylim(0, 115)
    ax.grid()
    ax.legend()

    # Right: runtime
    ax = axes[1]
    ax.plot(sizes, [t*1000 for t in runtimes["Greedy"]],
            "s-", color=COLORS["Greedy"], label="Greedy", linewidth=2, markersize=6)
    ax.plot(sizes, [t*1000 for t in runtimes["SimulatedAnnealing"]],
            "o-", color=COLORS["SimulatedAnnealing"], label="Simulated Annealing", linewidth=2, markersize=6)
    ax.plot(sizes, [t*1000 for t in runtimes["QAOA(p=1)"]],
            "^-", color=COLORS["QAOA"], label="QAOA (p=1)", linewidth=2, markersize=6)

    ax.set_xlabel("Problem size n")
    ax.set_ylabel("Runtime (ms) — log scale")
    ax.set_title("Runtime vs n")
    ax.set_yscale("log")
    ax.grid()
    ax.legend()

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "scaling_experiment.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out}")


# ---------------------------------------------------------------------------
# Experiment 4 — QAOA depth p
# ---------------------------------------------------------------------------
def experiment_qaoa_depth(
    n: int = 8,
    seed: int = 42,
    budget_ratio: float = 0.40,
    depths=None,
):
    if depths is None:
        depths = [1, 2, 3]

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 4 — QAOA Depth (n={n}, depths={depths})")
    print(f"{'='*70}")

    data = generate_dataset(n=n, seed=seed, budget_ratio=budget_ratio)

    # Reference (optimal) if possible
    if n <= 20:
        bf = brute_force(data)
        opt = bf.risk_reduction if bf.risk_reduction > 0 else 1.0
    else:
        opt = 1.0

    results_by_depth = {}
    for p in depths:
        print(f"\n  QAOA p = {p}")
        qa = run_qaoa(data, p=p, shots=512, verbose=False)
        results_by_depth[p] = qa
        ratio = approx_ratio(qa, opt)
        print(f"    Risk={qa.risk_reduction:.2f} / opt≈{opt:.2f} → ratio={ratio:.3f} ({ratio*100:.1f}%)  time={qa.runtime_s:.2f}s")

    return depths, results_by_depth, opt, n


def plot_qaoa_depth(depths, results_by_depth, optimal_risk, n: int) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"QAOA Performance vs Circuit Depth p (n={n})", fontsize=14)

    ratios = [approx_ratio(results_by_depth[p], optimal_risk) * 100 for p in depths]
    runtimes = [results_by_depth[p].runtime_s for p in depths]

    # Left: ratio
    ax = axes[0]
    ax.plot(depths, ratios, "^-", color=COLORS["QAOA"], linewidth=2.5, markersize=8, label="QAOA")
    ax.axhline(100, linestyle="--", linewidth=1.5, label=f"Reference (opt≈{optimal_risk:.1f})")
    ax.set_xlabel("QAOA Depth p")
    ax.set_ylabel("Approximation Ratio (%)")
    ax.set_title("Approximation Ratio vs p")
    ax.set_xticks(depths)
    ax.set_ylim(0, 115)
    ax.grid()
    ax.legend()

    for x, y in zip(depths, ratios):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, color="white")

    # Right: runtime
    ax = axes[1]
    ax.plot(depths, runtimes, "o-", linewidth=2.5, markersize=8, label="Wall-clock time (s)")
    ax.set_xlabel("QAOA Depth p")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime vs p")
    ax.set_xticks(depths)
    ax.grid()
    ax.legend()

    for x, y in zip(depths, runtimes):
        ax.annotate(f"{y:.1f}s", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, color="white")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "qaoa_depth_experiment.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("  REINSURANCE OPTIMIZATION — SOLVER BENCHMARK (DUMMY DATASET)")
    print("=" * 70)
    print(f"Plots dir: {PLOTS_DIR}\n")

    # ── Exp 1 + 2
    results, data = experiment_solution_quality(n=10, seed=42, budget_ratio=0.40)
    plot_solution_quality(results, data)
    plot_runtime_comparison(results)

    # ── Exp 3
    sizes, ratios, runtimes, opt_used = experiment_scaling(sizes=[4, 5, 6, 7, 8, 10], seed=42, budget_ratio=0.40)
    plot_scaling(sizes, ratios, runtimes)

    # ── Exp 4
    depths, depth_results, opt_risk, n = experiment_qaoa_depth(n=8, seed=42, budget_ratio=0.40, depths=[1, 2, 3])
    plot_qaoa_depth(depths, depth_results, opt_risk, n=n)

    print("\n" + "=" * 70)
    print(f"Done. Plots saved to: {PLOTS_DIR}")
    print("Files: solution_quality.png, runtime_comparison.png, scaling_experiment.png, qaoa_depth_experiment.png")
    print("=" * 70 + "\n")