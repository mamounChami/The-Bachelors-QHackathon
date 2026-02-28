"""
benchmark.py — Benchmarking Engines on Natural Disasters Dataset
===============================================================

This script benchmarks multiple "engines" (implementations) on the same
data operations and generates plots in ./plots/.

It works out-of-the-box with a default Pandas baseline engine, and
can optionally benchmark your custom engines if you add them in Engines/.

Usage
-----
    python BenchMarking/benchmark.py

Expected dataset
----------------
DataSet/natural_disasters_cleaned.csv
"""

import os
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ----------------------------
# Paths
# ----------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
DATA_PATH = os.path.join(ROOT_DIR, "DataSet", "natural_disasters_cleaned.csv")

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ----------------------------
# Plot Style (dark)
# ----------------------------
COLORS = {
    "PandasBaseline": "#4C9BE8",
    "EngineA": "#2A9D8F",
    "EngineB": "#F4A261",
    "EngineC": "#E76F51",
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


# ----------------------------
# Benchmark result container
# ----------------------------
@dataclass
class BenchResult:
    engine: str
    task: str
    n_rows: int
    runtime_s: float
    metric_value: float  # generic "quality" metric (e.g., checksum)
    metric_name: str     # label for plots


# ----------------------------
# Helper: stable numeric conversion
# ----------------------------
def _to_numeric_series(s: pd.Series) -> pd.Series:
    """
    Convert messy numeric columns to numeric.

    Handles:
      - empty strings
      - commas as decimal separators (e.g. "2,84")
      - non-numeric -> NaN
    """
    if s.dtype == object:
        s = s.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
        s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            f"Make sure you have DataSet/natural_disasters_cleaned.csv"
        )
    df = pd.read_csv(path)

    # Normalize column names a bit (strip BOM etc.)
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    # Ensure common columns exist
    needed = ["Year", "Country", "Disaster Type", "Total Deaths", "Total Affected"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}\nColumns: {list(df.columns)}")

    # Convert key numeric columns
    df["Year"] = _to_numeric_series(df["Year"]).astype("Int64")
    df["Total Deaths"] = _to_numeric_series(df["Total Deaths"])
    df["Total Affected"] = _to_numeric_series(df["Total Affected"])

    # Fill NaNs with 0 for aggregations
    df["Total Deaths"] = df["Total Deaths"].fillna(0.0)
    df["Total Affected"] = df["Total Affected"].fillna(0.0)

    return df

# ----------------------------
# Benchmark Tasks (operations)
# ----------------------------
def task_aggregate_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by Country and compute totals.
    Returns a small DataFrame.
    """
    out = (
        df.groupby("Country", as_index=False)[["Total Deaths", "Total Affected"]]
          .sum()
          .sort_values("Total Deaths", ascending=False)
          .head(50)
          .reset_index(drop=True)
    )
    return out


def task_aggregate_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by Year and compute totals.
    """
    out = (
        df.groupby("Year", as_index=False)[["Total Deaths", "Total Affected"]]
          .sum()
          .sort_values("Year", ascending=True)
          .reset_index(drop=True)
    )
    return out


def task_filter_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter post-1950 and rank by deaths.
    """
    dff = df[df["Year"] >= 1950]
    out = (
        dff[["Year", "Country", "Disaster Type", "Total Deaths", "Total Affected"]]
          .sort_values(["Total Deaths", "Total Affected"], ascending=[False, False])
          .head(200)
          .reset_index(drop=True)
    )
    return out


def task_top_disaster_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate by Disaster Type.
    """
    out = (
        df.groupby("Disaster Type", as_index=False)[["Total Deaths", "Total Affected"]]
          .sum()
          .sort_values("Total Deaths", ascending=False)
          .reset_index(drop=True)
    )
    return out


# A "quality metric": checksum of numeric output to ensure engines return same result
def checksum_df(d: pd.DataFrame) -> float:
    # Sum of all numeric values (stable-ish)
    numeric = d.select_dtypes(include=[np.number])
    return float(np.round(numeric.to_numpy(dtype=np.float64).sum(), 6))


# ----------------------------
# Engine interface
# ----------------------------
class Engine:
    """
    Simple engine interface.

    Implement:
      - name (str)
      - run(task_fn, df) -> pd.DataFrame
    """
    name: str

    def run(self, task_fn: Callable[[pd.DataFrame], pd.DataFrame], df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class PandasBaselineEngine(Engine):
    name = "PandasBaseline"

    def run(self, task_fn: Callable[[pd.DataFrame], pd.DataFrame], df: pd.DataFrame) -> pd.DataFrame:
        return task_fn(df)


def discover_custom_engines() -> List[Engine]:
    """
    Optional: import your engines from Engines/ if you add them.

    Convention:
      Engines/your_engine.py must expose a function:
         build_engine() -> Engine
    """
    engines: List[Engine] = []
    engines_dir = os.path.join(ROOT_DIR, "Engines")
    if not os.path.isdir(engines_dir):
        return engines

    for fname in os.listdir(engines_dir):
        if not fname.endswith(".py"):
            continue
        if fname.startswith("__") or fname.startswith("dummy"):
            continue

        mod_name = fname[:-3]
        module_path = f"Engines.{mod_name}"

        try:
            mod = __import__(module_path, fromlist=["build_engine"])
            if hasattr(mod, "build_engine"):
                eng = mod.build_engine()
                if hasattr(eng, "run") and hasattr(eng, "name"):
                    engines.append(eng)
        except Exception as e:
            print(f"  ! Could not load engine from {fname}: {e}")

    return engines


# ----------------------------
# Benchmark runner
# ----------------------------
def time_one(engine: Engine, task_name: str, task_fn: Callable[[pd.DataFrame], pd.DataFrame], df: pd.DataFrame) -> BenchResult:
    t0 = time.perf_counter()
    out = engine.run(task_fn, df)
    t1 = time.perf_counter()

    # metric = checksum to compare outputs
    metric = checksum_df(out)
    return BenchResult(
        engine=engine.name,
        task=task_name,
        n_rows=len(df),
        runtime_s=(t1 - t0),
        metric_value=metric,
        metric_name="checksum(sum of numeric outputs)"
    )


def run_benchmark(
    df: pd.DataFrame,
    engines: List[Engine],
    tasks: List[Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]],
    sizes: Optional[List[int]] = None,
    repeats: int = 3,
) -> List[BenchResult]:
    """
    For each size (number of rows), benchmark each engine on each task.

    repeats: take median runtime over repeats for stability.
    """
    results: List[BenchResult] = []

    if sizes is None:
        sizes = [1000, 3000, 6000, 10000]

    # Safety: if df smaller than largest size, adapt
    max_size = min(max(sizes), len(df))
    sizes = [s for s in sizes if s <= max_size]
    if not sizes:
        sizes = [len(df)]

    for n in sizes:
        dfn = df.head(n).copy()

        print(f"\n{'='*60}")
        print(f"BENCHMARK SIZE: n_rows = {n}")
        print(f"{'='*60}")

        for task_name, task_fn in tasks:
            print(f"\nTask: {task_name}")
            baseline_checksum = None

            for eng in engines:
                runtimes = []
                metric = None

                for _ in range(repeats):
                    r = time_one(eng, task_name, task_fn, dfn)
                    runtimes.append(r.runtime_s)
                    metric = r.metric_value

                med = float(np.median(runtimes))

                if baseline_checksum is None:
                    baseline_checksum = metric

                # record one result with median runtime
                results.append(BenchResult(
                    engine=eng.name,
                    task=task_name,
                    n_rows=n,
                    runtime_s=med,
                    metric_value=metric,
                    metric_name=r.metric_name
                ))

                ok = "OK" if np.isclose(metric, baseline_checksum) else "DIFF"
                print(f"  {eng.name:16s}  time={med*1000:9.3f} ms   metric={metric:.3f}   [{ok}]")

    return results

# ----------------------------
# Plotting
# ----------------------------
def plot_runtime_bars(results: List[BenchResult]) -> None:
    """
    Bar chart per task (for the largest n_rows).
    """
    _style()

    # pick largest n
    max_n = max(r.n_rows for r in results)
    subset = [r for r in results if r.n_rows == max_n]

    tasks = sorted(set(r.task for r in subset))
    engines = sorted(set(r.engine for r in subset))

    for task in tasks:
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.suptitle(f"Runtime Comparison — {task} (n_rows={max_n})", fontsize=14)

        rt = {r.engine: r.runtime_s * 1000 for r in subset if r.task == task}  # ms
        names = [e for e in engines if e in rt]
        vals = [rt[e] for e in names]
        cols = [COLORS.get(e, "#888888") for e in names]

        bars = ax.bar(names, vals, color=cols, alpha=0.85, edgecolor="#888", linewidth=0.8)
        ax.set_ylabel("Runtime (ms) — log scale")
        ax.set_yscale("log")
        ax.grid(axis="y")

        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v * 1.05,
                f"{v:.2f} ms",
                ha="center",
                va="bottom",
                fontsize=9,
                color="white",
            )

        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, f"runtime_{task.replace(' ', '_').lower()}.png")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved: {out}")


def plot_scaling_lines(results: List[BenchResult]) -> None:
    """
    Scaling plot: runtime vs n_rows for each engine, one plot per task.
    """
    _style()
    tasks = sorted(set(r.task for r in results))
    engines = sorted(set(r.engine for r in results))
    sizes = sorted(set(r.n_rows for r in results))

    for task in tasks:
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.suptitle(f"Scaling — {task}", fontsize=14)

        for eng in engines:
            ys = []
            xs = []
            for n in sizes:
                rs = [r for r in results if r.task == task and r.engine == eng and r.n_rows == n]
                if not rs:
                    continue
                xs.append(n)
                ys.append(rs[0].runtime_s * 1000)  # ms

            if xs:
                ax.plot(xs, ys, marker="o", linewidth=2, label=eng)

        ax.set_xlabel("Number of rows (n_rows)")
        ax.set_ylabel("Runtime (ms) — log scale")
        ax.set_yscale("log")
        ax.grid()
        ax.legend()

        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, f"scaling_{task.replace(' ', '_').lower()}.png")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved: {out}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  NATURAL DISASTERS — ENGINE BENCHMARK")
    print("=" * 60)
    print(f"Dataset: {DATA_PATH}")
    print(f"Plots dir: {PLOTS_DIR}")
    print()

    df = load_dataset(DATA_PATH)

    # tasks to run
    tasks = [
        ("Aggregate by Country (Top 50)", task_aggregate_by_country),
        ("Aggregate by Year", task_aggregate_by_year),
        ("Filter (>=1950) + Rank Top 200", task_filter_and_rank),
        ("Aggregate by Disaster Type", task_top_disaster_types),
    ]

    # engines
    engines: List[Engine] = [PandasBaselineEngine()]
    engines += discover_custom_engines()

    print(f"Engines detected: {[e.name for e in engines]}")
    print(f"Rows in dataset: {len(df)}")
    print()

    # run benchmark
    results = run_benchmark(
        df=df,
        engines=engines,
        tasks=tasks,
        sizes=[1000, 3000, 6000, 10000],
        repeats=3,
    )

    # plots
    plot_runtime_bars(results)
    plot_scaling_lines(results)

    print("\n" + "=" * 60)
    print(f"Done. Plots saved to: {PLOTS_DIR}")
    print("=" * 60 + "\n")