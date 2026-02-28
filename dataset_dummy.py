"""
dataset_dummy.py — Tiny ReinsuranceDataset for fast benchmarking
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ReinsuranceDataset:
    names: np.ndarray
    costs: np.ndarray
    risks: np.ndarray
    budget: float

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def total_risk(self) -> float:
        return float(np.sum(self.risks))

    def cost_of(self, bitstring: str) -> float:
        x = np.array([int(b) for b in bitstring], dtype=float)
        return float(self.costs @ x)

    def risk_of(self, bitstring: str) -> float:
        x = np.array([int(b) for b in bitstring], dtype=float)
        return float(self.risks @ x)

    def is_feasible(self, bitstring: str) -> bool:
        return self.cost_of(bitstring) <= self.budget


def generate_dataset(n: int = 8, seed: int = 42, budget_ratio: float = 0.40) -> ReinsuranceDataset:
    rng = np.random.default_rng(seed)

    costs = rng.integers(5, 25, size=n).astype(float)
    risks = (costs * rng.uniform(1.0, 3.0, size=n) + rng.integers(0, 10, size=n)).astype(float)

    names = np.array([f"Protection_{i}" for i in range(n)], dtype=object)
    budget = float(budget_ratio * np.sum(costs))

    return ReinsuranceDataset(
        names=names,
        costs=costs,
        risks=risks,
        budget=budget,
    )


def print_dataset(data: ReinsuranceDataset) -> None:
    print("\nDataset instance")
    print("-" * 60)
    print(f"n={data.n}   budget={data.budget:.2f}   total_risk={data.total_risk:.2f}")
    print("-" * 60)
    print(f"{'i':>2}  {'name':<15}  {'cost':>8}  {'risk':>8}")
    for i in range(data.n):
        print(f"{i:>2}  {data.names[i]:<15}  {data.costs[i]:>8.2f}  {data.risks[i]:>8.2f}")
    print("-" * 60)