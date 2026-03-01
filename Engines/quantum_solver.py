"""
quantum_solver.py — QAOA-based Solver for Reinsurance Allocation
=================================================================
Implements the full quantum pipeline:

  Step 1 : Build the QUBO matrix from the reinsurance problem
  Step 2 : Convert QUBO → Ising Hamiltonian (spin variables ±1)
  Step 3 : Construct the parameterised QAOA circuit
  Step 4 : Optimise QAOA parameters with a classical optimizer (COBYLA)
  Step 5 : Sample the optimal circuit on the Aer simulator
  Step 6 : Decode the most-probable bitstring into an insurance decision

Dependencies
------------
    pip install qiskit qiskit-aer numpy scipy

Background
----------
The Reinsurance Allocation problem is an instance of the binary knapsack:

    maximise   Σ r_i x_i          (maximise risk reduction)
    subject to Σ c_i x_i ≤ B     (budget constraint)
               x_i ∈ {0, 1}

To apply QAOA we must convert this into an *unconstrained* minimisation
problem — a QUBO — by absorbing the constraint as a penalty term:

    H(x) = - Σ r_i x_i  +  λ (Σ c_i x_i − B)²

The scalar λ > 0 must be large enough that any violation of the budget
constraint gives a higher energy than the best feasible solution.

QUBO → Ising conversion:
    Substitute  x_i = (1 − z_i) / 2   with  z_i ∈ {−1, +1}
    Gives a cost Hamiltonian:  H_C = Σ_{i<j} J_{ij} Z_i Z_j + Σ_i h_i Z_i + const

QAOA ansatz (depth p):
    |ψ(β,γ)⟩ = [∏_{k=1}^{p} e^{-iβ_k H_B} e^{-iγ_k H_C}] |+⟩^⊗n

where H_B = Σ_i X_i is the transverse-field mixer Hamiltonian.
"""

import warnings
import time
import numpy as np
from itertools import product
from typing import Tuple

# ── Qiskit imports ────────────────────────────────────────────────────────
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from scipy.optimize import minimize

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Step 1 – Build the QUBO matrix
# ---------------------------------------------------------------------------

def build_qubo(r, premiums, total_budget, lam: float = 1.0) -> np.ndarray:
    """
    Build the upper-triangular QUBO matrix Q such that:

        E(x) = x^T Q x     (MINIMIZE)

    corresponding to the penalized objective:

        H(x) = - sum_i r_i x_i  +  lam * (sum_i c_i x_i - B)^2

    where:
      r_i = profit/score for item i (we want to maximize sum r_i x_i)
      c_i = premiums/costs for item i
      B   = total_budget

    Parameters
    ----------
    r : array-like, shape (N,) or (1,N)
        Scores/profits to maximize.
    premiums : array-like, shape (N,)
        Costs of selecting each item.
    total_budget : float
        Budget constraint B.
    lam : float
        Penalty strength.

    Returns
    -------
    Q : np.ndarray, shape (N, N)
        Upper-triangular QUBO matrix.
    """
    r = np.asarray(r, dtype=float).reshape(-1)               # (N,)
    c = np.asarray(premiums, dtype=float).reshape(-1)        # (N,)
    B = float(total_budget)

    n = r.size
    Q = np.zeros((n, n), dtype=float)

    # Diagonal: -r_i + lam*c_i^2 - 2*lam*B*c_i
    Q[np.diag_indices(n)] = -r + lam * (c ** 2) - 2.0 * lam * B * c

    # Upper triangle off-diagonal: 2*lam*c_i*c_j for i<j
    # Vectorized version:
    for i in range(n):
        Q[i, i+1:] = 2.0 * lam * c[i] * c[i+1:]

    return Q


# ---------------------------------------------------------------------------
# Step 2 – Convert QUBO → Ising Hamiltonian
# ---------------------------------------------------------------------------

def qubo_to_ising(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Map QUBO minimisation to Ising minimisation via  xᵢ = (1 − zᵢ) / 2.

    The Ising Hamiltonian is:

        H_C = Σ_{i<j} J_{ij} Zᵢ Zⱼ + Σᵢ hᵢ Zᵢ + offset

    This function derives J, h, and the constant energy offset from Q.

    Derivation:
        Substituting xᵢ = (1 − zᵢ)/2 into E(x) = x^T Q x:

        For diagonal term Q_{ii}:
            Q_{ii} xᵢ = Q_{ii}(1 − zᵢ)/2
            → contributes  -Q_{ii}/2  to hᵢ,  Q_{ii}/2  to offset

        For off-diagonal Q_{ij} (i < j):
            Q_{ij} xᵢ xⱼ = Q_{ij}(1 − zᵢ)(1 − zⱼ)/4
            → contributes  Q_{ij}/4  to J_{ij}
            → contributes  -Q_{ij}/4  to hᵢ  and  hⱼ
            → contributes  Q_{ij}/4  to offset

    Parameters
    ----------
    Q : upper-triangular QUBO matrix (n × n)

    Returns
    -------
    J      : symmetric coupling matrix (n × n), J_{ij} for i ≠ j
    h      : local field vector (n,)
    offset : constant energy offset (float)
    """
    n = Q.shape[0]
    J = np.zeros((n, n))
    h = np.zeros(n)
    offset = 0.0

    # Diagonal terms
    for i in range(n):
        h[i]    += -Q[i, i] / 2.0
        offset  +=  Q[i, i] / 2.0

    # Off-diagonal terms
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] +=  Q[i, j] / 4.0
            J[j, i] +=  Q[i, j] / 4.0   # symmetrise
            h[i]    += -Q[i, j] / 4.0
            h[j]    += -Q[i, j] / 4.0
            offset  +=  Q[i, j] / 4.0

    return J, h, offset


# ---------------------------------------------------------------------------
# Step 3 – Build the QAOA circuit
# ---------------------------------------------------------------------------

def build_qaoa_circuit(
    J: np.ndarray,
    h: np.ndarray,
    p: int,
    gamma: np.ndarray,
    beta: np.ndarray,
) -> QuantumCircuit:
    """
    Construct a QAOA circuit of depth p.

    QAOA ansatz:
        |ψ(β,γ)⟩ = [∏_{k=1}^{p} U_B(βₖ) U_C(γₖ)] |+⟩^⊗n

    where:
        U_C(γ) = exp(-iγ H_C)   — cost-layer: encodes the problem
        U_B(β) = exp(-iβ H_B)   — mixer-layer: explores the space

    Decomposed into native gates:

    1. Initialisation:
        Apply H (Hadamard) to each qubit → equal superposition |+⟩^⊗n

    2. Cost unitary U_C(γ):
        For each ZZ coupling J_{ij}:
            CNOT(i→j), Rz(2γ J_{ij}), CNOT(i→j)
        For each local field hᵢ:
            Rz(2γ hᵢ) on qubit i

    3. Mixer unitary U_B(β):
        Rx(2β) on every qubit   [transverse-field flip mixer]

    Parameters
    ----------
    J     : coupling matrix (n × n), symmetric
    h     : local field vector (n,)
    p     : circuit depth (number of QAOA layers)
    gamma : cost-layer angles, shape (p,)
    beta  : mixer-layer angles, shape (p,)

    Returns
    -------
    QuantumCircuit of n qubits with 2p parameters substituted.
    """
    n  = len(h)
    qc = QuantumCircuit(n)

    # ── Initialisation: equal superposition ──────────────────────────────
    qc.h(range(n))

    # ── p QAOA layers ─────────────────────────────────────────────────────
    for k in range(p):

        # Cost unitary U_C(γ_k)
        # ZZ interactions: exp(-i γ J_{ij} Zᵢ Zⱼ)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    qc.cx(i, j)                        # CNOT i→j
                    qc.rz(2 * gamma[k] * J[i, j], j)  # Rz rotation
                    qc.cx(i, j)                        # CNOT i→j (undo)

        # Local field: exp(-i γ hᵢ Zᵢ)
        for i in range(n):
            if abs(h[i]) > 1e-12:
                qc.rz(2 * gamma[k] * h[i], i)

        # Mixer unitary U_B(β_k)
        # exp(-i β H_B) = exp(-i β Σ Xᵢ) = ∏ exp(-i β Xᵢ) = ∏ Rx(2β)
        for i in range(n):
            qc.rx(2 * beta[k], i)

    # Measure all qubits
    qc.measure_all()
    return qc


# ---------------------------------------------------------------------------
# Step 4+5 – Optimise and sample QAOA
# ---------------------------------------------------------------------------

# assumes you already have:
# - build_qubo(r, premiums, total_budget, lam)
# - qubo_to_ising(Q)
# - build_qaoa_circuit(J, h, p, gamma_k, beta_k)

def run_qaoa(
    r,
    premiums,
    total_budget,
    p: int = 1,
    lam: float | None = None,
    shots: int = 4096,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, float, bool]:
    """
    QAOA pipeline for:
        maximize r^T x
        s.t. premiums^T x <= total_budget
    via QUBO minimization.

    Returns:
      best_x (N,) binary vector
      best_score (float) = r^T best_x
      runtime_s (float)
      feasible (bool)
    """

    r = np.asarray(r, dtype=float).reshape(-1)
    c = np.asarray(premiums, dtype=float).reshape(-1)
    B = float(total_budget)
    n = r.size

    if c.size != n:
        raise ValueError("r and premiums must have same length")

    # --- auto λ if not provided ---
    if lam is None:
        lam = float(np.max(np.abs(r))) / float(np.min(c)) + 1.0
        if verbose:
            print(f"  Auto-selected λ = {lam:.3f}")

    # --- Step 1: Build QUBO ---
    if verbose:
        print(f"  Building QUBO (n={n}, λ={lam:.3f}) ...")
    Q = build_qubo(r, c, B, lam=lam)

    # --- Step 2: QUBO -> Ising ---
    J, h, offset = qubo_to_ising(Q)
    if verbose:
        zz = int(np.sum(J != 0) // 2)
        hf = int(np.sum(h != 0))
        print(f"  Ising: {zz} ZZ couplings, {hf} local fields, offset={offset:.3f}")

    # --- Step 3+4: Optimize QAOA params ---
    backend = AerSimulator(method="statevector")

    def qaoa_energy(params: np.ndarray) -> float:
        gamma_k = params[:p]
        beta_k  = params[p:]

        qc = build_qaoa_circuit(J, h, p, gamma_k, beta_k)

        from qiskit import transpile
        qc_t   = transpile(qc, backend, optimization_level=0)
        job    = backend.run(qc_t, shots=512)
        counts = job.result().get_counts()

        total_energy = 0.0
        total_shots  = sum(counts.values())

        for bitstring, count in counts.items():
            bs = bitstring[::-1]  # reverse (qiskit qubit0 on right)
            x = np.fromiter((int(b) for b in bs), dtype=float, count=n)

            energy = float(x @ Q @ x)
            total_energy += (count / total_shots) * energy

        return total_energy

    if verbose:
        print(f"  Optimising QAOA (p={p}) with COBYLA ...")

    rng0 = np.random.default_rng(0)
    x0 = rng0.uniform(0, np.pi, size=2 * p)

    opt = minimize(
        qaoa_energy,
        x0,
        method="COBYLA",
        options={"maxiter": 200, "rhobeg": 0.5},
    )

    if verbose:
        print(f"  Optimisation: nfev={opt.nfev}, final energy={opt.fun:.4f}")

    gamma_opt = opt.x[:p]
    beta_opt  = opt.x[p:]

    # --- Step 5: Sample optimized circuit ---
    if verbose:
        print(f"  Sampling with {shots} shots ...")

    qc_opt = build_qaoa_circuit(J, h, p, gamma_opt, beta_opt)

    from qiskit import transpile
    qc_t   = transpile(qc_opt, backend, optimization_level=0)
    job    = backend.run(qc_t, shots=shots)
    counts = job.result().get_counts()

    # --- Step 6: Decode best feasible bitstring ---
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    best_x = np.zeros(n, dtype=int)
    best_score = -np.inf
    best_feas = False

    def is_feasible(x_vec: np.ndarray) -> bool:
        return float(np.dot(c, x_vec)) <= B

    for bitstring, count in sorted_counts:
        bs = bitstring[::-1]
        if len(bs) != n:
            continue
        x = np.fromiter((int(b) for b in bs), dtype=int, count=n)

        score = float(np.dot(r, x))
        feas = is_feasible(x)

        # Prefer feasible; among feasible maximize score
        if feas and score > best_score:
            best_x = x
            best_score = score
            best_feas = True

    # If no feasible found, return most probable (still give its score)
    if not best_feas and sorted_counts:
        bs = sorted_counts[0][0][::-1]
        best_x = np.fromiter((int(b) for b in bs), dtype=int, count=n)
        best_score = float(np.dot(r, best_x))
        best_feas = is_feasible(best_x)
        if verbose:
            print("  ⚠️ No feasible found in samples — returning most probable.")

    return best_x, best_score, best_feas