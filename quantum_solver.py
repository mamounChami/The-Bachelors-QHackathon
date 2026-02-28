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

from dataset import ReinsuranceDataset, generate_dataset, print_dataset
from classical_solver import SolverResult

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Step 1 – Build the QUBO matrix
# ---------------------------------------------------------------------------

def build_qubo(data: ReinsuranceDataset, lam: float = 5.0) -> np.ndarray:
    """
    Construct the QUBO matrix Q for the reinsurance problem.

    The QUBO energy is defined as:

        E(x) = x^T Q x          (we MINIMISE this)

    For a binary vector x ∈ {0,1}^n (note: xᵢ² = xᵢ for binary variables),
    and using the Hamiltonian:

        H(x) = -Σ r_i x_i  +  λ (Σ c_i x_i − B)²

    Expanding the penalty term:
        λ (Σ c_i x_i − B)²
      = λ [ Σᵢ cᵢ² xᵢ  +  2 Σ_{i<j} cᵢcⱼ xᵢxⱼ  −  2B Σᵢ cᵢ xᵢ  +  B² ]

    Combining with the objective (B² is a constant, ignored in QUBO):

        Q_{ii} = -r_i  +  λ cᵢ²  −  2λ B cᵢ       (diagonal — linear terms)
        Q_{ij} = 2λ cᵢ cⱼ    for i < j             (upper triangle — quadratic terms)

    By convention Q is upper-triangular so that x^T Q x covers all terms.

    Parameters
    ----------
    data : ReinsuranceDataset
    lam  : penalty strength λ for the budget constraint

    Returns
    -------
    Q : np.ndarray of shape (n, n)
    """
    n = data.n
    c = data.costs
    r = data.risks
    B = data.budget

    Q = np.zeros((n, n))

    # Diagonal terms: linear part of the QUBO
    for i in range(n):
        Q[i, i] = -r[i] + lam * c[i] ** 2 - 2 * lam * B * c[i]

    # Off-diagonal terms: quadratic interaction between protections i and j
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = 2 * lam * c[i] * c[j]

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

def run_qaoa(
    data: ReinsuranceDataset,
    p: int    = 1,
    lam: float = None,
    shots: int = 4096,
    verbose: bool = True,
) -> SolverResult:
    """
    Full QAOA pipeline for the reinsurance allocation problem.

    1. Build QUBO matrix (Section 2a)
    2. Convert to Ising H_C = Σ J_{ij} ZZ + Σ hᵢ Zᵢ (Section 2b)
    3. Build & optimise the QAOA circuit (Sections 3–4)
    4. Sample the optimised circuit on Aer (Section 5)
    5. Extract most-probable feasible bitstring (Section 6)

    Parameters
    ----------
    data    : reinsurance problem instance
    p       : QAOA circuit depth (number of alternating layers)
              Higher p → better approximation, deeper circuit
    lam     : penalty coefficient λ for budget constraint
              If None, auto-set to max(r_i) * n / B (heuristic)
    shots   : number of measurement shots for sampling
    verbose : print progress if True

    Returns
    -------
    SolverResult — decoded insurance decision from the quantum circuit.
    """
    t0 = time.perf_counter()
    n  = data.n

    # ── Auto-tune λ if not provided ───────────────────────────────────────
    # λ must be large enough so that any infeasible solution has higher
    # energy than the best feasible one.  A safe heuristic:  λ ≥ max(r_i)/min(c_i)
    if lam is None:
        lam = float(np.max(data.risks)) / float(np.min(data.costs)) + 1.0
        if verbose:
            print(f"  Auto-selected λ = {lam:.3f}")

    # ── Step 1: Build QUBO ────────────────────────────────────────────────
    if verbose:
        print(f"  Building QUBO (n={n}, λ={lam:.3f}) ...")
    Q = build_qubo(data, lam=lam)

    # ── Step 2: QUBO → Ising ──────────────────────────────────────────────
    J, h, offset = qubo_to_ising(Q)
    if verbose:
        print(f"  Ising Hamiltonian: {int(np.sum(J != 0)/2)} ZZ-couplings, "
              f"{int(np.sum(h != 0))} local fields, offset={offset:.3f}")

    # ── Step 3+4: QAOA parameter optimisation ─────────────────────────────
    # Objective function: expected energy ⟨ψ(β,γ)|H_C|ψ(β,γ)⟩
    # We approximate this by evaluating the QUBO energy on all sampled bitstrings

    if verbose:
        print(f"  Optimising QAOA (p={p}) with COBYLA ...")

    # Use Aer simulator for fast statevector simulation
    backend = AerSimulator(method="statevector")

    def qaoa_energy(params: np.ndarray) -> float:
        """
        Compute the expected QUBO energy for given QAOA parameters.
        Used as the objective for the classical COBYLA optimizer.
        """
        gamma_k = params[:p]
        beta_k  = params[p:]

        qc = build_qaoa_circuit(J, h, p, gamma_k, beta_k)

        # Transpile and run on simulator
        from qiskit import transpile
        qc_t    = transpile(qc, backend, optimization_level=0)
        job     = backend.run(qc_t, shots=512)  # lower shots for speed during optimisation
        counts  = job.result().get_counts()

        # Compute weighted average QUBO energy over measurement outcomes
        total_energy = 0.0
        total_shots  = sum(counts.values())

        for bitstring, count in counts.items():
            # Qiskit returns bitstrings with qubit 0 on the RIGHT
            # Reverse so index 0 corresponds to protection 0
            bs_ordered = bitstring[::-1]
            x = np.array([int(b) for b in bs_ordered], dtype=float)

            # Evaluate QUBO energy: E = x^T Q x
            energy      = float(x @ Q @ x)
            total_energy += (count / total_shots) * energy

        return total_energy   # we minimise this

    # Initial parameters: small random values in [0, π]
    rng0   = np.random.default_rng(0)
    x0     = rng0.uniform(0, np.pi, size=2 * p)

    # Classical optimisation loop (COBYLA — derivative-free)
    result = minimize(
        qaoa_energy,
        x0,
        method="COBYLA",
        options={"maxiter": 200, "rhobeg": 0.5},
    )

    opt_params = result.x
    if verbose:
        print(f"  Optimisation: {result.nfev} function evaluations, "
              f"final energy = {result.fun:.4f}")

    # ── Step 5: Sample the optimised circuit with more shots ──────────────
    if verbose:
        print(f"  Sampling optimised circuit with {shots} shots ...")

    gamma_opt = opt_params[:p]
    beta_opt  = opt_params[p:]
    qc_opt    = build_qaoa_circuit(J, h, p, gamma_opt, beta_opt)

    from qiskit import transpile
    qc_t    = transpile(qc_opt, backend, optimization_level=0)
    job     = backend.run(qc_t, shots=shots)
    counts  = job.result().get_counts()

    # ── Step 6: Decode the most-probable *feasible* bitstring ─────────────
    # Sort by probability (count), pick the most probable feasible solution
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    best_bs   = "0" * n
    best_risk = -1.0

    for bitstring, count in sorted_counts:
        bs_ordered = bitstring[::-1]    # Qiskit: qubit 0 on right → reverse
        if len(bs_ordered) != n:
            continue

        risk = data.risk_of(bs_ordered)

        # Prefer feasible solutions; among feasible, pick max risk reduction
        if data.is_feasible(bs_ordered) and risk > best_risk:
            best_risk = risk
            best_bs   = bs_ordered

    # If no feasible solution found in samples, return the most-probable one
    if best_risk < 0:
        best_bs = sorted_counts[0][0][::-1]
        if verbose:
            print("  ⚠️  No feasible solution found in samples — returning most probable.")

    if verbose:
        top5 = [(bs[::-1], c) for bs, c in sorted_counts[:5]]
        print(f"  Top-5 outcomes (reversed):")
        for bs, c in top5:
            feas = "✓" if data.is_feasible(bs) else "✗"
            print(f"    {bs} — {c:>5} shots  {feas}  risk={data.risk_of(bs):.1f}")

    return SolverResult(
        solver_name    = f"QAOA(p={p})",
        bitstring      = best_bs,
        cost           = data.cost_of(best_bs),
        risk_reduction = data.risk_of(best_bs),
        runtime_s      = time.perf_counter() - t0,
        feasible       = data.is_feasible(best_bs),
    )


# ---------------------------------------------------------------------------
# Circuit inspection helpers
# ---------------------------------------------------------------------------

def circuit_stats(qc: QuantumCircuit) -> dict:
    """Return basic circuit statistics."""
    ops = qc.count_ops()
    return {
        "num_qubits"  : qc.num_qubits,
        "depth"       : qc.depth(),
        "gate_counts" : ops,
        "num_gates"   : sum(ops.values()),
    }


def describe_qubo(Q: np.ndarray, data: ReinsuranceDataset) -> None:
    """Print a readable description of the QUBO matrix."""
    n = Q.shape[0]
    print(f"\n  QUBO matrix ({n}×{n}):")
    print("  Diagonal (linear terms):")
    for i in range(n):
        print(f"    Q[{i},{i}] = {Q[i,i]:+.3f}  ({data.names[i]})")
    print("  Off-diagonal (quadratic terms):")
    for i in range(n):
        for j in range(i+1, n):
            if abs(Q[i,j]) > 1e-6:
                print(f"    Q[{i},{j}] = {Q[i,j]:+.3f}  ({data.names[i]}×{data.names[j]})")


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 68)
    print("  QUANTUM SOLVER (QAOA) — DEMO")
    print("=" * 68)

    # Small problem for fast demo
    data = generate_dataset(n=6, seed=42, budget_ratio=0.40)
    print_dataset(data)

    # Show QUBO
    lam = 3.0
    Q   = build_qubo(data, lam=lam)
    print(f"\nQUBO matrix (λ={lam}):")
    print(np.round(Q, 2))

    # Show Ising
    J, h, offset = qubo_to_ising(Q)
    print(f"\nIsing h (local fields): {np.round(h, 3)}")
    print(f"Ising offset          : {offset:.3f}")

    # Show circuit (p=1, example angles)
    gamma_ex = np.array([0.3])
    beta_ex  = np.array([0.7])
    qc_ex    = build_qaoa_circuit(J, h, p=1, gamma=gamma_ex, beta=beta_ex)
    print(f"\nQAOA circuit (p=1) stats: {circuit_stats(qc_ex)}")
    print(qc_ex.draw(output="text", fold=-1))

    # Run QAOA
    print("\nRunning QAOA optimisation (p=1) ...")
    result = run_qaoa(data, p=1, lam=lam, shots=2048, verbose=True)
    print(f"\n{result}")