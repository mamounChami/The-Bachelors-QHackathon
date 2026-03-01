# Reinsurance Allocation Optimization

We study the **Reinsurance Allocation Optimization** problem.

An insurer must decide which reinsurance contracts to purchase in order to reduce exposure to catastrophic losses while respecting a strict budget constraint.

This project reformulates the problem as a **Constrained Binary Optimization (CBO)** problem, then converts it into a **Quadratic Unconstrained Binary Optimization (QUBO)** model to solve it using both classical algorithms and quantum-inspired methods (QAOA).

---

# Problem Description

The insurer faces multiple potential catastrophic loss scenarios, each associated with a quantified financial loss.

To mitigate extreme tail risks, the company can purchase reinsurance contracts.

Each contract:

- Reduces exposure to large losses  
- Has a fixed premium cost  
- Is either purchased or not (binary decision)  

Because the insurer operates under a limited reinsurance budget, it cannot purchase all contracts. Therefore, it must select an optimal subset that maximizes expected risk reduction while respecting the budget constraint.

We assume:

- Binary decision variables  
- Linear expected value aggregation  
- Budget constraint enforced via penalty in the QUBO formulation  

---

# Decision Variables

We define binary decision variables:

xᵢ ∈ {0,1}

- xᵢ = 1 → contract i is purchased  
- xᵢ = 0 → contract i is not purchased  

If there are **N contracts**, the solution vector is:

x ∈ {0,1}ᴺ

In the quantum formulation, the number of qubits required is equal to **N**.

---

# Risk Assessment & Dataset

We use the **Natural Disasters Emergency Events Database**.

Each record contains:

- Country  
- Year  
- Number of disaster occurrences  
- Total loss (inflation-adjusted USD)  

We filter the dataset by:

- European countries  
- Weather-related disasters  
- Loss per event between 100M and 500M USD  

From the filtered dataset:

1. We compute a histogram of event losses  
2. Convert it into a probability density function (PDF)  
3. Estimate the annual expected frequency of catastrophic events  

We assume only a fraction of total events impact our insurer portfolio, leading to an adjusted PDF used in expected loss calculations.

---

# Modeling Reinsurance Policies

Each policy is modeled using:

- Threshold Tᵢ  
- Coverage proportion pᵢ  
- Premium cost cᵢ  

The payout behaves as follows:

- If loss < Tᵢ → full coverage  
- If loss ≥ Tᵢ → proportional coverage  
- Outside [100M, 500M] → no payout  

The expected payout for policy i is:

rᵢ = ∫ fᵢ(x) · p(x) dx

Where:

- fᵢ(x) = payout function  
- p(x) = probability density function  

The resulting vector:

r = (r₁, r₂, ..., rᴺ)

represents the expected risk reduction of each policy.

---

# Constrained Binary Optimization (CBO)

The insurer solves:

max rᵀx  

subject to:

cᵀx ≤ B  

Where:

- r = expected payout vector  
- c = premium cost vector  
- B = total budget  

This is a 0-1 knapsack-type constrained optimization problem.

---

# From CBO to QUBO

Quantum algorithms like QAOA require an unconstrained formulation.

We convert the constrained problem by introducing a quadratic penalty:

min −rᵀx + λ (cᵀx − B)²

Where:

- λ = penalty coefficient enforcing the budget constraint  

This produces the QUBO form:

E(x) = xᵀ Q x

---

# QUBO → Ising Mapping

To run QAOA, we convert binary variables:

xᵢ = (1 − zᵢ) / 2  

Where:

zᵢ ∈ {−1, +1}

This yields an Ising Hamiltonian:

H = Σ_{i<j} Jᵢⱼ Zᵢ Zⱼ + Σᵢ hᵢ Zᵢ + offset

Which can be directly implemented in a quantum circuit.

---

# Solvers Implemented

## Classical Solvers

- **Brute Force** (exact, exponential complexity O(2ᴺ))  
- **Greedy heuristic** (O(N log N))  
- **Simulated Annealing** (O(iterations × N))  

## Quantum Solver

- **QAOA (Quantum Approximate Optimization Algorithm)**  

We benchmark:

- Solution quality  
- Runtime scaling  
- Approximation ratio  
- Sensitivity to circuit depth p  

---

# Benchmarking

We compare classical and quantum approaches across increasing values of N.

Expected scaling:

- Brute force → exponential  
- Greedy → ~ O(N log N)  
- Simulated Annealing → ~ O(kN)  
- QAOA → depends on:
  - Circuit depth p  
  - Classical optimizer iterations  
  - Number of measurement shots  

The framework allows direct runtime and solution quality comparison.

# Reinsurance Allocation Optimization

We study the **Reinsurance Allocation Optimization** problem.

An insurer must decide which reinsurance contracts to purchase in order to reduce exposure to catastrophic losses while respecting a strict budget constraint.

This project reformulates the problem as a **Constrained Binary Optimization (CBO)** problem, then converts it into a **Quadratic Unconstrained Binary Optimization (QUBO)** model to solve it using both classical algorithms and quantum-inspired methods (QAOA).

---

# Problem Description

The insurer faces multiple potential catastrophic loss scenarios, each associated with a quantified financial loss.

To mitigate extreme tail risks, the company can purchase reinsurance contracts.

Each contract:

- Reduces exposure to large losses  
- Has a fixed premium cost  
- Is either purchased or not (binary decision)  

Because the insurer operates under a limited reinsurance budget, it cannot purchase all contracts. Therefore, it must select an optimal subset that maximizes expected risk reduction while respecting the budget constraint.

We assume:

- Binary decision variables  
- Linear expected value aggregation  
- Budget constraint enforced via penalty in the QUBO formulation  

---

# Decision Variables

We define binary decision variables:

xᵢ ∈ {0,1}

- xᵢ = 1 → contract i is purchased  
- xᵢ = 0 → contract i is not purchased  

If there are **N contracts**, the solution vector is:

x ∈ {0,1}ᴺ

In the quantum formulation, the number of qubits required is equal to **N**.

---

# Risk Assessment & Dataset

We use the **Natural Disasters Emergency Events Database**.

Each record contains:

- Country  
- Year  
- Number of disaster occurrences  
- Total loss (inflation-adjusted USD)  

We filter the dataset by:

- European countries  
- Weather-related disasters  
- Loss per event between 100M and 500M USD  

From the filtered dataset:

1. We compute a histogram of event losses  
2. Convert it into a probability density function (PDF)  
3. Estimate the annual expected frequency of catastrophic events  

We assume only a fraction of total events impact our insurer portfolio, leading to an adjusted PDF used in expected loss calculations.

---

# Modeling Reinsurance Policies

Each policy is modeled using:

- Threshold Tᵢ  
- Coverage proportion pᵢ  
- Premium cost cᵢ  

The payout behaves as follows:

- If loss < Tᵢ → full coverage  
- If loss ≥ Tᵢ → proportional coverage  
- Outside [100M, 500M] → no payout  

The expected payout for policy i is:

rᵢ = ∫ fᵢ(x) · p(x) dx

Where:

- fᵢ(x) = payout function  
- p(x) = probability density function  

The resulting vector:

r = (r₁, r₂, ..., rᴺ)

represents the expected risk reduction of each policy.

---

# Constrained Binary Optimization (CBO)

The insurer solves:

max rᵀx  

subject to:

cᵀx ≤ B  

Where:

- r = expected payout vector  
- c = premium cost vector  
- B = total budget  

This is a 0-1 knapsack-type constrained optimization problem.

---

# From CBO to QUBO

Quantum algorithms like QAOA require an unconstrained formulation.

We convert the constrained problem by introducing a quadratic penalty:

min −rᵀx + λ (cᵀx − B)²

Where:

- λ = penalty coefficient enforcing the budget constraint  

This produces the QUBO form:

E(x) = xᵀ Q x

---

# QUBO → Ising Mapping

To run QAOA, we convert binary variables:

xᵢ = (1 − zᵢ) / 2  

Where:

zᵢ ∈ {−1, +1}

This yields an Ising Hamiltonian:

H = Σ_{i<j} Jᵢⱼ Zᵢ Zⱼ + Σᵢ hᵢ Zᵢ + offset

Which can be directly implemented in a quantum circuit.

---

# Solvers Implemented

## Classical Solvers

- **Brute Force** (exact, exponential complexity O(2ᴺ))  
- **Greedy heuristic** (O(N log N))  
- **Simulated Annealing** (O(iterations × N))  

## Quantum Solver

- **QAOA (Quantum Approximate Optimization Algorithm)**  

We benchmark:

- Solution quality  
- Runtime scaling  
- Approximation ratio  
- Sensitivity to circuit depth p  

---

# Benchmarking

We compare classical and quantum approaches across increasing values of N.

Expected scaling:

- Brute force → exponential  
- Greedy → ~ O(N log N)  
- Simulated Annealing → ~ O(kN)  
- QAOA → depends on:
  - Circuit depth p  
  - Classical optimizer iterations  
  - Number of measurement shots  

The framework allows direct runtime and solution quality comparison.

---

# Conclusion

We successfully:

- Modeled a real-world insurance risk allocation problem  
- Derived expected risk values from historical catastrophe data  
- Formulated the problem as a QUBO  
- Implemented both classical and quantum-inspired solvers  
- Benchmarked performance and scalability  

This project demonstrates how financial risk management problems can be mapped to quantum-ready optimization frameworks.