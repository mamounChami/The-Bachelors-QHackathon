# Reinsurance Allocation Optimization

We study the **Reinsurance Allocation Optimization** problem.

An insurer must decide which reinsurance contracts to purchase in order to reduce exposure to catastrophic losses while respecting a strict budget constraint.

This project reformulates the problem as a **Constrained Binary Optimization (CBO)** problem, then converts it into a **Quadratic Unconstrained Binary Optimization (QUBO)** model in order to solve it using both classical algorithms and quantum-inspired methods (QAOA).

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

$$ x_i \in \{0,1\} $$

where each variable corresponds to one reinsurance contract.

- \( x_i = 1 \) → contract *i* is purchased  
- \( x_i = 0 \) → contract *i* is not purchased  

If there are **N contracts**, the solution vector is:

\[
x \in \{0,1\}^N
\]

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
- Loss per event between **100M and 500M USD**

From the filtered dataset:

1. We compute a **histogram of event losses**
2. Convert it into a **probability density function (PDF)**
3. Estimate the annual expected frequency of catastrophic events

We assume only a fraction of total events impact our insurer portfolio, leading to an adjusted PDF used in expected loss calculations.

---

# Modeling Reinsurance Policies

Each policy is modeled using:

- A threshold \( T_i \)
- A coverage proportion \( p_i \)
- A premium cost \( c_i \)

The payout function behaves as follows:

- If loss < threshold → full coverage  
- If loss ≥ threshold → proportional coverage  
- Outside [100M, 500M] → no payout  

The expected payout for policy \( i \) is computed as:

\[
r_i = \int_{100M}^{500M} f_i(x) \cdot p(x) \, dx
\]

Where:

- \( f_i(x) \) = payout function of policy i  
- \( p(x) \) = probability density function  

The resulting vector:

\[
r = (r_1, r_2, ..., r_N)
\]

represents the **expected risk reduction** for each policy.

---

# Constrained Binary Optimization (CBO)

The insurer wants to:

\[
\max \; r^T x
\]

Subject to:

\[
c^T x \leq B
\]

Where:

- \( r \) = expected payout vector
- \( c \) = premium cost vector
- \( B \) = total budget

This is a **0-1 knapsack-type constrained optimization problem**.

---

# From CBO to QUBO

Quantum algorithms like QAOA require an **Unconstrained Binary Optimization (UBO)** form.

We convert the constrained problem into an unconstrained one by adding a quadratic penalty term:

\[
\min \; -r^T x + \lambda (c^T x - B)^2
\]

Where:

- \( \lambda \) = penalty coefficient enforcing budget constraint

This produces a QUBO matrix:

\[
E(x) = x^T Q x
\]

---

# QUBO → Ising Mapping

To run QAOA, we convert binary variables:

\[
x_i = \frac{1 - z_i}{2}
\]

Where:

\[
z_i \in \{-1, +1\}
\]

This yields an Ising Hamiltonian:

\[
H = \sum_{i<j} J_{ij} Z_i Z_j + \sum_i h_i Z_i + \text{offset}
\]

Which can be directly implemented in a quantum circuit.

---

# Solvers Implemented

## Classical Solvers

- Brute Force (exact, exponential complexity)
- Greedy heuristic (O(N log N))
- Simulated Annealing (approximate, stochastic)

## Quantum Solver

- QAOA (Quantum Approximate Optimization Algorithm)

We benchmark:

- Solution quality
- Runtime scaling
- Approximation ratio
- Sensitivity to circuit depth p

---

# Benchmarking

We compare classical and quantum approaches across increasing values of N.

- Brute force scales exponentially.
- Greedy scales approximately O(N log N).
- Simulated Annealing scales approximately O(iterations × N).
- QAOA runtime depends on:
  - Circuit depth p
  - Number of optimizer evaluations
  - Shot count

The framework allows direct runtime and solution quality comparison.


---

# Conclusion

We successfully:

- Modeled a real-world insurance risk allocation problem
- Derived expected risk values from historical catastrophe data
- Formulated the problem as QUBO
- Implemented both classical and quantum-inspired solvers
- Benchmarked performance and scalability

This project demonstrates how financial risk management problems can be mapped to quantum-ready optimization frameworks.