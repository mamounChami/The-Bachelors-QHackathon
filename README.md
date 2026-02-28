# Reinsurance Allocation Optimization

We choose the Reinsurance Allocation Optimization problem.
The insurer must decide which reinsurance contracts to purchase in order to reduce exposure to catastrophic losses while respecting a budget constraint.

## Description of the problem

The insurer faces several potential catastrophic loss scenarios, each associated with a quantified financial loss. To mitigate these extreme risks, the company can purchase reinsurance contracts. Each contract, if selected, reduces the corresponding loss exposure but comes at a fixed cost. 
The insurer operates under a constrained reinsurance budget, meaning it cannot purchase all available contracts and must therefore optimize its selection. 

We assume a binary decision framework: each contract is either purchased or not. The reduction in losses is modeled as linear with respect to the selected contracts, and the overall budget constraint is enforced through a penalty mechanism within the optimization model.

### Decisison variable 

We define binary decision variables 𝑥𝑖 ∈{0,1}, where each variable represents a reinsurance contract.

- 𝑥𝑖 = 1 → Contract i is purchased​
- 𝑥𝑖 = 0 → Contract i is not purchased

In the quantum formulation, the number of qubits required is equal to the number of available contracts (N).

### Objective Function 

