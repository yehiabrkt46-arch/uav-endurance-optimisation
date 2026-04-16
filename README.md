# uav-endurance-optimisation
UAV endurance maximisation using SLSQP and genetic algorithms
# UAV Endurance Optimisation

Constrained optimisation of an electrically powered flying-wing UAV to 
maximise flight endurance. Implements and compares two methods — 
gradient-based SLSQP and a real-coded Genetic Algorithm (GA) — across 
three design variables: wingspan, motor power, and battery capacity.

## Problem Overview

The design problem balances competing trade-offs:
- Larger wingspan → lower induced drag, but higher structural weight
- More battery capacity → more energy, but heavier aircraft requiring more power
- Motor must sustain level flight without being oversized

Three design variables are optimised subject to four constraints (max 
weight, available vs. required power, aspect ratio bounds).

## Results

Both methods converge to the same optimal design:

| Parameter        | Optimal Value |
|-----------------|---------------|
| Wingspan        | 2.81 m        |
| Motor Power     | 5.12 W        |
| Battery         | 12,000 mAh    |
| Endurance       | 2,079.6 min (34.7 hrs) |
| Aspect Ratio    | 9.89          |

The power constraint is the sole active constraint at the optimum 
(Pr = Pa = 3.843 W), confirming the motor is sized exactly to sustain 
level flight — consistent with KKT complementary slackness conditions.

## Methods

**SLSQP** (`Flying-Wing-SLSQP.py`)  
Gradient-based solver via `scipy.optimize.minimize`. Converges in 24 
iterations / 104 function evaluations. Multi-start analysis from 6 
initial points all converge to the same solution, supporting global 
optimality.

**Genetic Algorithm** (`Flying-Wing-GA.py`)  
Implemented with `pymoo`. Population: 100, SBX crossover, polynomial 
mutation. Converges in ~35 generations / 10,700 function evaluations. 
Reaches identical solution independently, validating the SLSQP result.

SLSQP is ~100× more computationally efficient for this smooth, 
continuous problem.

## Parametric Study

Sensitivity of the optimal design to three parameters:

- **Flight speed** — endurance decreases monotonically with speed due 
  to cubic parasite drag growth
- **Battery voltage** — endurance saturates beyond 4S (14.8V) where 
  the weight constraint becomes active
- **Energy density** — most influential parameter; a 131% endurance 
  improvement across the range studied through a compounding 
  weight-drag-power mechanism

## Dependencies

```bash
pip install numpy scipy pymoo matplotlib
```

## Files

| File | Description |
|------|-------------|
| `Flying-Wing-Model.py` | Core aircraft model and endurance function |
| `Flying-Wing-SLSQP.py` | Gradient-based optimisation |
| `Flying-Wing-GA.py` | Genetic algorithm optimisation |
| `Flying-Wing-parametric study.py` | Parametric sensitivity analysis |
| `GA-Validation.py` | Cross-validation of SLSQP results using GA |
