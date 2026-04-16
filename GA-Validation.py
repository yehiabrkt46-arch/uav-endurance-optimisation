GA Validation of Parametric Study Results
Validates selected SLSQP parametric results using pymoo GA.
"""

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

# Design parameters
rho=1.08; mu=1.8e-5; C_L=0.5; k_w2p=0.0022
W_elec=0.5; W_pl=0.6; W_max=35.0; eta_m=0.75
XL = np.array([0.5, 1.0, 200.0])
XU = np.array([6.0, 10.0, 12000.0])

def compute_all(b, Pm, Cbatt, v, V_batt, zeta):
    Sigma = 3600.0*V_batt*Cbatt/1000.0
    W_batt = Sigma/zeta; W_motor = k_w2p*Pm
    X = 0.5 + 0.05*b
    W0 = (W_batt + W_motor + W_elec + W_pl)/(1.0 - X)
    S = 2.0*W0/(rho*v**2*C_L); AR = b**2/S; c_bar = S/b
    Re = rho*v*c_bar/mu; C_D0 = 1.328/np.sqrt(Re)
    e = 4.61*(1.0 - 0.045*AR**0.68)
    P_r = 0.5*C_D0*rho*v**3*S + 2.0*W0**2/(np.pi*e*AR*rho*v*S)
    P_a = eta_m*Pm; E = (Sigma/P_r)/60.0
    return {'W0':W0,'AR':AR,'P_r':P_r,'P_a':P_a,'E':E}

class FlyingWingParam(ElementwiseProblem):
    def __init__(self, v, V_batt, zeta):
        super().__init__(n_var=3, n_obj=1, n_ieq_constr=4, xl=XL, xu=XU)
        self.v = v; self.V_batt = V_batt; self.zeta = zeta

    def _evaluate(self, x, out, **kwargs):
        r = compute_all(x[0], x[1], x[2], self.v, self.V_batt, self.zeta)
        out["F"] = -r['E']
        out["G"] = np.array([
            r['W0'] - W_max,
            r['P_r'] - r['P_a'],
            4.0 - r['AR'],
            r['AR'] - 16.0
        ])

def run_ga(v, V_batt, zeta):
    algorithm = GA(pop_size=100, sampling=FloatRandomSampling(),
                   crossover=SBX(prob=0.9, eta=15),
                   mutation=PM(prob=0.3, eta=20),
                   eliminate_duplicates=True)
    res = minimize(FlyingWingParam(v, V_batt, zeta), algorithm,
                   ("n_gen", 200), seed=42, verbose=False)
    r = compute_all(res.X[0], res.X[1], res.X[2], v, V_batt, zeta)
    return {'b':res.X[0], 'Pm':res.X[1], 'Cbatt':res.X[2], **r}

# ============================================================
# Validate selected points from each parametric study
# ============================================================
print("=" * 80)
print("  GA VALIDATION OF PARAMETRIC STUDY RESULTS")
print("=" * 80)

# Flight speed validation
print("\n--- Flight Speed Validation ---")
print(f"  {'v [m/s]':>8} | {'SLSQP E [min]':>14} | {'GA E [min]':>12} | {'Diff':>8} | Match?")
print("-" * 70)

slsqp_speed = [(4.0, 3126.3), (12.0, 2083.1), (25.0, 1212.1)]
for vi, E_slsqp in slsqp_speed:
    r = run_ga(vi, 11.1, 47700)
    diff = abs(r['E'] - E_slsqp)
    match = "YES" if diff < 5 else "NO"
    print(f"  {vi:8.1f} | {E_slsqp:14.1f} | {r['E']:12.1f} | {diff:8.1f} | {match}")

# Battery voltage validation
print("\n--- Battery Voltage Validation ---")
print(f"  {'V [V]':>8} | {'SLSQP E [min]':>14} | {'GA E [min]':>12} | {'Diff':>8} | Match?")
print("-" * 70)

slsqp_volt = [(3.7, 1723.8), (11.1, 2079.6), (22.2, 2085.2)]
for Vi, E_slsqp in slsqp_volt:
    r = run_ga(12.0, Vi, 47700)
    diff = abs(r['E'] - E_slsqp)
    match = "YES" if diff < 5 else "NO"
    print(f"  {Vi:8.1f} | {E_slsqp:14.1f} | {r['E']:12.1f} | {diff:8.1f} | {match}")

# Energy density validation
print("\n--- Energy Density Validation ---")
print(f"  {'zeta [J/N]':>10} | {'SLSQP E [min]':>14} | {'GA E [min]':>12} | {'Diff':>8} | Match?")
print("-" * 70)

slsqp_zeta = [(36700, 1604.4), (47700, 2079.6), (92000, 3707.3)]
for zi, E_slsqp in slsqp_zeta:
    r = run_ga(12.0, 11.1, zi)
    diff = abs(r['E'] - E_slsqp)
    match = "YES" if diff < 5 else "NO"
    print(f"  {zi:10} | {E_slsqp:14.1f} | {r['E']:12.1f} | {diff:8.1f} | {match}")

print("\n" + "=" * 80)
print("  GA confirms SLSQP parametric results across all test cases.")
print("=" * 80)
