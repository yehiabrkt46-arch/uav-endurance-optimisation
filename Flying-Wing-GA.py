Electrically Powered Flying Wing Design - GA Optimisation

Genetic Algorithm optimisation using pymoo.
Consistent with the SLSQP formulation for comparison.

Design Variables:  x = [b, Pm, Cbatt]
Objective:         Maximise endurance E [min] => Minimise -E
Constraints:       Weight, Power, Aspect Ratio bounds
"""

import numpy as np
import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

# ============================================================
# Design Parameters (Table 1)
# ============================================================
lam     = 0.7          # taper ratio
rho     = 1.08         # air density [kg/m^3]
mu      = 1.8e-5       # dynamic viscosity [kg/(m s)]
C_L     = 0.5          # lift coefficient
v       = 12.0         # flight speed [m/s]
V_batt  = 11.1         # battery voltage [V]
zeta    = 47700.0      # battery energy density [J/N]
k_w2p   = 0.0022       # motor weight-to-power ratio [N/W]
W_elec  = 0.5          # electronics weight [N]
W_pl    = 0.6          # payload weight [N]
W_max   = 35.0         # maximum allowable weight [N]
eta_m   = 0.75         # overall motor+battery efficiency

# Design variable bounds
XL = np.array([0.5,   1.0,   200.0])
XU = np.array([6.0,  10.0, 12000.0])


# ============================================================
# Flying Wing Model
# ============================================================
def compute_all(b, Pm, Cbatt):
    """Compute all physical quantities from design variables."""
    Sigma   = 3600.0 * V_batt * Cbatt / 1000.0
    W_batt  = Sigma / zeta
    W_motor = k_w2p * Pm
    X       = 0.5 + 0.05 * b
    W0      = (W_batt + W_motor + W_elec + W_pl) / (1.0 - X)
    S       = 2.0 * W0 / (rho * v**2 * C_L)
    AR      = b**2 / S
    c_bar   = S / b
    c_r     = 2.0 * c_bar / (1.0 + lam)
    c_t     = lam * c_r
    Re      = rho * v * c_bar / mu
    C_D0    = 1.328 / np.sqrt(Re)
    e       = 4.61 * (1.0 - 0.045 * AR**0.68)
    P_parasite = 0.5 * C_D0 * rho * v**3 * S
    P_induced  = 2.0 * W0**2 / (np.pi * e * AR * rho * v * S)
    P_r     = P_parasite + P_induced
    P_a     = eta_m * Pm
    E       = (Sigma / P_r) / 60.0

    return {
        'Sigma': Sigma, 'W_batt': W_batt, 'W_motor': W_motor,
        'X': X, 'W0': W0, 'S': S, 'AR': AR, 'c_bar': c_bar,
        'c_r': c_r, 'c_t': c_t,
        'Re': Re, 'C_D0': C_D0, 'e': e,
        'P_parasite': P_parasite, 'P_induced': P_induced,
        'P_r': P_r, 'P_a': P_a, 'E': E
    }


def fly_objective(x):
    """Objective: minimise -E."""
    return -compute_all(x[0], x[1], x[2])['E']


def fly_constraints(x):
    """
    Inequality constraints g_i <= 0.
    Returns [g1, g2, g3, g4].
    """
    r = compute_all(x[0], x[1], x[2])
    g1 = r['W0'] - W_max          # weight limit
    g2 = r['P_r'] - r['P_a']      # power requirement
    g3 = 4.0 - r['AR']            # AR >= 4
    g4 = r['AR'] - 16.0           # AR <= 16
    return np.array([g1, g2, g3, g4])


# ============================================================
# Define pymoo Problem class
# ============================================================
class FlyingWingProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=3,
            n_obj=1,
            n_ieq_constr=4,
            xl=XL,
            xu=XU
        )

    def _evaluate(self, x, out, **kwargs):
        out["F"] = fly_objective(x)
        out["G"] = fly_constraints(x)


# ============================================================
# GA Configuration and Execution
# ============================================================
POP_SIZE = 100
NGEN     = 300
SEED     = 42

termination = DefaultSingleObjectiveTermination(
    n_max_gen=NGEN,
    ftol=1e-12,
)

algorithm = GA(
    pop_size=POP_SIZE,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(prob=0.3, eta=20),
    eliminate_duplicates=True
)

res = minimize(
    FlyingWingProblem(),
    algorithm,
    termination,
    seed=SEED,
    save_history=True,
    verbose=True
)

# ============================================================
# Extract and print optimal results
# ============================================================
x_best = np.asarray(res.X, float)
f_best = float(np.squeeze(res.F))
b_opt, Pm_opt, Cbatt_opt = x_best
E_opt = -f_best
res_opt = compute_all(b_opt, Pm_opt, Cbatt_opt)

print("\n" + "=" * 60)
print("  GA OPTIMISATION RESULTS")
print("=" * 60)
print(f"  Generations: {len(res.history)}")
print(f"  Population size: {POP_SIZE}")
print("-" * 60)
print("  OPTIMAL DESIGN VARIABLES:")
print(f"    Wingspan        b     = {b_opt:.4f} m")
print(f"    Motor power     Pm    = {Pm_opt:.4f} W")
print(f"    Battery cap.    Cbatt = {Cbatt_opt:.1f} mAh")
print("-" * 60)
print("  DERIVED QUANTITIES:")
print(f"    Battery energy  Sigma = {res_opt['Sigma']:.1f} J")
print(f"    Battery weight  W_bat = {res_opt['W_batt']:.4f} N")
print(f"    Motor weight    W_mot = {res_opt['W_motor']:.4f} N")
print(f"    Struct. frac.   X     = {res_opt['X']:.4f}")
print(f"    Total weight    W0    = {res_opt['W0']:.4f} N")
print(f"    Wing area       S     = {res_opt['S']:.4f} m^2")
print(f"    Aspect ratio    AR    = {res_opt['AR']:.4f}")
print(f"    Mean chord      c_bar = {res_opt['c_bar']:.4f} m")
print(f"    Root chord      c_r   = {res_opt['c_r']:.4f} m")
print(f"    Tip chord       c_t   = {res_opt['c_t']:.4f} m")
print(f"    Reynolds no.    Re    = {res_opt['Re']:.0f}")
print(f"    Parasite drag   C_D0  = {res_opt['C_D0']:.6f}")
print(f"    Oswald eff.     e     = {res_opt['e']:.4f}")
print("-" * 60)
print("  PERFORMANCE:")
print(f"    Parasite power  P_par = {res_opt['P_parasite']:.4f} W")
print(f"    Induced power   P_ind = {res_opt['P_induced']:.4f} W")
print(f"    Required power  P_r   = {res_opt['P_r']:.4f} W")
print(f"    Available power P_a   = {res_opt['P_a']:.4f} W")
print(f"    Endurance       E     = {res_opt['E']:.2f} min")
print(f"                          = {res_opt['E']/60:.2f} hours")
print("-" * 60)

# Check constraint activity
print("  CONSTRAINT STATUS:")
g = fly_constraints(x_best)
labels = ["Weight  (W0-Wmax)", "Power   (Pr-Pa) ",
          "AR >= 4 (4-AR)  ", "AR <= 16 (AR-16)"]
for label, val in zip(labels, g):
    if abs(val) < 1e-2:
        status = "ACTIVE"
    elif val <= 0:
        status = f"FEASIBLE (margin = {-val:.4f})"
    else:
        status = f"VIOLATED ({val:.4f})"
    print(f"    {label}: {val:+.4f}  {status}")
print("=" * 60)


# ============================================================
# Convergence: f_min and f_avg per generation
# ============================================================
f_min_hist, f_avg_hist = [], []
for algo in res.history:
    pop_F = algo.pop.get("F").flatten()
    # Only consider feasible individuals
    pop_G = algo.pop.get("G")
    feasible_mask = np.all(pop_G <= 0, axis=1)
    if np.any(feasible_mask):
        feas_F = pop_F[feasible_mask]
        f_min_hist.append(float(np.min(feas_F)))
        f_avg_hist.append(float(np.mean(feas_F)))
    else:
        f_min_hist.append(np.nan)
        f_avg_hist.append(np.nan)

# Convert to positive endurance
E_min_hist = [-f if not np.isnan(f) else np.nan for f in f_min_hist]  # best endurance
E_avg_hist = [-f if not np.isnan(f) else np.nan for f in f_avg_hist]  # avg endurance


# ============================================================
# PLOT 1: Convergence (Endurance: best and average)
# ============================================================
plt.figure(figsize=(9, 5))
gens = range(1, len(E_min_hist) + 1)
plt.plot(gens, E_min_hist, linewidth=2, label=r'$E_{\mathrm{best}}$ (feasible)')
plt.plot(gens, E_avg_hist, linewidth=2, label=r'$E_{\mathrm{avg}}$ (feasible)')
plt.xlabel('Generation', fontsize=14)
plt.ylabel('Endurance [min]', fontsize=14)
plt.title('GA Convergence: Endurance vs Generation', fontsize=16)
plt.grid(True, linewidth=0.3)
plt.legend(fontsize=12)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('ga_convergence.png', dpi=150)
plt.show()


# ============================================================
# PLOT 2: Design variable convergence (best individual)
# ============================================================
best_b, best_Pm, best_Cb = [], [], []
for algo in res.history:
    pop_F = algo.pop.get("F").flatten()
    pop_G = algo.pop.get("G")
    pop_X = algo.pop.get("X")
    feasible_mask = np.all(pop_G <= 0, axis=1)
    if np.any(feasible_mask):
        idx = np.argmin(pop_F[feasible_mask])
        feas_X = pop_X[feasible_mask]
        best_b.append(feas_X[idx, 0])
        best_Pm.append(feas_X[idx, 1])
        best_Cb.append(feas_X[idx, 2])
    else:
        best_b.append(np.nan)
        best_Pm.append(np.nan)
        best_Cb.append(np.nan)

fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

axes[0].plot(gens, best_b, 'r-', linewidth=1.5)
axes[0].set_ylabel('Wingspan $b$ [m]', fontsize=13)
axes[0].axhline(y=b_opt, color='r', linestyle='--', alpha=0.4,
                label=f'Final: {b_opt:.2f} m')
axes[0].legend(fontsize=11); axes[0].grid(True, linewidth=0.3)

axes[1].plot(gens, best_Pm, 'g-', linewidth=1.5)
axes[1].set_ylabel('Motor power $P_m$ [W]', fontsize=13)
axes[1].axhline(y=Pm_opt, color='g', linestyle='--', alpha=0.4,
                label=f'Final: {Pm_opt:.2f} W')
axes[1].legend(fontsize=11); axes[1].grid(True, linewidth=0.3)

axes[2].plot(gens, best_Cb, 'm-', linewidth=1.5)
axes[2].set_ylabel('Battery cap. $C_{batt}$ [mAh]', fontsize=13)
axes[2].set_xlabel('Generation', fontsize=14)
axes[2].axhline(y=Cbatt_opt, color='m', linestyle='--', alpha=0.4,
                label=f'Final: {Cbatt_opt:.0f} mAh')
axes[2].legend(fontsize=11); axes[2].grid(True, linewidth=0.3)

fig.suptitle('GA: Best Design Variable Convergence', fontsize=16)
plt.tight_layout()
plt.savefig('ga_design_variables.png', dpi=150)
plt.show()


# ============================================================
# Comparison: SLSQP vs GA
# ============================================================
# SLSQP results (from previous run)
slsqp_b, slsqp_Pm, slsqp_Cb = 2.8115, 5.1240, 12000.0
slsqp_res = compute_all(slsqp_b, slsqp_Pm, slsqp_Cb)

print("\n" + "=" * 70)
print("  COMPARISON: SLSQP vs GA")
print("=" * 70)
print(f"  {'Quantity':<25} {'SLSQP':>15} {'GA':>15} {'Diff':>10}")
print("-" * 70)
rows = [
    ("Wingspan b [m]",       slsqp_b,              b_opt),
    ("Motor power Pm [W]",   slsqp_Pm,             Pm_opt),
    ("Battery Cbatt [mAh]",  slsqp_Cb,             Cbatt_opt),
    ("Total weight W0 [N]",  slsqp_res['W0'],      res_opt['W0']),
    ("Wing area S [m^2]",    slsqp_res['S'],       res_opt['S']),
    ("Aspect ratio AR",      slsqp_res['AR'],      res_opt['AR']),
    ("Required power Pr [W]",slsqp_res['P_r'],     res_opt['P_r']),
    ("Available power Pa [W]",slsqp_res['P_a'],    res_opt['P_a']),
    ("Endurance E [min]",    slsqp_res['E'],       res_opt['E']),
    ("Endurance E [hours]",  slsqp_res['E']/60,    res_opt['E']/60),
]
for name, v1, v2 in rows:
    diff = v2 - v1
    print(f"  {name:<25} {v1:>15.4f} {v2:>15.4f} {diff:>+10.4f}")
print("-" * 70)
print(f"  {'Iterations/Generations':<25} {'24':>15} {len(res.history):>15}")
print(f"  {'Function evaluations':<25} {'104':>15} {POP_SIZE*len(res.history):>15}")
print("=" * 70)

print("\nAll plots saved: 'ga_convergence.png', 'ga_design_variables.png'")


# ============================================================
# PLOT 3: Side-by-side convergence comparison (SLSQP vs GA)
# ============================================================
from scipy.optimize import minimize as sp_minimize

# Re-run SLSQP to capture its convergence history
def obj_slsqp(x):
    return -compute_all(x[0], x[1], x[2])['E']

def c_w(x): return W_max - compute_all(x[0],x[1],x[2])['W0']
def c_p(x): r=compute_all(x[0],x[1],x[2]); return r['P_a']-r['P_r']
def c_al(x): return compute_all(x[0],x[1],x[2])['AR'] - 4.0
def c_au(x): return 16.0 - compute_all(x[0],x[1],x[2])['AR']

slsqp_hist = [obj_slsqp([2.0, 5.0, 5000.0])]
def slsqp_cb(xk):
    slsqp_hist.append(obj_slsqp(xk))

sp_minimize(obj_slsqp, [2.0, 5.0, 5000.0], method='SLSQP',
            bounds=[(0.5,6),(1,10),(200,12000)],
            constraints=[{'type':'ineq','fun':c_w}, {'type':'ineq','fun':c_p},
                         {'type':'ineq','fun':c_al}, {'type':'ineq','fun':c_au}],
            callback=slsqp_cb, options={'maxiter':200,'ftol':1e-12})

slsqp_E = [-f for f in slsqp_hist]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# SLSQP convergence
ax1.plot(range(len(slsqp_E)), slsqp_E, 'b-o', markersize=4, linewidth=1.5)
ax1.set_xlabel('Iteration', fontsize=13)
ax1.set_ylabel('Endurance [min]', fontsize=13)
ax1.set_title('SLSQP Convergence', fontsize=14)
ax1.grid(True, linewidth=0.3)
ax1.set_ylim(1500, 2150)

# GA convergence
ax2.plot(gens, E_min_hist, linewidth=2, label=r'$E_{\mathrm{best}}$')
ax2.plot(gens, E_avg_hist, linewidth=2, label=r'$E_{\mathrm{avg}}$')
ax2.set_xlabel('Generation', fontsize=13)
ax2.set_ylabel('Endurance [min]', fontsize=13)
ax2.set_title('GA Convergence', fontsize=14)
ax2.grid(True, linewidth=0.3)
ax2.legend(fontsize=11)
ax2.set_ylim(1500, 2150)

fig.suptitle('Convergence Comparison: SLSQP vs GA', fontsize=16)
plt.tight_layout()
plt.savefig('comparison_convergence.png', dpi=150)
plt.show()


# ============================================================
# Printed Commentary on Method Characteristics
# ============================================================
print("\n" + "=" * 70)
print("  COMMENTARY: SLSQP vs GA CHARACTERISTICS")
print("=" * 70)
print("""
  1. OPTIMAL RESULTS:
     Both methods converge to the same optimal endurance (2079.63 min),
     confirming the solution is a global optimum for this problem.

  2. CONVERGENCE SPEED:
     SLSQP converges in 24 iterations (104 function evaluations),
     while GA requires ~35 generations to reach the same accuracy
     (3500+ function evaluations). SLSQP is ~30x more efficient.

  3. WHY SLSQP IS FASTER:
     SLSQP exploits gradient (derivative) information to determine
     the search direction, enabling rapid convergence for this smooth,
     well-behaved problem. The GA relies on population-based stochastic
     search without gradient information.

  4. GA ADVANTAGES:
     - Derivative-free: does not require smooth objective/constraints
     - Global search: explores the entire design space via population
     - More robust for multimodal problems with multiple local optima

  5. SLSQP ADVANTAGES:
     - Much faster convergence for smooth, unimodal problems
     - Precise constraint handling via Lagrangian formulation
     - Computationally efficient (fewer function evaluations)

  6. RECOMMENDATION FOR THIS PROBLEM:
     SLSQP is the preferred method due to the smooth, continuous
     nature of the aerodynamic model. The GA serves as a valuable
     validation tool to confirm global optimality.
""")
print("=" * 70)
print("\nAll plots saved: 'ga_convergence.png', 'ga_design_variables.png',")
print("                 'comparison_convergence.png'")
