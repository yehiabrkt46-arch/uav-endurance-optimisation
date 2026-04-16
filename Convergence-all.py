# -*- coding: utf-8 -*-
"""
EN3080 Coursework - All convergence comparison plots
1. SLSQP: different initial guesses (endurance)
2. SLSQP: different flight speeds
3. SLSQP: different battery voltages
4. GA: different seeds (endurance)
5. GA: different flight speeds
6. GA: different battery voltages
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize as sp_minimize

# ============================================================
# Model
# ============================================================
rho=1.08; mu=1.8e-5; C_L=0.5; lam=0.7
k_w2p=0.0022; W_elec=0.5; W_pl=0.6; W_max=35.0; eta_m=0.75

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

def run_slsqp_history(x0, v, V_batt, zeta):
    def obj(x): return -compute_all(x[0],x[1],x[2],v,V_batt,zeta)['E']
    def c_w(x): return W_max - compute_all(x[0],x[1],x[2],v,V_batt,zeta)['W0']
    def c_p(x): r=compute_all(x[0],x[1],x[2],v,V_batt,zeta); return r['P_a']-r['P_r']
    def c_al(x): return compute_all(x[0],x[1],x[2],v,V_batt,zeta)['AR'] - 4.0
    def c_au(x): return 16.0 - compute_all(x[0],x[1],x[2],v,V_batt,zeta)['AR']
    cons = [{'type':'ineq','fun':c_w}, {'type':'ineq','fun':c_p},
            {'type':'ineq','fun':c_al}, {'type':'ineq','fun':c_au}]
    hist = [obj(x0)]
    def cb(xk): hist.append(obj(xk))
    sp_minimize(obj, x0, method='SLSQP', bounds=[(0.5,6),(1,10),(200,12000)],
                constraints=cons, callback=cb, options={'maxiter':200,'ftol':1e-12})
    return [-f for f in hist]

plt.rcParams.update({"font.size": 13, "axes.titlesize": 15,
                     "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11})
colors = ['blue', 'red', 'green']

# ============================================================
# SLSQP PLOT 1: Different initial guesses (baseline params)
# ============================================================
initial_guesses = [
    ([1.0, 2.0, 1000.0], '$x_0$ = [1.0, 2.0, 1000]'),
    ([2.0, 5.0, 5000.0], '$x_0$ = [2.0, 5.0, 5000]'),
    ([5.0, 10.0, 11000.0], '$x_0$ = [5.0, 10.0, 11000]'),
]

plt.figure(figsize=(9, 5))
for (x0, label), col in zip(initial_guesses, colors):
    E_hist = run_slsqp_history(x0, 12.0, 11.1, 47700.0)
    plt.plot(range(len(E_hist)), E_hist, '-o', color=col, markersize=4,
             linewidth=1.5, label=f'{label} (E* = {E_hist[-1]:.0f} min)')

plt.xlabel('Iteration')
plt.ylabel('Endurance [min]')
plt.title('SLSQP Convergence from Different Initial Guesses')
plt.legend(fontsize=10)
plt.grid(True, linewidth=0.3)
plt.tight_layout()
plt.savefig('slsqp_conv_initial_guesses.png', dpi=150)
plt.show()

# ============================================================
# SLSQP PLOT 2: Different flight speeds
# ============================================================
speeds = [4.0, 12.0, 25.0]

plt.figure(figsize=(9, 5))
for vi, col in zip(speeds, colors):
    E_hist = run_slsqp_history([2.0, 5.0, 5000.0], vi, 11.1, 47700.0)
    plt.plot(range(len(E_hist)), E_hist, '-o', color=col, markersize=4,
             linewidth=1.5, label=f'$v$ = {vi:.0f} m/s (E* = {E_hist[-1]:.0f} min)')

plt.xlabel('Iteration')
plt.ylabel('Endurance [min]')
plt.title('SLSQP Convergence at Different Flight Speeds')
plt.legend(fontsize=11)
plt.grid(True, linewidth=0.3)
plt.tight_layout()
plt.savefig('slsqp_conv_flight_speed.png', dpi=150)
plt.show()

# ============================================================
# SLSQP PLOT 3: Different battery voltages
# ============================================================
voltages = [3.7, 11.1, 22.2]
vlabels = ['1S (3.7V)', '3S (11.1V)', '6S (22.2V)']

plt.figure(figsize=(9, 5))
for Vi, label, col in zip(voltages, vlabels, colors):
    E_hist = run_slsqp_history([2.0, 5.0, 5000.0], 12.0, Vi, 47700.0)
    plt.plot(range(len(E_hist)), E_hist, '-o', color=col, markersize=4,
             linewidth=1.5, label=f'{label} (E* = {E_hist[-1]:.0f} min)')

plt.xlabel('Iteration')
plt.ylabel('Endurance [min]')
plt.title('SLSQP Convergence at Different Battery Voltages')
plt.legend(fontsize=11)
plt.grid(True, linewidth=0.3)
plt.tight_layout()
plt.savefig('slsqp_conv_battery_voltage.png', dpi=150)
plt.show()

print("SLSQP plots saved.")
print("Now running GA plots (this takes a few minutes)...")

# ============================================================
# GA Setup
# ============================================================
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

XL = np.array([0.5, 1.0, 200.0])
XU = np.array([6.0, 10.0, 12000.0])

class FlyingWingParam(ElementwiseProblem):
    def __init__(self, v, V_batt, zeta):
        super().__init__(n_var=3, n_obj=1, n_ieq_constr=4, xl=XL, xu=XU)
        self.v = v; self.V_batt = V_batt; self.zeta = zeta
    def _evaluate(self, x, out, **kwargs):
        r = compute_all(x[0], x[1], x[2], self.v, self.V_batt, self.zeta)
        out["F"] = -r['E']
        out["G"] = np.array([r['W0']-W_max, r['P_r']-r['P_a'], 4.0-r['AR'], r['AR']-16.0])

def run_ga_history(v, V_batt, zeta, seed=42):
    algorithm = GA(pop_size=100, sampling=FloatRandomSampling(),
                   crossover=SBX(prob=0.9, eta=15),
                   mutation=PM(prob=0.3, eta=20),
                   eliminate_duplicates=True)
    res = pymoo_minimize(FlyingWingParam(v, V_batt, zeta), algorithm,
                         ("n_gen", 150), seed=seed, save_history=True, verbose=False)
    E_best = []
    for algo in res.history:
        pop_F = algo.pop.get("F").flatten()
        pop_G = algo.pop.get("G")
        feas = np.all(pop_G <= 0, axis=1)
        if np.any(feas):
            E_best.append(-float(np.min(pop_F[feas])))
        else:
            E_best.append(np.nan)
    return E_best

# ============================================================
# GA PLOT 1: Different seeds (baseline params)
# ============================================================
seeds = [(42, 'Seed = 42'), (7, 'Seed = 7'), (123, 'Seed = 123')]

print("  GA: different seeds...")
plt.figure(figsize=(9, 5))
for (seed, label), col in zip(seeds, colors):
    E_hist = run_ga_history(12.0, 11.1, 47700.0, seed=seed)
    gens = range(1, len(E_hist)+1)
    plt.plot(gens, E_hist, '-', color=col, linewidth=2,
             label=f'{label} (E* = {E_hist[-1]:.0f} min)')

plt.xlabel('Generation')
plt.ylabel('Endurance [min]')
plt.title('GA Convergence with Different Random Seeds')
plt.legend(fontsize=11)
plt.grid(True, linewidth=0.3)
plt.tight_layout()
plt.savefig('ga_conv_seeds.png', dpi=150)
plt.show()

# ============================================================
# GA PLOT 2: Different flight speeds
# ============================================================
print("  GA: different flight speeds...")
plt.figure(figsize=(9, 5))
for vi, col in zip(speeds, colors):
    E_hist = run_ga_history(vi, 11.1, 47700.0)
    gens = range(1, len(E_hist)+1)
    plt.plot(gens, E_hist, '-', color=col, linewidth=2,
             label=f'$v$ = {vi:.0f} m/s (E* = {E_hist[-1]:.0f} min)')

plt.xlabel('Generation')
plt.ylabel('Endurance [min]')
plt.title('GA Convergence at Different Flight Speeds')
plt.legend(fontsize=11)
plt.grid(True, linewidth=0.3)
plt.tight_layout()
plt.savefig('ga_conv_flight_speed.png', dpi=150)
plt.show()

# ============================================================
# GA PLOT 3: Different battery voltages
# ============================================================
print("  GA: different battery voltages...")
plt.figure(figsize=(9, 5))
for Vi, label, col in zip(voltages, vlabels, colors):
    E_hist = run_ga_history(12.0, Vi, 47700.0)
    gens = range(1, len(E_hist)+1)
    plt.plot(gens, E_hist, '-', color=col, linewidth=2,
             label=f'{label} (E* = {E_hist[-1]:.0f} min)')

plt.xlabel('Generation')
plt.ylabel('Endurance [min]')
plt.title('GA Convergence at Different Battery Voltages')
plt.legend(fontsize=11)
plt.grid(True, linewidth=0.3)
plt.tight_layout()
plt.savefig('ga_conv_battery_voltage.png', dpi=150)
plt.show()

print("\nAll 6 plots saved:")
print("  slsqp_conv_initial_guesses.png")
print("  slsqp_conv_flight_speed.png")
print("  slsqp_conv_battery_voltage.png")
print("  ga_conv_seeds.png")
print("  ga_conv_flight_speed.png")
print("  ga_conv_battery_voltage.png")
