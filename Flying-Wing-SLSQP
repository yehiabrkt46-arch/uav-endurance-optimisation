Electrically Powered Flying Wing Design - SLSQP Optimisation

Gradient-based optimisation using Sequential Least Squares
Programming (SLSQP) from scipy.optimize.minimize.

Design Variables:  x = [b, Pm, Cbatt]
Objective:         Maximise endurance E [min] => Minimise -E
Constraints:       Weight, Power, Aspect Ratio bounds
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
BOUNDS = [(0.5, 6.0), (1.0, 10.0), (200.0, 12000.0)]


# ============================================================
# Flying Wing Model
# ============================================================
def compute_all(b, Pm, Cbatt):
    """
    Compute all physical and aerodynamic quantities from the
    three design variables (b, Pm, Cbatt).
    """
    # Step 1: Battery energy [J]
    Sigma = 3600.0 * V_batt * Cbatt / 1000.0

    # Step 2: Component weights [N]
    W_batt  = Sigma / zeta
    W_motor = k_w2p * Pm

    # Step 3: Structural weight fraction
    X = 0.5 + 0.05 * b

    # Step 4: Total weight [N]
    W0 = (W_batt + W_motor + W_elec + W_pl) / (1.0 - X)

    # Step 5: Wing planform area [m^2]
    S = 2.0 * W0 / (rho * v**2 * C_L)

    # Step 6: Aspect ratio
    AR = b**2 / S

    # Step 7: Mean chord [m]
    c_bar = S / b

    # Step 7b: Root and tip chords [m] (from taper ratio)
    c_r = 2.0 * c_bar / (1.0 + lam)
    c_t = lam * c_r

    # Step 8: Reynolds number
    Re = rho * v * c_bar / mu

    # Step 9: Parasite drag coefficient
    C_D0 = 1.328 / np.sqrt(Re)

    # Step 10: Oswald span efficiency factor
    e = 4.61 * (1.0 - 0.045 * AR**0.68)

    # Step 11: Drag terms and required power [W]
    P_parasite = 0.5 * C_D0 * rho * v**3 * S      # parasite (friction) drag power
    P_induced  = 2.0 * W0**2 / (np.pi * e * AR * rho * v * S)  # induced drag power
    P_r = P_parasite + P_induced                    # total required power

    # Step 12: Available power [W]
    P_a = eta_m * Pm

    # Step 13: Endurance [minutes]
    E = (Sigma / P_r) / 60.0

    return {
        'Sigma': Sigma, 'W_batt': W_batt, 'W_motor': W_motor,
        'X': X, 'W0': W0, 'S': S, 'AR': AR, 'c_bar': c_bar,
        'c_r': c_r, 'c_t': c_t,
        'Re': Re, 'C_D0': C_D0, 'e': e,
        'P_parasite': P_parasite, 'P_induced': P_induced,
        'P_r': P_r, 'P_a': P_a, 'E': E
    }


# ============================================================
# Objective function: minimise -E
# ============================================================
def objective(x):
    """Return -E (negative endurance) for minimisation."""
    b, Pm, Cbatt = x
    res = compute_all(b, Pm, Cbatt)
    return -res['E']


# ============================================================
# Constraint functions for SLSQP (each returns value >= 0)
# Note: scipy SLSQP 'ineq' constraints require f(x) >= 0
# ============================================================
def con_weight(x):
    """W_max - W0 >= 0"""
    res = compute_all(x[0], x[1], x[2])
    return W_max - res['W0']

def con_power(x):
    """P_a - P_r >= 0"""
    res = compute_all(x[0], x[1], x[2])
    return res['P_a'] - res['P_r']

def con_AR_lower(x):
    """AR - 4 >= 0"""
    res = compute_all(x[0], x[1], x[2])
    return res['AR'] - 4.0

def con_AR_upper(x):
    """16 - AR >= 0"""
    res = compute_all(x[0], x[1], x[2])
    return 16.0 - res['AR']


# ============================================================
# SLSQP Optimisation
# ============================================================

# Initial design point
x0 = np.array([2.0, 5.0, 5000.0])

# Store convergence history
history_x = [x0.copy()]
history_f = [objective(x0)]

def callback(xk):
    """Record design variables and objective at each iteration."""
    history_x.append(xk.copy())
    history_f.append(objective(xk))

# Define constraints list for SLSQP
constraints_list = [
    {'type': 'ineq', 'fun': con_weight},
    {'type': 'ineq', 'fun': con_power},
    {'type': 'ineq', 'fun': con_AR_lower},
    {'type': 'ineq', 'fun': con_AR_upper},
]

# Run SLSQP optimisation
options = {'maxiter': 200, 'ftol': 1e-12, 'disp': True}

result = minimize(
    objective,
    x0,
    method='SLSQP',
    bounds=BOUNDS,
    constraints=constraints_list,
    callback=callback,
    options=options
)


# ============================================================
# Extract and print optimal results
# ============================================================
b_opt, Pm_opt, Cbatt_opt = result.x
res_opt = compute_all(b_opt, Pm_opt, Cbatt_opt)

print("\n" + "=" * 60)
print("  SLSQP OPTIMISATION RESULTS")
print("=" * 60)
print(f"  Status: {result.message}")
print(f"  Iterations: {result.nit}")
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
g1 = W_max - res_opt['W0']
g2 = res_opt['P_a'] - res_opt['P_r']
g3 = res_opt['AR'] - 4.0
g4 = 16.0 - res_opt['AR']

labels = ["Weight  (Wmax-W0)", "Power   (Pa-Pr) ",
          "AR >= 4 (AR-4)  ", "AR <= 16 (16-AR)"]
values = [g1, g2, g3, g4]
for label, val in zip(labels, values):
    if abs(val) < 1e-3:
        status = "ACTIVE"
    elif val > 0:
        status = f"FEASIBLE (margin = {val:.4f})"
    else:
        status = f"VIOLATED ({val:.4f})"
    print(f"    {label}: {val:+.4f}  {status}")
print("=" * 60)


# ============================================================
# Sensitivity: test multiple initial points
# ============================================================
print("\n" + "=" * 60)
print("  SENSITIVITY TO INITIAL DESIGN POINT")
print("=" * 60)
print(f"{'x0':>30} | {'E* [min]':>10} | {'b*':>6} {'Pm*':>6} {'Cbatt*':>8} | Iters")
print("-" * 80)

initial_points = [
    [1.0, 2.0, 1000.0],
    [2.0, 5.0, 5000.0],
    [3.0, 7.0, 8000.0],
    [4.0, 9.0, 10000.0],
    [1.5, 3.0, 3000.0],
    [5.0, 10.0, 11000.0],
]

for x0_test in initial_points:
    res_test = minimize(
        objective, x0_test, method='SLSQP',
        bounds=BOUNDS, constraints=constraints_list,
        options={'maxiter': 200, 'ftol': 1e-12}
    )
    E_test = -res_test.fun
    b_t, Pm_t, Cb_t = res_test.x
    print(f"  {str(x0_test):>28} | {E_test:10.2f} | {b_t:6.2f} {Pm_t:6.2f} {Cb_t:8.1f} | {res_test.nit}")

print("=" * 60)


# ============================================================
# PLOT 1: Convergence (Endurance vs Iteration)
# ============================================================
traj = np.array(history_x)
endurance_hist = [-f for f in history_f]
iters = range(len(history_f))

plt.figure(figsize=(9, 5))
plt.plot(iters, endurance_hist, 'b-o', markersize=4, linewidth=1.5,
         label='Endurance (SLSQP)')
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Endurance [min]', fontsize=14)
plt.title('SLSQP Convergence: Endurance vs Iteration', fontsize=16)
plt.grid(True, linewidth=0.3)
plt.legend(fontsize=12)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('slsqp_convergence.png', dpi=150)
plt.show()


# ============================================================
# PLOT 2: Design Variable History (3 subplots)
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

axes[0].plot(iters, traj[:, 0], 'r-o', markersize=4, linewidth=1.5)
axes[0].set_ylabel('Wingspan $b$ [m]', fontsize=13)
axes[0].axhline(y=b_opt, color='r', linestyle='--', alpha=0.4,
                label=f'Optimal: {b_opt:.2f} m')
axes[0].legend(fontsize=11); axes[0].grid(True, linewidth=0.3)

axes[1].plot(iters, traj[:, 1], 'g-o', markersize=4, linewidth=1.5)
axes[1].set_ylabel('Motor power $P_m$ [W]', fontsize=13)
axes[1].axhline(y=Pm_opt, color='g', linestyle='--', alpha=0.4,
                label=f'Optimal: {Pm_opt:.2f} W')
axes[1].legend(fontsize=11); axes[1].grid(True, linewidth=0.3)

axes[2].plot(iters, traj[:, 2], 'm-o', markersize=4, linewidth=1.5)
axes[2].set_ylabel('Battery cap. $C_{batt}$ [mAh]', fontsize=13)
axes[2].set_xlabel('Iteration', fontsize=14)
axes[2].axhline(y=Cbatt_opt, color='m', linestyle='--', alpha=0.4,
                label=f'Optimal: {Cbatt_opt:.0f} mAh')
axes[2].legend(fontsize=11); axes[2].grid(True, linewidth=0.3)

fig.suptitle('SLSQP: Design Variable Convergence', fontsize=16)
plt.tight_layout()
plt.savefig('slsqp_design_variables.png', dpi=150)
plt.show()


# ============================================================
# PLOT 3: 2D Contour Plots (3 cross-sections)
# ============================================================
b_vals  = np.linspace(0.5, 6.0, 200)
Pm_vals = np.linspace(1.0, 10.0, 200)
Cb_vals = np.linspace(200, 12000, 200)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# --- Contour 1: b vs Cbatt (Pm fixed) ---
B1, CB1 = np.meshgrid(b_vals, Cb_vals)
E1 = np.zeros_like(B1)
feas1 = np.zeros_like(B1)
for i in range(B1.shape[0]):
    for j in range(B1.shape[1]):
        r = compute_all(B1[i,j], Pm_opt, CB1[i,j])
        E1[i,j] = r['E']
        ok = (W_max-r['W0']>=0) and (r['P_a']-r['P_r']>=0) and (r['AR']-4>=0) and (16-r['AR']>=0)
        feas1[i,j] = 1.0 if ok else np.nan

ax = axes[0]
cs = ax.contourf(B1, CB1, E1*feas1, levels=20, cmap='viridis')
ax.contour(B1, CB1, E1*feas1, levels=20, colors='k', linewidths=0.3, alpha=0.5)
plt.colorbar(cs, ax=ax, label='Endurance [min]')
ax.plot(traj[:,0], traj[:,2], 'r-o', markersize=3, linewidth=1.2, label='SLSQP path')
ax.plot(b_opt, Cbatt_opt, 'r*', markersize=15, label='Optimal')
ax.set_xlabel('Wingspan $b$ [m]', fontsize=13)
ax.set_ylabel('Battery capacity $C_{batt}$ [mAh]', fontsize=13)
ax.set_title(f'Endurance contour ($P_m$ = {Pm_opt:.2f} W fixed)', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, linewidth=0.2)

# --- Contour 2: b vs Pm (Cbatt fixed) ---
B2, PM2 = np.meshgrid(b_vals, Pm_vals)
E2 = np.zeros_like(B2)
feas2 = np.zeros_like(B2)
for i in range(B2.shape[0]):
    for j in range(B2.shape[1]):
        r = compute_all(B2[i,j], PM2[i,j], Cbatt_opt)
        E2[i,j] = r['E']
        ok = (W_max-r['W0']>=0) and (r['P_a']-r['P_r']>=0) and (r['AR']-4>=0) and (16-r['AR']>=0)
        feas2[i,j] = 1.0 if ok else np.nan

ax = axes[1]
cs2 = ax.contourf(B2, PM2, E2*feas2, levels=20, cmap='viridis')
ax.contour(B2, PM2, E2*feas2, levels=20, colors='k', linewidths=0.3, alpha=0.5)
plt.colorbar(cs2, ax=ax, label='Endurance [min]')
ax.plot(traj[:,0], traj[:,1], 'r-o', markersize=3, linewidth=1.2, label='SLSQP path')
ax.plot(b_opt, Pm_opt, 'r*', markersize=15, label='Optimal')
ax.set_xlabel('Wingspan $b$ [m]', fontsize=13)
ax.set_ylabel('Motor power $P_m$ [W]', fontsize=13)
ax.set_title(f'Endurance contour ($C_{{batt}}$ = {Cbatt_opt:.0f} mAh fixed)', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, linewidth=0.2)

# --- Contour 3: Pm vs Cbatt (b fixed) ---
PM3, CB3 = np.meshgrid(Pm_vals, Cb_vals)
E3 = np.zeros_like(PM3)
feas3 = np.zeros_like(PM3)
for i in range(PM3.shape[0]):
    for j in range(PM3.shape[1]):
        r = compute_all(b_opt, PM3[i,j], CB3[i,j])
        E3[i,j] = r['E']
        ok = (W_max-r['W0']>=0) and (r['P_a']-r['P_r']>=0) and (r['AR']-4>=0) and (16-r['AR']>=0)
        feas3[i,j] = 1.0 if ok else np.nan

ax = axes[2]
cs3 = ax.contourf(PM3, CB3, E3*feas3, levels=20, cmap='viridis')
ax.contour(PM3, CB3, E3*feas3, levels=20, colors='k', linewidths=0.3, alpha=0.5)
plt.colorbar(cs3, ax=ax, label='Endurance [min]')
ax.plot(traj[:,1], traj[:,2], 'r-o', markersize=3, linewidth=1.2, label='SLSQP path')
ax.plot(Pm_opt, Cbatt_opt, 'r*', markersize=15, label='Optimal')
ax.set_xlabel('Motor power $P_m$ [W]', fontsize=13)
ax.set_ylabel('Battery capacity $C_{batt}$ [mAh]', fontsize=13)
ax.set_title(f'Endurance contour ($b$ = {b_opt:.2f} m fixed)', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, linewidth=0.2)

plt.suptitle('SLSQP Optimisation: Feasible Design Space Contours', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('slsqp_contours.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nAll plots saved: 'slsqp_convergence.png', "
      "'slsqp_design_variables.png', 'slsqp_contours.png'")
