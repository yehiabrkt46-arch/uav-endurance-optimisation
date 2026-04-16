Electrically Powered Flying Wing Design - Parametric Study

Investigates how flight speed, battery voltage, and battery
energy density affect the optimal design and performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ============================================================
# Baseline Design Parameters (Table 1)
# ============================================================
rho     = 1.08
mu      = 1.8e-5
C_L     = 0.5
v_base  = 12.0
V_batt_base = 11.1
zeta_base   = 47700.0
k_w2p   = 0.0022
W_elec  = 0.5
W_pl    = 0.6
W_max   = 35.0
eta_m   = 0.75
BOUNDS  = [(0.5, 6.0), (1.0, 10.0), (200.0, 12000.0)]


# ============================================================
# Model (parameterised for v, V_batt, zeta)
# ============================================================
def compute_all(b, Pm, Cbatt, v, V_batt, zeta):
    Sigma   = 3600.0 * V_batt * Cbatt / 1000.0
    W_batt  = Sigma / zeta
    W_motor = k_w2p * Pm
    X       = 0.5 + 0.05 * b
    W0      = (W_batt + W_motor + W_elec + W_pl) / (1.0 - X)
    S       = 2.0 * W0 / (rho * v**2 * C_L)
    AR      = b**2 / S
    c_bar   = S / b
    Re      = rho * v * c_bar / mu
    C_D0    = 1.328 / np.sqrt(Re)
    e       = 4.61 * (1.0 - 0.045 * AR**0.68)
    P_r     = (0.5 * C_D0 * rho * v**3 * S
               + 2.0 * W0**2 / (np.pi * e * AR * rho * v * S))
    P_a     = eta_m * Pm
    E       = (Sigma / P_r) / 60.0
    return {'W0':W0, 'S':S, 'AR':AR, 'c_bar':c_bar,
            'P_r':P_r, 'P_a':P_a, 'E':E}


def run_optimisation(v, V_batt, zeta):
    """Run SLSQP for a given (v, V_batt, zeta) and return results."""
    def obj(x):
        return -compute_all(x[0], x[1], x[2], v, V_batt, zeta)['E']

    def c_w(x): return W_max - compute_all(x[0],x[1],x[2],v,V_batt,zeta)['W0']
    def c_p(x): r=compute_all(x[0],x[1],x[2],v,V_batt,zeta); return r['P_a']-r['P_r']
    def c_al(x): return compute_all(x[0],x[1],x[2],v,V_batt,zeta)['AR'] - 4.0
    def c_au(x): return 16.0 - compute_all(x[0],x[1],x[2],v,V_batt,zeta)['AR']

    cons = [{'type':'ineq','fun':c_w}, {'type':'ineq','fun':c_p},
            {'type':'ineq','fun':c_al}, {'type':'ineq','fun':c_au}]

    # Try multiple starting points, keep best feasible
    best_E = -np.inf
    best_result = None
    for x0 in [[2,5,5000],[1.5,3,3000],[3,7,8000],[4,9,10000]]:
        try:
            res = minimize(obj, x0, method='SLSQP', bounds=BOUNDS,
                           constraints=cons, options={'maxiter':300,'ftol':1e-12})
            if res.success or abs(res.fun) > 0:
                r = compute_all(res.x[0],res.x[1],res.x[2],v,V_batt,zeta)
                # Check feasibility
                g = [W_max-r['W0'], r['P_a']-r['P_r'], r['AR']-4, 16-r['AR']]
                if all(gi >= -1e-4 for gi in g) and r['E'] > best_E:
                    best_E = r['E']
                    best_result = res
        except:
            pass

    if best_result is None:
        return None

    r = compute_all(best_result.x[0], best_result.x[1], best_result.x[2],
                    v, V_batt, zeta)
    return {
        'b': best_result.x[0], 'Pm': best_result.x[1],
        'Cbatt': best_result.x[2], **r
    }


# ============================================================
# Parametric Study 1: Flight Speed v ∈ [4, 25] m/s
# ============================================================
print("=" * 60)
print("  PARAMETRIC STUDY 1: Flight Speed")
print("=" * 60)

v_range = np.linspace(4, 25, 30)
ps1 = {'v':[], 'b':[], 'AR':[], 'W0':[], 'E':[], 'P_r':[]}

for vi in v_range:
    r = run_optimisation(vi, V_batt_base, zeta_base)
    if r is not None:
        ps1['v'].append(vi)
        ps1['b'].append(r['b'])
        ps1['AR'].append(r['AR'])
        ps1['W0'].append(r['W0'])
        ps1['E'].append(r['E'])
        ps1['P_r'].append(r['P_r'])
        print(f"  v={vi:5.1f} m/s | b={r['b']:.3f} m | AR={r['AR']:.2f} "
              f"| W0={r['W0']:.2f} N | E={r['E']:.1f} min | Pr={r['P_r']:.3f} W")
    else:
        print(f"  v={vi:5.1f} m/s | NO FEASIBLE SOLUTION")


# ============================================================
# Parametric Study 2: Battery Voltage
# ============================================================
print("\n" + "=" * 60)
print("  PARAMETRIC STUDY 2: Battery Voltage")
print("=" * 60)

V_batt_values = [3.7, 7.4, 11.1, 14.8, 22.2]
V_batt_labels = ['1S (3.7V)', '2S (7.4V)', '3S (11.1V)', '4S (14.8V)', '6S (22.2V)']
ps2 = {'V':[], 'b':[], 'AR':[], 'W0':[], 'E':[], 'P_r':[]}

for Vi, label in zip(V_batt_values, V_batt_labels):
    r = run_optimisation(v_base, Vi, zeta_base)
    if r is not None:
        ps2['V'].append(Vi)
        ps2['b'].append(r['b'])
        ps2['AR'].append(r['AR'])
        ps2['W0'].append(r['W0'])
        ps2['E'].append(r['E'])
        ps2['P_r'].append(r['P_r'])
        print(f"  {label:>12} | b={r['b']:.3f} m | AR={r['AR']:.2f} "
              f"| W0={r['W0']:.2f} N | E={r['E']:.1f} min | Pr={r['P_r']:.3f} W")
    else:
        print(f"  {label:>12} | NO FEASIBLE SOLUTION")


# ============================================================
# Parametric Study 3: Battery Energy Density
# ============================================================
print("\n" + "=" * 60)
print("  PARAMETRIC STUDY 3: Battery Energy Density")
print("=" * 60)

zeta_values = [36700, 47700, 55000, 73000, 92000]
ps3 = {'zeta':[], 'b':[], 'AR':[], 'W0':[], 'E':[], 'P_r':[]}

for zi in zeta_values:
    r = run_optimisation(v_base, V_batt_base, zi)
    if r is not None:
        ps3['zeta'].append(zi)
        ps3['b'].append(r['b'])
        ps3['AR'].append(r['AR'])
        ps3['W0'].append(r['W0'])
        ps3['E'].append(r['E'])
        ps3['P_r'].append(r['P_r'])
        print(f"  zeta={zi:>6} J/N | b={r['b']:.3f} m | AR={r['AR']:.2f} "
              f"| W0={r['W0']:.2f} N | E={r['E']:.1f} min | Pr={r['P_r']:.3f} W")
    else:
        print(f"  zeta={zi:>6} J/N | NO FEASIBLE SOLUTION")


# ============================================================
# PLOTTING
# ============================================================
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14,
                     "axes.labelsize": 13, "xtick.labelsize": 11,
                     "ytick.labelsize": 11})

# --- PLOT SET 1: Flight Speed ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Parametric Study: Effect of Flight Speed $v$', fontsize=16)

axes[0,0].plot(ps1['v'], ps1['b'], 'b-o', markersize=3)
axes[0,0].set_ylabel('Wingspan $b$ [m]')
axes[0,0].set_title('Optimal Wingspan')
axes[0,0].grid(True, linewidth=0.3)

axes[0,1].plot(ps1['v'], ps1['AR'], 'r-o', markersize=3)
axes[0,1].set_ylabel('Aspect Ratio $AR$')
axes[0,1].set_title('Optimal Aspect Ratio')
axes[0,1].grid(True, linewidth=0.3)

axes[0,2].plot(ps1['v'], ps1['W0'], 'g-o', markersize=3)
axes[0,2].set_ylabel('Total Weight $W_0$ [N]')
axes[0,2].set_title('Optimal Total Weight')
axes[0,2].grid(True, linewidth=0.3)

axes[1,0].plot(ps1['v'], ps1['E'], 'm-o', markersize=3)
axes[1,0].set_xlabel('Flight Speed $v$ [m/s]')
axes[1,0].set_ylabel('Endurance $E$ [min]')
axes[1,0].set_title('Maximum Endurance')
axes[1,0].grid(True, linewidth=0.3)

axes[1,1].plot(ps1['v'], ps1['P_r'], 'k-o', markersize=3)
axes[1,1].set_xlabel('Flight Speed $v$ [m/s]')
axes[1,1].set_ylabel('Required Power $P_r$ [W]')
axes[1,1].set_title('Required Power')
axes[1,1].grid(True, linewidth=0.3)

axes[1,2].axis('off')
for ax in axes[0,:]:
    ax.set_xlabel('Flight Speed $v$ [m/s]')

plt.tight_layout()
plt.savefig('parametric_flight_speed.png', dpi=150)
plt.show()


# --- PLOT SET 2: Battery Voltage ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Parametric Study: Effect of Battery Voltage $V_{batt}$', fontsize=16)

axes[0,0].plot(ps2['V'], ps2['b'], 'b-o', markersize=6)
axes[0,0].set_ylabel('Wingspan $b$ [m]')
axes[0,0].set_title('Optimal Wingspan')
axes[0,0].grid(True, linewidth=0.3)

axes[0,1].plot(ps2['V'], ps2['AR'], 'r-o', markersize=6)
axes[0,1].set_ylabel('Aspect Ratio $AR$')
axes[0,1].set_title('Optimal Aspect Ratio')
axes[0,1].grid(True, linewidth=0.3)

axes[0,2].plot(ps2['V'], ps2['W0'], 'g-o', markersize=6)
axes[0,2].set_ylabel('Total Weight $W_0$ [N]')
axes[0,2].set_title('Optimal Total Weight')
axes[0,2].grid(True, linewidth=0.3)

axes[1,0].plot(ps2['V'], ps2['E'], 'm-o', markersize=6)
axes[1,0].set_xlabel('Battery Voltage $V_{batt}$ [V]')
axes[1,0].set_ylabel('Endurance $E$ [min]')
axes[1,0].set_title('Maximum Endurance')
axes[1,0].grid(True, linewidth=0.3)

axes[1,1].plot(ps2['V'], ps2['P_r'], 'k-o', markersize=6)
axes[1,1].set_xlabel('Battery Voltage $V_{batt}$ [V]')
axes[1,1].set_ylabel('Required Power $P_r$ [W]')
axes[1,1].set_title('Required Power')
axes[1,1].grid(True, linewidth=0.3)

axes[1,2].axis('off')
for ax in axes[0,:]:
    ax.set_xlabel('Battery Voltage $V_{batt}$ [V]')

plt.tight_layout()
plt.savefig('parametric_battery_voltage.png', dpi=150)
plt.show()


# --- PLOT SET 3: Battery Energy Density ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Parametric Study: Effect of Battery Energy Density $\\zeta$', fontsize=16)

zeta_k = [z/1000 for z in ps3['zeta']]  # convert to kJ/N for readability

axes[0,0].plot(zeta_k, ps3['b'], 'b-o', markersize=6)
axes[0,0].set_ylabel('Wingspan $b$ [m]')
axes[0,0].set_title('Optimal Wingspan')
axes[0,0].grid(True, linewidth=0.3)

axes[0,1].plot(zeta_k, ps3['AR'], 'r-o', markersize=6)
axes[0,1].set_ylabel('Aspect Ratio $AR$')
axes[0,1].set_title('Optimal Aspect Ratio')
axes[0,1].grid(True, linewidth=0.3)

axes[0,2].plot(zeta_k, ps3['W0'], 'g-o', markersize=6)
axes[0,2].set_ylabel('Total Weight $W_0$ [N]')
axes[0,2].set_title('Optimal Total Weight')
axes[0,2].grid(True, linewidth=0.3)

axes[1,0].plot(zeta_k, ps3['E'], 'm-o', markersize=6)
axes[1,0].set_xlabel('Energy Density $\\zeta$ [kJ/N]')
axes[1,0].set_ylabel('Endurance $E$ [min]')
axes[1,0].set_title('Maximum Endurance')
axes[1,0].grid(True, linewidth=0.3)

axes[1,1].plot(zeta_k, ps3['P_r'], 'k-o', markersize=6)
axes[1,1].set_xlabel('Energy Density $\\zeta$ [kJ/N]')
axes[1,1].set_ylabel('Required Power $P_r$ [W]')
axes[1,1].set_title('Required Power')
axes[1,1].grid(True, linewidth=0.3)

axes[1,2].axis('off')
for ax in axes[0,:]:
    ax.set_xlabel('Energy Density $\\zeta$ [kJ/N]')

plt.tight_layout()
plt.savefig('parametric_energy_density.png', dpi=150)
plt.show()

print("\nAll parametric plots saved.")
