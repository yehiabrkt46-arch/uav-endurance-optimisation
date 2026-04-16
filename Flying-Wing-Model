Electrically Powered Flying Wing Design - Model Functions

Design Variables:
    x = [b, Pm, Cbatt]
    b     : wingspan [m]
    Pm    : motor power [W]
    Cbatt : battery capacity [mAh]

Objective:
    Maximise endurance E [min]  =>  Minimise f(x) = -E

Constraints (all <= 0):
    g1: W0 - Wmax <= 0          (total weight limit)
    g2: Pr - Pa   <= 0          (sufficient power for flight)
    g3: 4 - AR    <= 0          (minimum aspect ratio)
    g4: AR - 16   <= 0          (maximum aspect ratio)

Bounds:
    0.5 <= b <= 6.0       [m]
    1.0 <= Pm <= 10.0     [W]
    200 <= Cbatt <= 12000  [mAh]
"""

import numpy as np

# ============================================================
# Design Parameters (Table 1 from Coursework Brief)
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
# Flying Wing Model: compute all quantities from design variables
# ============================================================
def compute_all(b, Pm, Cbatt, v=v, V_batt=V_batt, zeta=zeta):
    """
    Compute all physical and aerodynamic quantities from the
    three design variables (b, Pm, Cbatt).

    Parameters
    ----------
    b     : wingspan [m]
    Pm    : motor power [W]
    Cbatt : battery capacity [mAh]
    v     : flight speed [m/s] (default from Table 1)
    V_batt: battery voltage [V] (default from Table 1)
    zeta  : battery energy density [J/N] (default from Table 1)

    Returns
    -------
    dict with all intermediate and output quantities.
    """
    # Step 1: Battery energy [J]
    Sigma = 3600.0 * V_batt * Cbatt / 1000.0

    # Step 2: Component weights [N]
    W_batt  = Sigma / zeta
    W_motor = k_w2p * Pm

    # Step 3: Structural weight fraction (depends on wingspan)
    X = 0.5 + 0.05 * b

    # Step 4: Total weight [N]
    W0 = (W_batt + W_motor + W_elec + W_pl) / (1.0 - X)

    # Step 5: Wing planform area from lift = weight [m^2]
    S = 2.0 * W0 / (rho * v**2 * C_L)

    # Step 6: Aspect ratio
    AR = b**2 / S

    # Step 7: Mean aerodynamic chord [m]
    c_bar = S / b

    # Step 8: Reynolds number
    Re = rho * v * c_bar / mu

    # Step 9: Parasite drag coefficient (flat-plate laminar)
    C_D0 = 1.328 / np.sqrt(Re)

    # Step 10: Oswald span efficiency factor (swept wing)
    e = 4.61 * (1.0 - 0.045 * AR**0.68)

    # Step 11: Power required for level flight [W]
    P_r = (0.5 * C_D0 * rho * v**3 * S
           + 2.0 * W0**2 / (np.pi * e * AR * rho * v * S))

    # Step 12: Available power [W]
    P_a = eta_m * Pm

    # Step 13: Endurance [minutes]
    E = (Sigma / P_r) / 60.0

    return {
        'Sigma': Sigma, 'W_batt': W_batt, 'W_motor': W_motor,
        'X': X, 'W0': W0, 'S': S, 'AR': AR, 'c_bar': c_bar,
        'Re': Re, 'C_D0': C_D0, 'e': e,
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
# Constraint functions: g_i(x) <= 0
# ============================================================
def constraints(x):
    """
    Evaluate all four inequality constraints.
    Returns array [g1, g2, g3, g4], all must be <= 0.
    """
    b, Pm, Cbatt = x
    res = compute_all(b, Pm, Cbatt)

    g1 = res['W0'] - W_max          # total weight <= Wmax
    g2 = res['P_r'] - res['P_a']    # required power <= available power
    g3 = 4.0 - res['AR']            # AR >= 4
    g4 = res['AR'] - 16.0           # AR <= 16

    return np.array([g1, g2, g3, g4])


# ============================================================
# Quick verification
# ============================================================
if __name__ == "__main__":
    test_b, test_Pm, test_Cbatt = 3.0, 7.0, 8000.0
    res = compute_all(test_b, test_Pm, test_Cbatt)
    g = constraints([test_b, test_Pm, test_Cbatt])

    print("=" * 55)
    print("  FLYING WING MODEL - VERIFICATION")
    print("=" * 55)
    print(f"  Design: b={test_b} m, Pm={test_Pm} W, Cbatt={test_Cbatt} mAh")
    print("-" * 55)
    print(f"  Battery energy   Sigma = {res['Sigma']:.1f} J")
    print(f"  Total weight     W0    = {res['W0']:.3f} N")
    print(f"  Wing area        S     = {res['S']:.4f} m^2")
    print(f"  Aspect ratio     AR    = {res['AR']:.3f}")
    print(f"  Mean chord       c_bar = {res['c_bar']:.4f} m")
    print(f"  Reynolds number  Re    = {res['Re']:.0f}")
    print(f"  Required power   P_r   = {res['P_r']:.3f} W")
    print(f"  Available power  P_a   = {res['P_a']:.3f} W")
    print(f"  Endurance        E     = {res['E']:.1f} min")
    print("-" * 55)
    labels = ["Weight", "Power", "AR>=4", "AR<=16"]
    for i, (label, gi) in enumerate(zip(labels, g)):
        status = "FEASIBLE" if gi <= 0 else "VIOLATED"
        print(f"  g{i+1} ({label:>6}): {gi:+.3f}  {status}")
    print("=" * 55)
