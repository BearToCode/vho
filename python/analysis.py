# Reference:
# Thanapalan, K. (2010). Modelling of a helicopter system. 45-52.
# Paper presented at 1st Virtual Control Conference (VCC) , Aalborg University, Denmark, 2010, Denmark.

import sympy as sp

# = State of the helicopter =======================
u, v, w = sp.symbols("u, v, w", real=True)  # Velocity
p, q, r = sp.symbols("p, q, r", real=True)  # Roll, pitch, yaw rates
theta, phi, psi = sp.symbols("theta, phi, psi", real=True)  # Euler angles

# = Environment ===================================
Psi_w = sp.symbols("Psi_w", real=True)  # Side-slip angle


# = Control inputs ========================================
eta_1s, eta_1c, eta_c, eta_p = sp.symbols("eta_1s, eta_1c, eta_c, eta_p")

# = Air properties ========================================
rho = sp.symbols("rho", real=True, positive=True, nonzero=True)

# = Helicopter properties =================================
# Main rotor speed and radius
Omega, R = sp.symbols("Omega, R", real=True, positive=True)
# Main rotor lift curve and solidity
a_0, s = sp.symbols("a_0, s", real=True)
# Coning angle and first harmonic cyclic flapping angles
beta_0, beta_1cw, beta_2sw = sp.symbols("beta_0, beta_1cw, beta_2sw", real=True)
# Collective gearing constants
g_c0, g_c1 = sp.symbols("g_c0, g_c1", real=True)
# Autostabiliser feedback gain and aircraft normal acceleration increment
k_g, delta_n = sp.symbols("k_g, delta_n", real=True)
# Feedback gains
k_phi, k_p, k_theta, k_q = sp.symbols("k_phi, k_p, k_theta, k_q", real=True)
# Feedforward gains
k_lc, k_ls = sp.symbols("k_lc, k_ls", real=True)
eta_1s0, et_1c0 = sp.symbols("eta_1s0, et_1c0")

# = Derived variables =====================================
# Main rotor collective pitch
theta_0 = ((g_c0 + g_c1 * eta_c) + (k_g * delta_n)) / (1 + tau_c4 * s)
# Blade cyclic pitch components
# fmt: off
theta_1s = ((g_1s0 + g_1s1 * eta_1s + g_sc0 + g_sc1 * eta_c + k_theta * theta + k_q * q + k_1s * (eta_1s - eta_1s0))
            / (1 + eta_c1 * s)) * sp.cos(Psi_F) \
         + ((g_1c0 + g_1c1 * eta_1c + k_phi * phi + k_P * P + k_1c * (eta_1c - eta_1c0))
            / (1 + eta_c2 * s)) * sp.sin(Psi_F)
theta_1c = ((g_1c0 + g_1c1 * eta_1c + k_phi * phi + k_P * P + k_1c * (eta_1c - eta_1c0))
            / (1 + eta_c2 * s)) * sp.cos(Psi_F) \
         - ((g_1s0 + g_1s1 * eta_1s + g_sc0 + g_sc1 * eta_c + k_theta * theta + k_q * q + k_1s * (eta_1s - eta_1s0))
            / (1 + eta_c1 * s)) * sp.sin(Psi_F)
# fmt: on
theta_1sw = sp.cos(Psi_w) * theta_1s + sp.sin(Psi_w) * theta_1c
theta_1cw = sp.sin(Psi_w) * theta_1s - sp.cos(Psi_w) * theta_1c

# = Harmonic components of blade aerodynamic loads ========
# fmt: off
F1_0  = theta_0 * (1 / 3 + mu**2 / 2) + mu / 2 * (theta_1sw + p_w / 2) \
      + (mu_z - lambda_0) / 2 + 1 / 4 * (1 + mu**2) * theta_tw
F1_1s = (alpha_sw + theta_1sw) / 3 + mu * (theta_0 + mu_z - lambda_0 + 2 / 3 * theta_tw)
F1_1c = (alpha_cw + theta_cw) / 3 - (mu * beta_0) / 2
F1_2s =  mu / 2 * (theta_1cw - beta_1sw + (q_w - lambda_1cw) / 2 - mu / beta_0)
F1_2c = -mu / 2 * (theta_1sw + beta_1cw + (p_w - lambda_1sw) / 2 + mu * (theta_0 + theta_tw / 2))
F2_1s =  mu**2 / 2 * beta_0 * beta_1sw + (mu_z - lambda_0 - mu / 4 * beta_1cw) * alpha_sw \
      - mu / 4 + beta_1sw + alpha_cw + beta_0 * (alpha_sw / 3 + mu * (mu_z - lambda_0) - mu**2 / 4 * beta_1cw) \
      + (alpha_sw / 4 + mu / 2 * (mu_z - lambda_0 - (beta_1c * mu) / 4)) * theta_tw \
      + theta_1sw * ((mu_z - lambda_0) / 2 + mu * (3/8 * (p_w - lambda_1sw) + beta_1cw)) \
      + mu / 4 * theta_1cw * ((q_w - lambda_1cw) / 2 - beta_1sw - mu * beta_0) - (delta * mu) / a_0
F2_1c = -2 * beta_0 * mu * (mu_z - lambda_0 - 4/3 * mu * beta_1cw) + (mu_z - lambda_0 - 3/4 * beta_1cw * mu) * alpha_cw \
      - mu / 4 * beta_1sw * alpha_sw + theta_0 * (alpha_cw / 3 - mu / 2 * (beta_0 + mu / 2 * beta_1sw)) \
      + theta_tw * (alpha_cw / 4 - mu * (beta_0 / 3 + mu / 8 * beta_1sw)) \
      + theta_1cw * ((mu_z - lambda_0) / 2 + mu / 4 * ((p_w - lambda_1sw) / 2 - beta_1cw)) \
      + mu / 4 * theta_1sw * ((q_w - lambda_1cw) / 2 - beta_1sw - mu * beta_0)
# fmt: on


# = Aerodynamic properties ================================
# Main rotor aerodynamic coefficients
C_X = sp.cos(Psi_w) * C_XW - sp.sin(Psi_w) * C_YW
C_Y = sp.sin(Psi_w) * C_XW + sp.cos(Psi_w) * C_YW


# = Forces ================================================
# fmt:off
X = 1/2 * rho * (Omega * R)**2 * sp.pi * R**2 * a_0 * s * sp.cos(gamma_s) * (2 * C_X) / (a_0 * s) \
  - 1/2 * (rho * R)**2 * sp.pi * R**2 * a_0 * s * sp.sin(gamma_s) * (2 * C_Z) / (a_0 * s) \
  + 1/2 * rho * (Omega * R)**2 * sp.pi * a_0 * s * S_p * V_F**2 * C_XF(alpha_F)
# fmt: on
