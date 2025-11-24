import numpy as np
import math
import wave
import matplotlib.pyplot as plt

# ----------------------
# Params
# ----------------------
fs = 44100               # sample rate [Hz]
Tsec = 5.0               # duration [s]
dt = 1.0 / fs
N = int(Tsec * fs)

# ----------------------
# String geometry/physics
# ----------------------
L = 0.328                # string length [m]
rho_l = 0.0006           # linear density [kg/m]

# *** CHANGE THIS TO SET THE NOTE ***
f1 = 440.0               # fundamental [Hz], e.g., A4 = 440
# f1 = 196.0             # G3
# f1 = 293.66            # D4
# f1 = 659.26            # E5
# f1 = 523.25            # C5
# f1 = 880.0             # A5

# Derived wave speed and tension ensuring f1 = c/(2L)
c = 2 * L * f1
Tension = rho_l * c**2

M_modes = 40             # number of normal modes
zeta = 0.001             # modal damping

# ----------------------
# Bowing parameters
# ----------------------
x_b = 0.15 * L
N_bow = 1.0
mu_s, mu_k = 0.8, 0.3
v_s = 0.02
k_br = 15.0
g_br = 0.005
v_b_ss = 0.15
t_attack = 0.10

# ----------------------
# Bridge/body resonance
# ----------------------
m_b = 0.010
f_bridge = 550.0
k_b = (2*np.pi*f_bridge)**2 * m_b
r_b = 2 * 0.05 * np.sqrt(k_b*m_b)

# ----------------------
# Helmholtz air resonance
# ----------------------
rho_air = 1.2
c_air = 343.0
V = 2.1e-3
S_n = 2.0e-4
L_e = 8.5e-3
m_H = rho_air * S_n * L_e
K_H = rho_air * c_air**2 * (S_n**2) / V
r_H = 2 * 0.15 * np.sqrt(K_H*m_H)

S_eff = 0.015
alpha = 0.6

# ----------------------
# Modal data
# ----------------------
n = np.arange(1, M_modes+1)
omega_n = n * np.pi * c / L
shape_b = np.sin(n * np.pi * x_b / L)
slope_bridge = (n * np.pi / L) * np.cos(n * np.pi)
gain = (2.0 / (rho_l * L)) * shape_b

# ----------------------
# Helpers
# ----------------------
def bow_speed(t):
    return v_b_ss * (t / t_attack) if t < t_attack else v_b_ss

def smooth_sign(x, eps=1e-4):
    return np.tanh(x / eps)

# State vector: q[0:M], qd[M:2M], z, zd, xH, xHd, phi
dim = 2*M_modes + 5

def deriv(t, y):
    q = y[0:M_modes]
    qd = y[M_modes:2*M_modes]
    z, zd, xH, xHd, phi = y[-5:]

    ud_bow = np.dot(qd, shape_b) + (x_b / L) * zd
    vr = ud_bow - bow_speed(t)

    phi_dot = vr - (abs(vr)/g_br)*phi
    mu = mu_k + (mu_s - mu_k)*np.exp(-(abs(vr)/v_s)**2) + k_br*phi
    fb = N_bow * mu * smooth_sign(vr)

    qdd = -2*zeta*omega_n*qd - (omega_n**2)*q + gain * fb

    u_x_L = np.dot(q, slope_bridge) + (1.0/L) * z
    F_str = Tension * u_x_L

    p_c = -(rho_air*c_air**2 / V) * (S_n*xH + S_eff*z)

    zdd = (F_str - S_eff*p_c - r_b*zd - k_b*z) / m_b
    xHdd = (S_n*p_c - r_H*xHd - K_H*xH) / m_H

    dy = np.zeros_like(y)
    dy[0:M_modes] = qd
    dy[M_modes:2*M_modes] = qdd
    dy[-5:] = [zd, zdd, xHd, xHdd, phi_dot]
    return dy

# ----------------------
# RK4 integrator
# ----------------------
def rk4_step(t, y, h):
    k1 = deriv(t, y)
    k2 = deriv(t + 0.5*h, y + 0.5*h*k1)
    k3 = deriv(t + 0.5*h, y + 0.5*h*k2)
    k4 = deriv(t + h, y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ----------------------
# Simulation loop
# ----------------------
y = np.zeros(dim)
audio = np.zeros(N)

for i in range(N):
    t = i * dt
    z, zd, xH, xHd = y[2*M_modes], y[2*M_modes+1], y[2*M_modes+2], y[2*M_modes+3]
    audio[i] = alpha * zd + (1-alpha) * xHd
    y = rk4_step(t, y, dt)

audio -= np.mean(audio)
peak = np.max(np.abs(audio)) + 1e-12
audio_norm = 0.95 * audio / peak
audio_int16 = np.int16(np.clip(audio_norm, -1.0, 1.0) * 32767)

# ----------------------
# Output WAV file
# ----------------------
wav_path = "violin_note.wav"
with wave.open(wav_path, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(fs)
    wf.writeframes(audio_int16.tobytes())

print("WAV saved:", wav_path)