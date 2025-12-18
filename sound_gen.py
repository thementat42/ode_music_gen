import numpy as np
from scipy.io.wavfile import write
import math

# simulation parameters
fs = 44100              
dt = 1/fs               
duration = 8.0          #sustained G note
steps = int(duration*fs)

# ----------------target note ---------------
f1 = 440     #fundamental freq of D string

# physical parameters for G string
M = 30
L = 0.328
rho_l = 6e-4

# compute tension needed so f1 = (1/(2L)) * sqrt(T/(rho_l*L))
T = (2*L*f1)**2 * (rho_l * L)

xb = 0.1*L

mu_s = 0.8
mu_k = 0.3
vs = 0.1
g = 120.0
k_bristle = 80.0
N_force = 0.1       
vb = 0.001               #slightly lower bow speed to maintain sound against friction

mb = 0.01
kb_br = (2*np.pi*550)**2 * mb
rb = 0.08

Sn = 2.0e-4
V = 2.1e-3
rho_air = 1.2
c_air = 343.0
mH = rho_air*Sn*0.0085
KH = rho_air*(c_air**2)*(Sn**2) / V
rH = 0.15

# precompute modal constants
n = np.arange(1, M+1)
omega_n = n*np.pi*np.sqrt(T/(rho_l*L))/L
Gn = (2/(rho_l*L) * np.sin(n*np.pi*xb/L))
sin_xb = np.sin(n*np.pi*xb/L)
sign_alt = ((-1)**n)
nxpi_L = (n*np.pi/L)

# ODE System
def deriv(y):
    q = y[0:M]
    qd = y[M:2*M]
    zb = y[2*M]
    zbd = y[2*M+1]
    xH = y[2*M+2]
    xHd = y[2*M+3]
    phi = y[2*M+4]

    # bow-string relative velocity
    u_dot = np.dot(qd, sin_xb)
    vr = u_dot - vb

    # friction law
    if vr != 0:
        phi_dot = vr - (abs(vr)/g)*phi
        fb = (mu_k + (mu_s - mu_k)*np.exp(-(abs(vr)/vs)**2) + k_bristle*phi) * np.sign(vr) * N_force
    else:
        phi_dot = 0.0
        fb = 0.0

    # modal accelerations
    qdd = -1.5*0.001*omega_n*qd - omega_n**2*q + Gn*fb

    # string-to-bridge force
    uxL = np.dot(q, nxpi_L*sign_alt)
    Fstr = T * uxL

    # bridge
    zbdd = (Fstr - kb_br*zb - rb*zbd - Sn*xHd)/mb

    # air resonance
    p_c = -rho_air*(c_air**2)/V * (Sn*xH + Sn*zb)
    xHdd = (Sn*p_c - rH*xHd - KH*xH)/mH

    dydt = np.zeros_like(y)
    dydt[0:M] = qd
    dydt[M:2*M] = qdd
    dydt[2*M] = zbd
    dydt[2*M+1] = zbdd
    dydt[2*M+2] = xHd
    dydt[2*M+3] = xHdd
    dydt[2*M+4] = phi_dot
    return dydt

# RK4
def rk4_step(y):
    k1 = deriv(y)
    k2 = deriv(y + 0.5*dt*k1)
    k3 = deriv(y + 0.5*dt*k2)
    k4 = deriv(y + dt*k3)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# optionally, use Euler's method
def euler_step(y):
    # stabilized explicit Euler with substeps and slope limiting
    h = dt
    sub_steps = 8
    y_new = y.copy()
    h_sub = h / sub_steps
    for _ in range(sub_steps):
        k = deriv(y_new)
        # limit extreme slopes and sanitize NaNs/Infs to avoid overflow
        k = np.nan_to_num(k, nan=0.0, posinf=1e6, neginf=-1e6)
        k = np.clip(k, -1e6, 1e6)
        y_new = y_new + h_sub * k
        # clamp bristle state 'phi' to a reasonable range for stability
        y_new[2*M + 4] = float(np.clip(y_new[2*M + 4], -10.0, 10.0))
    return y_new

# optionally, use improved Euler's method
def improved_euler_step(y):
    # Heun's method (improved Euler) with substeps and slope limiting
    h = dt
    sub_steps = 4
    y_new = y.copy()
    h_sub = h / sub_steps
    for _ in range(sub_steps):
        k1 = deriv(y_new)
        k1 = np.nan_to_num(k1, nan=0.0, posinf=1e6, neginf=-1e6)
        k1 = np.clip(k1, -1e6, 1e6)
        y_pred = y_new + h_sub * k1
        k2 = deriv(y_pred)
        k2 = np.nan_to_num(k2, nan=0.0, posinf=1e6, neginf=-1e6)
        k2 = np.clip(k2, -1e6, 1e6)
        y_new = y_new + (h_sub / 2.0) * (k1 + k2)
        # clamp bristle state 'phi'
        y_new[2*M + 4] = float(np.clip(y_new[2*M + 4], -10.0, 10.0))
    return y_new

# run simulation
y = np.zeros(2*M + 5)
output = np.zeros(steps)

for i in range(steps):
    y = improved_euler_step(y)
    zbd = y[2*M+1]
    xHd = y[2*M+3]
    output[i] = 0.7*zbd + 0.3*xHd  # more bridge for warmth

# save
output /= np.max(np.abs(output)) + 1e-12
wav_path = "A4.wav"
write(wav_path, fs, (output*32767).astype(np.int16))
