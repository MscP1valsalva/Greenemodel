import numpy as np
import matplotlib.pyplot as plt
import sys

fix_factor_iso = 1
fix_factor_ejection = 1

T = 0.5        # maximum time in seconds
t_step = 0.1e-3  # integration time step in seconds

h0 = 0.234  # mmHg
h1 = 147    # mmHg/l
h2 = 106000 # mmHg/l^2
Pas = 10    # mmHg

N_max = int(T/t_step)
x1 = np.zeros(N_max)
x2 = np.zeros(N_max)
x3 = np.zeros(N_max)
x4 = np.zeros(N_max)
Pv_ejection = np.zeros(N_max)
Pv_isovolumic = np.zeros(N_max)
Pv = np.zeros(N_max)
E = np.zeros(N_max)

# --- Heart-model parameters (eqn 31) ---
Rm  = 12      # mitral valve resistance, mmHg*s/l (altered from 0.33 to 12 to match the correct physiological range)
PAS = 10      # atrial source pressure, mmHg
Ra = 0.6 *10
Fix_factor_passive = 0.0045  

# --- Arterial-load parameters (eqn 32) ---
CL = 2.35e-4 *2.4 # l/mmHg 
R1 = 60 *10   # mmHg*s/l *12
CR = 1.33e-4 *3.6 # l/mmHg
R2 = 300.0 *0.5  # mmHg*s/l
RL = 12640.0*0.22  # mmHg*s/l 
Rb = 60.0    # mmHg*s/l
RR = 8550.0 *0.22  # mmHg*s/l
L  = 21.9       # mmHg*s^2/l
RT = R1* Ra + R1*RL + Ra*RL   # total resistance, mmHg*s/l
RG = Rb+(R2*RR/(RR+R2))   # total resistance, mmHg*s/l (RR changed from Rb)

# Initial values for iso phase (from filling phase)
x1[0] = 33.42402730938285
x2[0] = 0.143365585556357060
x3[0] = 33.039866751437
x4[0] = 0.0014891060914810333
Pv[0] = 10

def A1(Ved): 
    if Ved <= 0.055:
        return 2.1e-3*Ved**2 - 0.2338*Ved + 10.6
    else:
        return -1.0133e-3*Ved**2 - 0.0626*Ved + 10.6
def A2(Ved): return 150.0
def A3(Ved): 
    if Ved <= 0.055:
        return 2.8e-3*Ved**2 - 0.3*Ved + 10.2
    else:
        return 6.476e-4*Ved**2 - 0.1816*Ved + 10.2
def B1(Ved): return -1e-4*Ved + 0.0157
def B2(Ved): return 45.0
def B3(Ved): return -4e-4*Ved + 0.0431
def C1(Ved): return 9.7e-6*Ved**2 - 2.8e-4*Ved + 0.12
def C2(Ved): return 0.06
def C3(Ved): return -40.2e-6*Ved**2 + 4.78e-3*Ved + 0.06
D = -320
scale_A = 9*0.6  # empirical correction to Ai(Ved) amplitudes to reach physiological pressures
scale_B = 0.1
scale_C = 5
def E_H(Ved, t):
    s = 0.0
    for A, B, C in [(A1(Ved),B1(Ved),C1(Ved)), (A2(Ved),B2(Ved),C2(Ved)), (A3(Ved),B3(Ved),C3(Ved))]:
        s += scale_A*A * np.exp(-((scale_B*B)**2) * (t - scale_C*C)**2)
    return s + D


def Pv_iso(Ved, t):
    return (E_H(Ved, t) * Ved) * fix_factor_iso

def Pv_ej(Ved, t):
    return (E_H(Ved, t) * Ved) * fix_factor_ejection

t = np.linspace(0, T, N_max)
phase = np.empty(N_max, dtype=object)
phase[0] = "Iso"

# Ejection phase
# for i in range(1,N_max):
#     x1[i] = x1[i-1] + t_step * ((-x1[i-1] + RR * x4[i-1]) / (CR * (R2 + RR)))
#     x2[i] = x2[i-1] + t_step * ((-(R1+RL)*E_H(x2[i-1],t[i])*x2[i-1]+RL*x3[i-1]-R1*RL*x4[i-1]) / RT)
#     x3[i] = x3[i-1] + t_step * ((RL * E_H(x2[i-1],t[i])*x2[i-1]-(RL+Ra)*x3[i-1]-RL*Ra*x4[i-1]) / (CL * RT))
#     x4[i] = x4[i-1] + t_step * ((-RR* x1[i-1] / (L*(R2+RR))) + (R1*RL*E_H(x2[i-1],t[i])*x2[i-1]+RL*Ra*x3[i-1]-(R1*RL*Ra+RG*RT)*x4[i-1]) / (L*RT))
#     Pv_ejection[i] = Pv_ej(x2[i], t[i])

# Isovolumic phases
# for i in range(1,N_max):
#     x1[i] = x1[i-1] + t_step * ((-x1[i-1] + RR * x4[i-1]) / (CR * (R2 + RR)))
#     x2[i] = x2[i-1] + t_step * 0
#     x3[i] = x3[i-1] + t_step * ((-x3[i-1]-RL*x4[i-1]) / (CL * (R1 + RL)))
#     x4[i] = x4[i-1] + t_step * ((-x4[i-1] / L) * (Rb + R1*RL/(R1+RL) + R2*RR/(R2+RR)) \
#               + RL * x3[i-1] / (L * (R1 + RL)) \
#               - RR * x1[i-1] / (L * (R2 + RR)))
#     Pv_isovolumic[i] = Pv_iso(x2[i], t[i])

for i in range(1,N_max):
    if Pv[i-1] < 80:
        phase[i] = "Iso"
        x1[i] = x1[i-1] + t_step * ((-x1[i-1] + RR * x4[i-1]) / (CR * (R2 + RR)))
        x2[i] = x2[i-1] + t_step * 0
        x3[i] = x3[i-1] + t_step * ((-x3[i-1]-RL*x4[i-1]) / (CL * (R1 + RL)))
        x4[i] = x4[i-1] + t_step * ((-x4[i-1] / L) * (Rb + R1*RL/(R1+RL) + R2*RR/(R2+RR)) \
              + RL * x3[i-1] / (L * (R1 + RL)) \
              - RR * x1[i-1] / (L * (R2 + RR)))
        Pv[i] = Pv_iso(x2[i], t[i])
        E[i] = E_H(x2[i],t[i])
    elif Pv[i-1] >= 80:
        phase[i] = "ejection"
        x1[i] = x1[i-1] + t_step * ((-x1[i-1] + RR * x4[i-1]) / (CR * (R2 + RR)))
        x2[i] = x2[i-1] + t_step * ((-(R1+RL)*E_H(x2[i-1],t[i])*x2[i-1]-RL*x3[i-1]-R1*RL*x4[i-1]) / RT)
        x3[i] = x3[i-1] + t_step * ((RL * E_H(x2[i-1],t[i])*x2[i-1]-(RL+Ra)*x3[i-1]-RL*Ra*x4[i-1]) / (CL * RT))
        x4[i] = x4[i-1] + t_step * ((-RR* x1[i-1] / (L*(R2+RR))) + (R1*RL*E_H(x2[i-1],t[i])*x2[i-1]+RL*Ra*x3[i-1]-(R1*RL*Ra+RG*RT)*x4[i-1]) / (L*RT))
        Pv[i] = Pv_ej(x2[i], t[i])
        E[i] = E_H(x2[i],t[i])

fig, (ax_pv, ax_volume) = plt.subplots(2, 1)
ax_pv.plot(t, Pv)
ax_pv.set_xlabel('time (s)')
ax_pv.set_ylabel('Pv (mmHg)')
ax_pv.set_title('Pv active curve')
ax_pv.grid()
ax_volume.plot(t, E)
ax_volume.set_xlabel('time')
ax_volume.set_ylabel('Elastance')
ax_volume.grid()


# phase_changes = np.flatnonzero(phase[1:] != phase[:-1]) + 1
# for index in phase_changes:
#     ax_pv.axvline(t[index], color='black', linestyle='--', linewidth=1)
#     ax_volume.axvline(t[index], color='black', linestyle='--', linewidth=1)


plt.tight_layout()
plt.show()
