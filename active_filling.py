import numpy as np
import matplotlib.pyplot as plt

# fix_factor = 0.1 # could be another number
fix_factor_iso = 0.04 # could be another number
fix_factor_ejection = 0.2 # could be another number

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

# --- Heart-model parameters (eqn 31) ---
Rm  = 12        # mitral valve resistance, mmHg*s/l
PAS = 10.0      # atrial source pressure, mmHg
Ra  = 200       # Changed from 0.60
# --- Arterial-load parameters (eqn 32) ---
CL = 2.35e-4    # l/mmHg
R1 = 60.0       # mmHg*s/l
CR = 1.33e-4    # l/mmHg
R2 = 300.0      # mmHg*s/l
RL = 12640.0    # mmHg*s/l
Rb = 60.0       # mmHg*s/l
RR = 8550.0     # mmHg*s/l
L  = 1000       # mmHg*s^2/l
RT = R1* Ra + R1*RL + Ra*RL   # total resistance, mmHg*s/l
RG = Rb *(1+R2/(RR+R2))   # total resistance, mmHg*s/l

x1[0] = 32.29404549379497
x2[0] = 0.14336558555635703 # l (hypothetical end-diastolisk volume from beat before)
x3[0] = 31.94898291094089
x4[0] = 0.001227387223264677

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
D = 100.0
scale = 100   # empirical correction to Ai(Ved) amplitudes to reach physiological pressures

def E_H(Ved, t):
    s = 0.0
    for A, B, C in [(scale*A1(Ved),B1(Ved),C1(Ved)), (scale*A2(Ved),B2(Ved),C2(Ved)), (scale*A3(Ved),B3(Ved),C3(Ved))]:
        s += A * np.exp(-(B**2) * (t - C)**2)
    return s + D

def Pv_iso(Ved, t):
    return (E_H(Ved, t) * Ved) * fix_factor_iso

def Pv_ej(Ved, t):
    return (E_H(Ved, t) * Ved) * fix_factor_ejection

t = np.linspace(0, T, N_max)

# Ejection phase
for i in range(1,N_max):
    x1[i] = x1[i-1] + t_step * ((-x1[i-1] + RR * x4[i-1]) / (CR * (R2 + RR)))
    x2[i] = x2[i-1] + t_step * ((-(R1+RL)*E_H(x2[i-1],t[i])*x2[i-1]-RL*x3[i-1]-R1*RL*x4[i-1]) / RT)
    x3[i] = x3[i-1] + t_step * ((RL * E_H(x2[i-1],t[i])*x2[i-1]-(RL+Ra)*x3[i-1]-RL*Ra*x4[i-1]) / (CL * RT))
    x4[i] = x4[i-1] + t_step * ((-RR* x1[i-1] / (L*(R2+RR))) + (R1*RL*E_H(x2[i-1],t[i])*x2[i-1]+RL*Ra*x3[i-1]-(R1*RL*Ra+RG*RT)*x4[i-1]) / (L*RT))
    Pv_ejection[i] = Pv_ej(x2[i], t[i])

# Isovolumic phases
for i in range(1,N_max):
    x1[i] = x1[i-1] + t_step * ((-x1[i-1] + RR * x4[i-1]) / (CR * (R2 + RR)))
    x2[i] = x2[i-1] + t_step * 0
    x3[i] = x3[i-1] + t_step * ((-x3[i-1]-RL*x4[i-1]) / (CL * (R1 + RL)))
    x4[i] = x4[i-1] + t_step * ((-x4[i-1] / L) * (Rb + R1*RL/(R1+RL) + R2*RR/(R2+RR)) \
              + RL * x3[i-1] / (L * (R1 + RL)) \
              - RR * x1[i-1] / (L * (R2 + RR)))
    Pv_isovolumic[i] = Pv_iso(x2[i], t[i])


plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t, Pv_ejection)
plt.xlabel('time (s)')
plt.ylabel('Pv (mmHg)')
plt.title('Ejection Phase')
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(t, Pv_isovolumic)
plt.xlabel('time (s)')
plt.ylabel('Pv (mmHg)')
plt.title('Isovolumic Phase')
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(t, x3)
plt.xlabel('time (s)')
plt.ylabel('x3 (mmHg)')
plt.title('Aortic pressure (x3)')
plt.grid()
plt.tight_layout()
plt.show()