import numpy as np
from scipy.integrate import solve_ivp
import scipy
import matplotlib.pyplot as plt

# --- Heart-model parameters (eqn 31) ---
Rm  = 0.33      # mitral valve resistance, mmHg*s/l
PAS = 10.0      # atrial source pressure, mmHg
Ra  = 0.60
# --- Arterial-load parameters (eqn 32) ---
CL = 2.35e-4    # l/mmHg
R1 = 60.0       # mmHg*s/l
CR = 1.33e-4    # l/mmHg
R2 = 300.0      # mmHg*s/l
RL = 12640.0    # mmHg*s/l
Rb = 60.0       # mmHg*s/l
RR = 8550.0     # mmHg*s/l
L  = 21.9       # mmHg*s^2/l
RT = R1* Ra + R1*RL + Ra*RL   # total resistance, mmHg*s/l
RG = Rb *(1+R2/(RR+R2))   # total resistance, mmHg*s/l
# ---------- Passive elastance (eqn 18, 30) ----------
h0, h1, h2 = 0.234, 147.0, 106000.0
def E_p(Vv):
    return h2*Vv**2 + h1*Vv + h0
    #return 2+h2*Vv + h1


# ---------- Active elastance (eqn 17, 20-29) ----------
# CAUTION: these amplitude coefficients give a peak E_H far below Fig.5's
# scale (0-8000 mmHg/l) -- verify against the original paper before trusting
# the resulting pressures quantitatively.

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
scale = 100.0   # empirical correction to Ai(Ved) amplitudes to reach physiological pressures
def E_H(Ved, t):
    s = 0.0
    for A, B, C in [(scale*A1(Ved),B1(Ved),C1(Ved)), (scale*A2(Ved),B2(Ved),C2(Ved)), (scale*A3(Ved),B3(Ved),C3(Ved))]:
        s += A * np.exp(-(B**2) * (t - C)**2)
    return s + D


def filling_rhs(t, x):
    x1, x2, x3, x4 = x   # x1=V_CR, x2=Vv, x3=V_CL, x4=flow through L
    dx1 = (-x1 + RR * x4) / (CR * (R2 + RR))
    dx2 = (PAS - E_p(x2)*x2 ) / Rm
    dx3 = (-x3 - RL * x4) / (CL * (R1 + RL))
    dx4 = (-x4 / L) * (Rb + R1*RL/(R1+RL) + R2*RR/(R2+RR)) \
          + RL * x3 / (L * (R1 + RL)) \
          - RR * x1 / (L * (R2 + RR))
    return [dx1, dx2, dx3, dx4]


def ejection(t, x):
    x1, x2, x3, x4 = x   # x1=V_CR, x2=Vv, x3=V_CL, x4=flow through L
    dx1 = (-x1 + RR * x4) / (CR * (R2 + RR))
    dx2 = (-(R1+RL)*E_H(x2,t)*x2-RL*x3-R1*RL*x4) / RT
    dx3 = (RL * E_H(x2,t)*x2-(RL+Ra)*x3-RL*Ra*x4) / (CL * RT)
    dx4 = (-RR* x1 / (L*(R2+RR))) + (R1*RL*E_H(x2,t)*x2+RL*Ra*x3-(R1*RL*Ra+RG*RT)*x4) / (L*RT)
    return [dx1, dx2, dx3, dx4]




def isovolumic(t, x):
    x1, x2, x3, x4 = x   # x1=V_CR, x2=Vv, x3=V_CL, x4=flow through L
    dx1 = (-x1 + RR * x4) / (CR * (R2 + RR))
    dx2 = 0
    dx3 = (-x3-RL*x4) / (CL * (R1 + RL))
    dx4 = (-x4 / L) * (Rb + R1*RL/(R1+RL) + R2*RR/(R2+RR)) \
          + RL * x3 / (L * (R1 + RL)) \
          - RR * x1 / (L * (R2 + RR))
    return [dx1, dx2, dx3, dx4]

def P_A(x3, dx3):
    return dx3 * CL *R1 +x3



#P_A_ejection(sol_ejection.y[2,-1], sol_ejection.y[2,-1])


def aorta_valve_iso_event(t, x):
    x1, x2, x3, x4 = x   # x1=V_CR, x2=Vv, x3=V_CL, x4=flow through L
    dx3 = (-x3-RL*x4) / (CL * (R1 + RL))
    Pv = E_H(x2, t) * x[1]
    return P_A(x3, dx3) - Pv   # nul når Pv netop når PA
aorta_valve_iso_event.terminal = True
aorta_valve_iso_event.direction = -1



def aorta_valve_ejection_event(t, x):
    x1, x2, x3, x4 = x   # x1=V_CR, x2=Vv, x3=V_CL, x4=flow through L
    dx3 = (RL * E_H(x2,t)-(RL+Ra)*x3-RL*Ra*x4) / (CL * RT)
    Pv = E_H(x2, t) * x[1]
    return P_A(x3, dx3) - Pv   # nul når Pv netop når PA
aorta_valve_ejection_event.terminal = True
aorta_valve_ejection_event.direction = -1


def mitral_valve_event(t, x):
    Pv = E_p(x[1]) * x[1]
    #print(Pv)
    return 10- Pv   # nul når Pv netop når PA
mitral_valve_event.terminal = True
mitral_valve_event.direction = -1




record_x1, record_x2, record_x3, record_x4 = [], [], [], []
record_t, record_phase = [], []
def store(sol, phase):
    record_t.append(sol.t)
    record_x1.append(sol.y[0])
    record_x2.append(sol.y[1])
    record_x3.append(sol.y[2])
    record_x4.append(sol.y[3])
    record_phase.append(phase)


t_span = (0, 1)   # typical diastolic filling window
t_eval = np.linspace(*t_span, 2000)
max_step = 1e-3

for i in range(10):
    print(f"Cycle {i+1}")
    # Ejection phase
    if i == 0:
        x0 = [0.80, 0.004, 0.80, 0.002]  # reset to initial state for subsequent cycles
    else:
         x0 = [sol_isovolumic.y[0,-1], sol_isovolumic.y[1,-1], sol_isovolumic.y[2,-1], sol_isovolumic.y[3,-1]]  # final state of filling phase
        

    sol_ejection = solve_ivp(ejection, t_span, x0, max_step=max_step,
                             events=aorta_valve_ejection_event)
    print("Ejection phase completed. Event at t =", sol_ejection.t_events)
    print(sol_ejection.status)
    print(sol_ejection.message)
    store(sol_ejection, "ejection")
    # Isovolumic relaxation phase
    x0 = [sol_ejection.y[0,-1], sol_ejection.y[1,-1], sol_ejection.y[2,-1], sol_ejection.y[3,-1]]  # final state of ejection phase
  
    sol_isovolumic = solve_ivp(isovolumic, t_span, x0, max_step=max_step,
                                events=mitral_valve_event)
    print("Isovolumic relaxation phase completed. Event at t =", sol_isovolumic.t_events)
    store(sol_isovolumic, "relaxation")

    # Filling phase
    x0 = [sol_isovolumic.y[0,-1], sol_isovolumic.y[1,-1], sol_isovolumic.y[2,-1], sol_isovolumic.y[3,-1]]  # final state of filling phase
    sol_filling = solve_ivp(filling_rhs, t_span, x0, 
                             max_step=max_step, events=mitral_valve_event)
    print("Filling phase completed. Event at t =", sol_filling.t_events)
    store(sol_filling, "filling")
    # Isovolumic contraction phase
    x0 = [sol_filling.y[0,-1], sol_filling.y[1,-1], sol_filling.y[2,-1], sol_filling.y[3,-1]]  # final state of filling phase
    sol_isovolumic = solve_ivp(isovolumic, t_span, x0, max_step=max_step,
                                events= aorta_valve_iso_event)
    print("Isovolumic contraction phase completed. Event at t =", sol_isovolumic.t_events)
    store(sol_isovolumic, "contraction")
    

"""

x1 = np.concatenate(record_x1)
x2 = np.concatenate(record_x2)
x3 = np.concatenate(record_x3)
x4 = np.concatenate(record_x4)

t_offset = 0.0
t_segments = []
for t in record_t:
    t_segments.append(t + t_offset)
    t_offset += t[-1]
time = np.concatenate(t_segments)

plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.title("x1: pressure across descending aorta")
plt.xlabel("Time (s)")
plt.ylabel("Pressure (mmHg)")
plt.plot(time, x1, color='#2a78d6')
plt.grid(True, color='#e1e0d9')

plt.subplot(2, 2, 2)
plt.title("x2:  ventricular volume")
plt.xlabel("Time (s)")
plt.ylabel("Volume (L)")
plt.plot(time, x2, color='#2a78d6')
plt.grid(True, color='#e1e0d9')




plt.subplot(2, 2, 3)
plt.title("x3: pressure across ascending aorta")
plt.xlabel("Time (s)")
plt.ylabel("Pressure (mmHg)")
plt.plot(time, x3, color='#2a78d6')
plt.grid(True, color='#e1e0d9')

plt.subplot(2, 2, 4)
plt.title("x4: flow through L")
plt.xlabel("Time (s)")
plt.ylabel("Flow (L/s)")
plt.plot(time, x4, color='#2a78d6')
plt.grid(True, color='#e1e0d9')
plt.tight_layout()
plt.show()4
"""

PHASE_COLORS = {
    "contraction": "#eb6834",
    "ejection":    "#2a78d6",
    "relaxation":  "#e87ba4",
    "filling":     "#1baf7a",
}
PHASE_LABELS = {
    "contraction": "Isovolumisk kontraktion",
    "ejection":    "Ejection",
    "relaxation":  "Isovolumisk relaksation",
    "filling":     "Fyldning",
}

# tidsakse: hvert segment fortsætter hvor det foregående sluttede
t_offset = 0.0
t_segments = []
for t in record_t:
    t_segments.append(t + t_offset)
    t_offset += t[-1]

fig, axs = plt.subplots(2, 2, figsize=(10, 6))

titles  = ["x1: pressure across $C_R$", "x2: ventricular volume",
           "x3: pressure across $C_L$", "x4: flow through $L$"]
ylabels = ["Pressure (mmHg)", "Volume (L)", "Pressure (mmHg)", "Flow (L/s)"]
data    = [record_x1, record_x2, record_x3, record_x4]

for ax, title, ylabel, values in zip(axs.flat, titles, ylabels, data):
    for t, y, phase in zip(t_segments, values, record_phase):
        ax.plot(t, y, color=PHASE_COLORS[phase], lw=1)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e1e0d9")

plt.tight_layout(rect=[0, 0, 1, 0.93])   # efterlad de øverste ~7% til legend

handles = [plt.Line2D([0], [0], color=PHASE_COLORS[p], lw=2, label=PHASE_LABELS[p])
           for p in ["contraction", "ejection", "relaxation", "filling"]]
fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))

plt.show()
plt.show()