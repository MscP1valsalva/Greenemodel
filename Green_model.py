import numpy as np
from scipy.integrate import solve_ivp
import scipy
import matplotlib.pyplot as plt
import  sys

# --- Creating a log file for print statements ---
log_file = open("Green_model_log.txt", "w")
sys.stdout = log_file

# --- Heart-model parameters (eqn 31) ---
Rm  = 12      # mitral valve resistance, mmHg*s/l (altered from 0.33 to 12 to match the correct physiological range)
PAS = 10.0      # atrial source pressure, mmHg
Ra  = 24 # Changed from 0.60
Fix_factor_passive = 0.0045    
Fix_factor_ej = 6 # Subject to change
Fix_factor_iso = 6 # Subject to change
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
    return h2*Vv + h1 + h0/Vv
    #return 2+h2*Vv + h1

def Pv(Vv):
    return (h2*Vv**2 + h1*Vv + h0) * Fix_factor_passive

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
D = 100
scale = 1   # empirical correction to Ai(Ved) amplitudes to reach physiological pressures
def E_H(Ved, t):
    s = 0.0
    for A, B, C in [(scale*A1(Ved),B1(Ved),C1(Ved)), (scale*A2(Ved),B2(Ved),C2(Ved)), (scale*A3(Ved),B3(Ved),C3(Ved))]:
        s += A * np.exp(-((B)**2) * (t - C)**2)
    return s + D

# Functions for the four phases:

def filling_rhs(t, x, Ved):
    x1, x2, x3, x4 = x   # x1=V_CR, x2=Vv, x3=V_CL, x4=flow through L
    dx1 = (-x1 + RR * x4) / (CR * (R2 + RR))
    dx2 = (PAS - Pv(x2) ) / Rm
    dx3 = (-x3 - RL * x4) / (CL * (R1 + RL))
    dx4 = (-x4 / L) * (Rb + R1*RL/(R1+RL) + R2*RR/(R2+RR)) \
          + RL * x3 / (L * (R1 + RL)) \
          - RR * x1 / (L * (R2 + RR))
    return [dx1, dx2, dx3, dx4]

def ejection(t, x, Ved):
    x1, x2, x3, x4 = x   # x1=V_CR, x2=Vv, x3=V_CL, x4=flow through L
    dx1 = (-x1 + RR * x4) / (CR * (R2 + RR))
    dx2 = (-(R1+RL)*(E_H(Ved,t)*x2)*Fix_factor_ej-RL*x3-R1*RL*x4) / RT
    dx3 = (RL * (E_H(Ved,t)*x2)*Fix_factor_ej-(RL+Ra)*x3-RL*Ra*x4) / (CL * RT)
    dx4 = (-RR* x1 / (L*(R2+RR))) + (R1*RL*(E_H(Ved,t)*x2)*Fix_factor_ej+RL*Ra*x3-(R1*RL*Ra+RG*RT)*x4) / (L*RT)
    return [dx1, dx2, dx3, dx4]

def isovolumic(t, x, Ved):
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


def contraction_end_event(t, x, Ved):
    x1, x2, x3, x4 = x   # x1=V_CR, x2=Vv, x3=V_CL, x4=flow through L
    dx3 = (-x3-RL*x4) / (CL * (R1 + RL))
    Pv_active = (E_H(Ved, t) * x[1]) * Fix_factor_iso
    print("Contraction phase: Pv_active =", Pv_active, "P_A =", P_A(x3, dx3), "x1 =", x1, "x2 =", x2, "x3 =", x3, "x4 =", x4)
    return P_A(x3, dx3) - Pv_active   # nul når Pv netop når x3
    # return 80 - Pv_active
contraction_end_event.terminal = True
contraction_end_event.direction = -1 # Because Pv is increasing during contraction, we want to detect when there is a zero crossing from positive to negative, which corresponds to the moment when Pv becomes greater than PA.


def ejection_end_event(t, x, Ved):
    x1, x2, x3, x4 = x   # x1=V_CR, x2=Vv, x3=V_CL, x4=flow through L
    dx3 = (RL * E_H(x2,t)-(RL+Ra)*x3-RL*Ra*x4) / (CL * RT)
    Pv_active = (E_H(Ved, t) * x[1]) * Fix_factor_ej
    print("Ejection phase: Pv_active =", Pv_active, "P_A =", P_A(x3, dx3), "x1 =", x1, "x2 =", x2, "x3 =", x3, "x4 =", x4)
    return P_A(x3, dx3) - Pv_active   # nul når Pv netop når x3
    # return 80 - Pv_active
ejection_end_event.terminal = True
ejection_end_event.direction = +1 # Because Pv is decreasing during ejection, we want to detect when there is a zero crossing from negative to positive, which corresponds to the moment when Pv becomes less than PA. 

def relaxation_end_event(t, x, Ved):
    x1, x2, x3, x4 = x   # x1=V_CR, x2=Vv, x3=V_CL, x4=flow through L
    # dx3 = (-x3-RL*x4) / (CL * (R1 + RL))
    Pv_active = (E_H(Ved, t) * x[1]) * Fix_factor_iso
    print("Relaxation phase: Pv_active =", Pv_active, "x1 =", x1, "x2 =", x2, "x3 =", x3, "x4 =", x4)
    return 9.9 - Pv_active   # nul når Pv netop når PA
relaxation_end_event.terminal = True
relaxation_end_event.direction = +1 # Because Pv is decreasing during relaxation, we want to detect when there is a zero crossing from negative to positive, which corresponds to the moment when Pv becomes less than 10 mmHg.


def filling_end_event(t, x, Ved):
    print("Filling phase: Pv_passive =", Pv(x[1]), "x1 =", x[0], "x2 =", x[1], "x3 =", x[2], "x4 =", x[3])
    return 9.9- Pv(x[1])   # nul når Pv netop når PA
filling_end_event.terminal = True
filling_end_event.direction = -1 # Because Pv is increasing during filling, we want to detect when there is a zero crossing from positive to negative, which corresponds to the moment when Pv becomes greater than 10 mmHg.



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
        Ved = 0.14
    else:
         x0 = [sol_isovolumic.y[0,-1], sol_isovolumic.y[1,-1], sol_isovolumic.y[2,-1], sol_isovolumic.y[3,-1]]  # final state of filling phase
        

    sol_ejection = solve_ivp(ejection, t_span, x0, args=(Ved,), max_step=max_step,
                             events=ejection_end_event)
    print("Ejection phase completed. Event at t =", sol_ejection.t_events)
    print(sol_ejection.status)
    # print(sol_ejection.message)
    store(sol_ejection, "ejection")
    # Isovolumic relaxation phase
    x0 = [sol_ejection.y[0,-1], sol_ejection.y[1,-1], sol_ejection.y[2,-1], sol_ejection.y[3,-1]]  # final state of ejection phase
  
    sol_isovolumic = solve_ivp(isovolumic, t_span, x0, args=(Ved,), max_step=max_step,
                                events=relaxation_end_event)
    print("Isovolumic relaxation phase completed. Event at t =", sol_isovolumic.t_events)
    print(sol_isovolumic.status)
    store(sol_isovolumic, "relaxation")

    # Filling phase
    x0 = [sol_isovolumic.y[0,-1], sol_isovolumic.y[1,-1], sol_isovolumic.y[2,-1], sol_isovolumic.y[3,-1]]  # final state of filling phase
    sol_filling = solve_ivp(filling_rhs, t_span, x0, args=(Ved,),
                             max_step=max_step, events=filling_end_event)
    print("Filling phase completed. Event at t =", sol_filling.t_events)
    print(sol_filling.status)
    store(sol_filling, "filling")
    # Isovolumic contraction phase
    x0 = [sol_filling.y[0,-1], sol_filling.y[1,-1], sol_filling.y[2,-1], sol_filling.y[3,-1]]  # final state of filling phase
    Ved = sol_filling.y[1,-1] # End diastolic Volume
    sol_isovolumic = solve_ivp(isovolumic, t_span, x0, args=(Ved,), max_step=max_step,
                                events=contraction_end_event)
    print("Isovolumic contraction phase completed. Event at t =", sol_isovolumic.t_events)
    print(sol_isovolumic.status)
    store(sol_isovolumic, "contraction")
    


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

titles  = ["x1: pressure across $C_R$ (descending & abdominal aorta)", "x2: ventricular volume",
           "x3: pressure across $C_L$ (ascending aorta)", "x4: flow through $L$"]
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

sys.stdout = sys.__stdout__  # Reset standard output to console
log_file.close()  # Close the log file