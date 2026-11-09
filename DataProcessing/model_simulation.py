import numpy as np
from scipy.integrate import solve_ivp
import scipy
import matplotlib.pyplot as plt
import  sys

# --- Heart-model parameters (eqn 31) ---
Rm  = 12      # mitral valve resistance, mmHg*s/l (altered from 0.33 to 12 to match the correct physiological range)
PAS = 10.0      # atrial source pressure, mmHg
Ra = 0.6 
Fix_factor_passive = 0.0045    

# --- Arterial-load parameters (eqn 32) ---
CL = 2.35e-4 *0.1  # l/mmHg 
R1 = 60*12       # mmHg*s/l *12
CR = 1.33e-4   # l/mmHg
R2 = 300.0 *2.4     # mmHg*s/l
RL = 12640.0 *2.5    # mmHg*s/l 
Rb = 60.0       # mmHg*s/l
RR = 8550.0 *3    # mmHg*s/l
L  = 21.9      # mmHg*s^2/l
RT = R1* Ra + R1*RL + Ra*RL   # total resistance, mmHg*s/l
RG = Rb+(R2*RR/(RR+R2))   # total resistance, mmHg*s/l (RR changed from Rb)

# ---------- Passive elastance (eqn 18, 30) ----------
h0, h1, h2 = 0.234, 147.0, 106000.0
def E_p(Vv):
    return h2*Vv + h1 + h0/Vv
    #return 2+h2*Vv + h1

def Pv(Vv):
    return (h2*Vv**2 + h1*Vv + h0) * Fix_factor_passive

# ---------- Active elastance (eqn 17, 20-29) ----------
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
D = -180
scale_A = 8  # empirical correction to Ai(Ved) amplitudes to reach physiological pressures
scale_B = 0.15
scale_C = 4
def E_H(Ved, t):
    s = 0.0
    for A, B, C in [(A1(Ved),B1(Ved),C1(Ved)), (A2(Ved),B2(Ved),C2(Ved)), (A3(Ved),B3(Ved),C3(Ved))]:
        s += scale_A*A * np.exp(-((scale_B*B)**2) * (t - scale_C*C)**2)
    return s + D

# --------- Functions for the four phases: ---------

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
    dx2 = (-(R1+RL)*(E_H(Ved,t)*x2)-RL*x3-R1*RL*x4) / RT # The sign before RL should maybe be positive
    dx3 = (RL * (E_H(Ved,t)*x2)-(RL+Ra)*x3-RL*Ra*x4) / (CL * RT)
    dx4 = (-RR* x1 / (L*(R2+RR))) + (R1*RL*(E_H(Ved,t)*x2)+RL*Ra*x3-(R1*RL*Ra+RG*RT)*x4) / (L*RT)
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

# --------- Thresholds for the ends of the four phases ---------

def contraction_end_event(t, x, Ved):
    x1, x2, x3, x4 = x   # x1=V_CR, x2=Vv, x3=V_CL, x4=flow through L
    dx3 = (-x3-RL*x4) / (CL * (R1 + RL))
    Pv_active = (E_H(Ved, t) * x[1])
    return P_A(x3, dx3) - Pv_active   # 0 when Pv reaches P_A

contraction_end_event.terminal = True
contraction_end_event.direction = -1 # Because Pv is increasing during contraction, we want to detect when there is a zero crossing from positive to negative, which corresponds to the moment when Pv becomes greater than PA.


def ejection_end_event(t, x, Ved):
    x1, x2, x3, x4 = x   # x1=V_CR, x2=Vv, x3=V_CL, x4=flow through L
    dx3 = (RL * (E_H(Ved,t)*x2)-(RL+Ra)*x3-RL*Ra*x4) / (CL * RT)
    Pv_active = (E_H(Ved, t) * x[1])
    return P_A(x3, dx3) - Pv_active   # 0 when Pv reaches P_

ejection_end_event.terminal = True
ejection_end_event.direction = +1 # Because Pv is decreasing during ejection, we want to detect when there is a zero crossing from negative to positive, which corresponds to the moment when Pv becomes less than PA. 

def relaxation_end_event(t, x, Ved):
    x1, x2, x3, x4 = x   # x1=V_CR, x2=Vv, x3=V_CL, x4=flow through L
    Pv_active = (E_H(Ved, t) * x[1])
    return 9.9 - Pv_active   # 0 when Pv reaches P_AS

relaxation_end_event.terminal = True
relaxation_end_event.direction = +1 # Because Pv is decreasing during relaxation, we want to detect when there is a zero crossing from negative to positive, which corresponds to the moment when Pv becomes less than 10 mmHg.


def filling_end_event(t, x, Ved):
    # record_Pv.append(Pv(x[1]))
    return 9.9- Pv(x[1])   # 0 when Pv reaches P_AS

filling_end_event.terminal = True
filling_end_event.direction = -1 # Because Pv is increasing during filling, we want to detect when there is a zero crossing from positive to negative, which corresponds to the moment when Pv becomes greater than 10 mmHg.

# --------- Solving the equations ---------

record_x1, record_x2, record_x3, record_x4 = [], [], [], []
record_t, record_phase = [], []
record_Pv, record_PA = [], []

def store(sol, phase, cycle_offset):
    plot_time = sol.t + cycle_offset
    record_t.append(plot_time)
    record_x1.append(sol.y[0])
    record_x2.append(sol.y[1])
    record_x3.append(sol.y[2])
    record_x4.append(sol.y[3])
    record_phase.append(phase)

    if phase == "filling":
        pv_values = Pv(sol.y[1])
        dx3 = (-sol.y[2] - RL * sol.y[3]) / (CL * (R1 + RL))
        pa_values = P_A(sol.y[2], dx3)
    elif phase == "ejection":
        pv_values = E_H(Ved, sol.t) * sol.y[1]
        dx3 = (RL * (E_H(Ved,sol.t)*sol.y[1])-(RL+Ra)*sol.y[2]-RL*Ra*sol.y[3]) / (CL * RT)
        pa_values = P_A(sol.y[2], dx3)
    else:  # contraction eller relaxation
        pv_values = E_H(Ved, sol.t) * sol.y[1]
        dx3 = (-sol.y[2]-RL*sol.y[3]) / (CL * (R1 + RL))
        pa_values = P_A(sol.y[2], dx3)

    record_Pv.append(pv_values)
    record_PA.append(pa_values)


t_span = (0, 1)   # A maximal cutoff for every phase (this should never be reached)
max_step = 1e-3   # The resolution of datapoints
cycle_offset = 0

for i in range(180): # Number of heart cycles

    # Ejection phase
    if i == 0:
        x0 = [85.04, 0.14, 82.02, -0.0017]  # reset to initial state for subsequent cycles
        Ved = 0.14
    else:
         x0 = [sol_filling.y[0,-1], sol_filling.y[1,-1], sol_filling.y[2,-1], sol_filling.y[3,-1]]  # final state of filling phase

    # Isovolumic contraction phase
    sol_isovolumic = solve_ivp(isovolumic, t_span, x0, args=(Ved,), max_step=max_step,
                                events=contraction_end_event)
    store(sol_isovolumic, "contraction", cycle_offset)

    # Ejection phase
    ej_starttime = sol_isovolumic.t[-1]
    ej_span = (ej_starttime, ej_starttime + 1)
    x0 = [sol_isovolumic.y[0,-1], sol_isovolumic.y[1,-1], sol_isovolumic.y[2,-1], sol_isovolumic.y[3,-1]]  # final state of contraction phase
    sol_ejection = solve_ivp(ejection, ej_span, x0, args=(Ved,), max_step=max_step,
                             events=ejection_end_event)
    store(sol_ejection, "ejection", cycle_offset)
    
    # Isovolumic relaxation phase
    isor_starttime = sol_ejection.t[-1]
    isor_span = (isor_starttime, isor_starttime + 1)
    x0 = [sol_ejection.y[0,-1], sol_ejection.y[1,-1], sol_ejection.y[2,-1], sol_ejection.y[3,-1]]  # final state of ejection phase
    sol_isovolumic = solve_ivp(isovolumic, isor_span, x0, args=(Ved,), max_step=max_step,
                                events=relaxation_end_event)
    store(sol_isovolumic, "relaxation", cycle_offset)

    # Filling phase
    x0 = [sol_isovolumic.y[0,-1], sol_isovolumic.y[1,-1], sol_isovolumic.y[2,-1], sol_isovolumic.y[3,-1]]  # final state of filling phase
    fill_starttime = sol_isovolumic.t[-1]
    fill_span = (fill_starttime, fill_starttime +1)
    sol_filling = solve_ivp(filling_rhs, fill_span, x0, args=(Ved,),
                             max_step=max_step, events=filling_end_event)
    Ved = sol_filling.y[1,-1] # End diastolic Volume
    store(sol_filling, "filling", cycle_offset)

    cycle_offset += sol_filling.t[-1]

# Making the time axis
t_segments = []
for t in record_t:
    t_segments.append(t)

