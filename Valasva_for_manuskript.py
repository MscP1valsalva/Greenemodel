import numpy as np
from scipy.integrate import solve_ivp
import scipy
import matplotlib.pyplot as plt
import  sys

print("Starting...")

# --- Creating a log file for print statements ---
log_file = open("Green_model_log.txt", "w")
sys.stdout = log_file

# --- Heart-model parameters (eqn 31) ---
Rm  = 12    # mitral valve resistance, mmHg*s/l (altered from 0.33 to 12 to match the correct physiological range)
PAS_L=5
def PAS_R(t):
    tau = 2.0

    if t < 15:
        return 5.0

    elif t <= 30:
        # Normaliseret eksponentielt fald
        exp_now = np.exp(-(t - 15) / tau)
        exp_end = np.exp(-(30 - 15) / tau)

        return 10.0 + (15.0 - 10.0) * (
            (exp_now - exp_end) / (1.0 - exp_end)
        )

    else:
        return 5.0

Ra = 0.6*20
Fix_factor_passive = 0.0045    

def ITP(t_global):
    u=np.where((t_global>=15)&(t_global<=30),40,0)
    return u

# --- Arterial-load parameters (eqn 32) ---
CL = 2.35e-4 * 5 # the lower it is, the more pressure in the ascending aorta, but it reduces stroke volume
R1 = 60 *3# sets the height of the notch, and also the height of the curve.
#R2=R1
CR = 1.33e-4 *8# the lower it is, the more pressure in the descending aorta, but it reduces stroke volume
R2 = 300.0*0.1 # a high value removes the notch, increases amplitude, lowers the cutting point. A 0.x value raises the cutting point, but also the pressure
RL = 12640.0 * 0.2 # the lower it is, the more volume ejected per beat.
Rb = 60.0 # if it increases, the pressure rises but the cut points do not.
#RR = 8550.0 * 0.5 # the lower it is, the more volume ejected per beat
#RR=1710*0.8 #use for RR part later maybe
#RL=RR+1641
L  = 21.9
RT = R1* Ra + R1*RL + Ra*RL   # total resistance, mmHg*s/l
#RG = Rb+(R2*RR/(RR+R2))   # total resistance, mmHg*s/l (RR changed from Rb)
# when RL is larger than RR, the end of the notch moves up.
# ---------- Passive elastance (eqn 18, 30) ----------
h0, h1, h2 = 0.234, 147.0, 106000.0
def E_p(Vv):
    return h2*Vv + h1 + h0/Vv
    #return 2+h2*Vv + h1

def Pv(t_global, Vv):
    return ((h2*Vv**2 + h1*Vv + h0)+ITP(t_global)) * Fix_factor_passive

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
D = -150
scale_A = 6.5
scale_B = 0.4
scale_C = 5.5
 
def E_H(Ved, t):
    fs = (N_ursino(P_reflex) - f_min)/(f_max - f_min) # from 0 to 1 
    afs = 1 + 0.5*fs - 0.35*fs**2
    bfs = 1 + 0.1*fs - 0.1*fs**2
    
    s = 0.0
    for A, B, C in [(A1(Ved),B1(Ved),C1(Ved)), (A2(Ved),B2(Ved),C2(Ved)), (A3(Ved),B3(Ved),C3(Ved))]:
        s += afs*scale_A*A * np.exp(-((scale_B*B)**2) * (t*bfs - scale_C*C)**2)
    return s + D

# --------- Functions for the four phases: ---------

def filling_rhs(t, x, Ved,cycle_offset):
    x1, x2, x3, x4 = x   # x1=V_CR, x2=Vv, x3=V_CL, x4=flow through L
    dx1 = (-x1 + RR * x4) / (CR * (R2 + RR))
    dx2 = (PAS_L+PAS_R(t+cycle_offset) - Pv(t+cycle_offset, x2) ) / Rm
    dx3 = (-x3 - RL * x4) / (CL * (R1 + RL))
    dx4 = (-x4 / L) * (Rb + R1*RL/(R1+RL) + R2*RR/(R2+RR)) \
          + RL * x3 / (L * (R1 + RL)) \
          - RR * x1 / (L * (R2 + RR))
    return [dx1, dx2, dx3, dx4]

def ejection(t, x, Ved, cycle_offset):
    x1, x2, x3, x4 = x   # x1=V_CR, x2=Vv, x3=V_CL, x4=flow through L
    dx1 = (-x1 + RR * x4) / (CR * (R2 + RR))
    dx2 = (-(R1+RL)*((E_H(Ved,t)*x2)+ITP(t+cycle_offset))+RL*x3-R1*RL*x4) / RT # The sign before RL should maybe be positive
    dx3 = (RL * (E_H(Ved,t)*x2+ITP(t+cycle_offset))-(RL+Ra)*x3-RL*Ra*x4) / (CL * RT)
    dx4 = (-RR* x1 / (L*(R2+RR))) + (R1*RL*((E_H(Ved,t)*x2)+ITP(t+cycle_offset))+RL*Ra*x3-(R1*RL*Ra+RG*RT)*x4) / (L*RT)
    return [dx1, dx2, dx3, dx4]

def isovolumic(t, x, Ved, cycle_offset):
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

def contraction_end_event(t, x, Ved, cycle_offset):
    x1, x2, x3, x4 = x   # x1=V_CR, x2=Vv, x3=V_CL, x4=flow through L
    dx3 = (-x3-RL*x4) / (CL * (R1 + RL))
    Pv_active = (E_H(Ved, t) * x[1]+ITP(t+cycle_offset))
    print("Contraction phase: Pv_active =", Pv_active, "P_A =", P_A(x3, dx3), "x1 =", x1, "x2 =", x2, "x3 =", x3, "x4 =", x4)
    return P_A(x3, dx3) - Pv_active   # 0 when Pv reaches P_A
contraction_end_event.terminal = True
contraction_end_event.direction = -1 # Because Pv is increasing during contraction, we want to detect when there is a zero crossing from positive to negative, which corresponds to the moment when Pv becomes greater than PA.


def ejection_end_event(t, x, Ved, cycle_offset):
    x1, x2, x3, x4 = x   # x1=V_CR, x2=Vv, x3=V_CL, x4=flow through L
    dx3 = (RL * (E_H(Ved,t)*x2+ITP(t+cycle_offset))-(RL+Ra)*x3-RL*Ra*x4) / (CL * RT)
    Pv_active = (E_H(Ved, t) * x[1]+ITP(t+cycle_offset))
    print("Ejection phase: Pv_active =", Pv_active, "P_A =", P_A(x3, dx3), "x1 =", x1, "x2 =", x2, "x3 =", x3, "x4 =", x4)
    return P_A(x3, dx3) - Pv_active   # 0 when Pv reaches P_
ejection_end_event.terminal = True
ejection_end_event.direction = +1 # Because Pv is decreasing during ejection, we want to detect when there is a zero crossing from negative to positive, which corresponds to the moment when Pv becomes less than PA. 


def relaxation_end_event(t, x, Ved, cycle_offset):
    x1, x2, x3, x4 = x   # x1=V_CR, x2=Vv, x3=V_CL, x4=flow through L
    Pv_active = (E_H(Ved, t) * x[1]+ITP(t+cycle_offset))
    print("Relaxation phase: Pv_active =", Pv_active, "x1 =", x1, "x2 =", x2, "x3 =", x3, "x4 =", x4)
    return (PAS_L+PAS_R(t+cycle_offset) - 0.1) - Pv_active   # 0 when Pv reaches P_AS. NEW: was 9.9; now follows PAS so it still works when PAS is lowered
relaxation_end_event.terminal = True
relaxation_end_event.direction = +1 # Because Pv is decreasing during relaxation, we want to detect when there is a zero crossing from negative to positive, which corresponds to the moment when Pv becomes less than 10 mmHg.


# ======================================================================
# NEW: Ursino (1998) baroreflex sets the heart period- this is all kind of based on Ursino's equations which are all listed
# ----------------------------------------------------------------------
# ======================================================================
Pn, ka, f_min, f_max = 92, 11.758, 2.52, 47.78              # carotid afferent  NEW
# NEW: Pn moved from Ursino's 92 mmHg to this model's resting mean pressure at RB
#      (74.7 mmHg, from the run without the reflex). when we change the baseline we can then change the values 🙂
f_es_inf, f_es0, k_es, f_es_min = 2.10, 16.11, 0.0675, 2.66   # sympathetic  NEW from table 3 
f_ev0, f_ev_inf, k_ev, f_cs0 = 3.2, 6.3, 7.06, 25.0           # vagal  NEW from table 3
G_Ts, G_Tv, T0 = -0.13, 0.15, 0.58                            # heart period  NEW from table 3

def N_ursino(P):
    x = (P - Pn) / ka

    # Stop programmet ved ugyldige værdier
    if not np.isfinite(x):
        raise ValueError(
            f"Ugyldig værdi i N_ursino: "
            f"P={P}, Pn={Pn}, ka={ka}, x={x}"
        )

    # Begræns eksponenten, så np.exp ikke giver overflow
    x = np.clip(x, -50.0, 50.0)

    e = np.exp(x)

    return (f_min + f_max * e) / (1.0 + e)

def sym_ursino(P):                          # Eq. 33: sympathetic firing  NEW
    return f_es_inf + (f_es0 - f_es_inf)*np.exp(-k_es*N_ursino(P))  # NEW

def para_ursino(P):                         # Eq. 34: vagal firing  NEW
    e = np.exp((N_ursino(P) - f_cs0)/k_ev)  # NEW
    return (f_ev0 + f_ev_inf*e)/(1 + e)  # NEW

def heart_period(P_sym,P_vag):                        # Eqs. 38-42, static  NEW
    if sym_ursino(P_sym) >= f_es_min:
        dT_s = G_Ts * np.log(
            sym_ursino(P_sym) - f_es_min + 1.0
        )
    else:
        dT_s = 0.0
    
    dT_v = G_Tv*para_ursino(P_vag)                                        # vagal: lengthens  NEW
    return T0 + dT_s + dT_v  # NEW

_trapz = getattr(np, "trapezoid", None) or np.trapz  # NEW
T_beat = heart_period(Pn,Pn)       # NEW: heart period for the first beat (resting value)

#------------------------------------------------------------------------
# NEW ADDITION: Ursino (1998) peripheral-resistance reflex, Eqs. 35-37, Table 3
# ----------------------------------------------------------------------
# Some definitions 

G_Rsp, Rsp0_u = 0.695, 2.49        # splanchnic gain and resistance without innervation, mmHg s/ml  NEW ADDITION
G_Rep, Rep0_u = 0.53, 0.78         # extrasplanchnic gain and resistance without innervation, mmHg s/ml  NEW ADDITION
tau_R, D_R = 6.0, 2.0              # time constant and pure delay, s  NEW ADDITION

def sigma_R(G, fes):                                           # NEW ADDITION: Eq. 35 static log curve
    return G*np.log(np.maximum(fes - f_es_min, 0) + 1)         # NEW ADDITION

def TPR_u(dRsp, dRep):                                         # NEW ADDITION: two beds in parallel; have to solve the parallel circuit 
    Rsp, Rep = Rsp0_u + dRsp, Rep0_u + dRep                    # NEW ADDITION
    return Rsp*Rep/(Rsp + Rep)                                 # NEW ADDITION

fes_ref = sym_ursino(Pn)                                       # NEW ADDITION: sympathetic firing at the set point
dRsp, dRep = sigma_R(G_Rsp, fes_ref), sigma_R(G_Rep, fes_ref)  # NEW ADDITION: start at the resting values
TPR_ref = TPR_u(dRsp, dRep)                                    # NEW ADDITION: resting Ursino TPR
RR_base = 8550*0.2                                                  # NEW ADDITION: Greene's resting RR
kR = 1.0                                                       # NEW ADDITION: RR scale factor used by the next beat
fes_hist_t, fes_hist = [0.0], [fes_ref]                        # NEW ADDITION: sympathetic firing history for the delay

def fes_delayed(t, D):                                         # NEW ADDITION: f_es(t - D), resting value before t = 0
    return np.interp(t - D, fes_hist_t, fes_hist, left=fes_ref)  # NEW ADDITION

def lag(Delta, s, T, tau):                                     # NEW ADDITION: exact first-order step over one beat
    return s + (Delta - s)*np.exp(-T/tau)                      # NEW ADDITION


# --------- Solving the equations ---------

record_x1, record_x2, record_x3, record_x4 = [], [], [], []
record_t, record_phase = [], []
record_Pv, record_PA = [], []
record_PB,record_Pdistal = [],[]
record_P_sym, record_P_vag = [], []  # NEW: sympathetic and vagal firing rates
record_T_beat, record_meanPB, record_beat_start = [], [], []  # NEW
record_RR = []  # NEW ADDITION

def store(sol, phase, cycle_offset):
    plot_time = sol.t + cycle_offset
    record_t.append(plot_time)
    record_x1.append(sol.y[0])
    record_x2.append(sol.y[1])
    record_x3.append(sol.y[2])
    record_x4.append(sol.y[3])
    record_phase.append(phase)

    if phase == "filling":
        pv_values = Pv(plot_time, sol.y[1])
        dx3 = (-sol.y[2] - RL * sol.y[3]) / (CL * (R1 + RL))
        pa_values = P_A(sol.y[2], dx3)
    elif phase == "ejection":
        pv_values = E_H(Ved, sol.t) * sol.y[1]+ITP(sol.t+cycle_offset)
        dx3 = (RL * (E_H(Ved,sol.t)*sol.y[1]+ITP(sol.t+cycle_offset))-(RL+Ra)*sol.y[2]-RL*Ra*sol.y[3]) / (CL * RT)
        pa_values = P_A(sol.y[2], dx3)
    else:  # contraction or relaxation
        pv_values = E_H(Ved, sol.t) * sol.y[1]+ITP(sol.t+cycle_offset)
        dx3 = (-sol.y[2]-RL*sol.y[3]) / (CL * (R1 + RL))
        pa_values = P_A(sol.y[2], dx3)
    ## pressure across RB
    delta_PB = Rb * sol.y[3]
    Pb = pa_values - delta_PB
    record_PB.append(Pb)
    record_Pv.append(pv_values)
    record_PA.append(pa_values)


t_span = (0, 1)   # A maximal cutoff for every phase (this should never be reached)
max_step = 1e-3   # The resolution of datapoints
cycle_offset = 0

# NEW: perturbation. PAS (venous return) is lowered for a block of beats,
#      like the fall in venous return in Valsalva phase II.
#PAS_base, PAS_low = 10.0, 6.0          # mmHg  NEW
#pert_beats = range(8, 18)              # beats 9-18 (0-based index)  NEW
n_beats = 50  # NEW
record_PAS = []  # NEW
P_reflex = Pn
for i in range(n_beats): # Number of heart cycles  NEW
    print(f"Cycle {i+1}")
    #PAS = PAS_low if i in pert_beats else PAS_base   # NEW
    PAS=PAS_L+PAS_R(cycle_offset)
    record_PAS.append(PAS)  # NEW

    RR = RR_base*kR                     # NEW ADDITION: reflex-scaled peripheral resistance for this beat
    RG = Rb + (R2*RR/(RR + R2))         # NEW ADDITION: RG depends on RR, so recompute it
    record_RR.append(RR)                # NEW ADDITION: just a record function:)  
    

    # Ejection phase
    if i == 0:
        #x0 = [85.04, 0.14, 82.02, -0.0017] 
        x0 = [70, 0.014, 70, -0.000017]  # reset to initial state for subsequent cycles
        Ved = 0.14
    else:
         x0 = [sol_filling.y[0,-1], sol_filling.y[1,-1], sol_filling.y[2,-1], sol_filling.y[3,-1]]  # final state of filling phase

    # Isovolumic contraction phase
    print("Ved: ",Ved)
    sol_isovolumic = solve_ivp(isovolumic, t_span, x0, args=(Ved,cycle_offset), max_step=max_step,
                                events=contraction_end_event)
    print("Isovolumic contraction phase completed at t =", sol_isovolumic.t[-1])
    print(sol_isovolumic.status)
    store(sol_isovolumic, "contraction", cycle_offset)

    # Ejection phase
    ej_starttime = sol_isovolumic.t[-1]
    ej_span = (ej_starttime, ej_starttime + 1)
    x0 = [sol_isovolumic.y[0,-1], sol_isovolumic.y[1,-1], sol_isovolumic.y[2,-1], sol_isovolumic.y[3,-1]]  # final state of contraction phase
    sol_ejection = solve_ivp(ejection, ej_span, x0, args=(Ved,cycle_offset), max_step=max_step,
                             events=ejection_end_event)
    print("Ejection phase completed at t =", sol_ejection.t[-1])
    print(sol_ejection.status)
    print("RR",RR)
    store(sol_ejection, "ejection", cycle_offset)
    
    # Isovolumic relaxation phase
    isor_starttime = sol_ejection.t[-1]
    isor_span = (isor_starttime, isor_starttime + 0.25)
    x0 = [sol_ejection.y[0,-1], sol_ejection.y[1,-1], sol_ejection.y[2,-1], sol_ejection.y[3,-1]]  # final state of ejection phase
    sol_isovolumic = solve_ivp(isovolumic, isor_span, x0, args=(Ved,cycle_offset), max_step=max_step,
                                events=relaxation_end_event)
    print("Isovolumic relaxation phase completed at t =", sol_isovolumic.t[-1])
    print(sol_isovolumic.status)
    store(sol_isovolumic, "relaxation", cycle_offset)

    # Filling phase
    x0 = [sol_isovolumic.y[0,-1], sol_isovolumic.y[1,-1], sol_isovolumic.y[2,-1], sol_isovolumic.y[3,-1]]  # final state of filling phase
    fill_starttime = sol_isovolumic.t[-1]
    t_max=T_beat-fill_starttime
    if t_max < 0.01:
        t_max = 0.01
    # NEW: the beat ends at the heart period T_beat set by the reflex,
    #      instead of when Pv reaches 9.9 mmHg (filling_end_event no longer used
    fill_span = (fill_starttime, fill_starttime + t_max)  # NEW
    sol_filling = solve_ivp(filling_rhs, fill_span, x0, args=(Ved,  cycle_offset),
                             max_step=max_step)  # NEW
    Ved = sol_filling.y[1,-1] # End diastolic Volume
    print("Filling phase completed at t =", sol_filling.t[-1])
    print(sol_filling.status)
    store(sol_filling, "filling", cycle_offset)

    # NEW: reflex. Mean pressure at RB over this beat sets the next heart period.
    tt = np.concatenate(record_t[-4:]); pp = np.concatenate(record_PB[-4:])  # NEW
    meanPB = _trapz(pp, tt)/(tt[-1] - tt[0])  # NEW
    P_reflex = meanPB
    record_beat_start.append(cycle_offset)  # NEW 
    record_T_beat.append(sol_filling.t[-1])  # NEW
    record_meanPB.append(meanPB)  # NEW
    if len(record_meanPB) >= 4:  # make sure we have at least 4 beats to compute the sympathetic firing rate
        P_sym = record_meanPB[-4]
    else:
        P_sym = Pn
    P_vag = meanPB # calulate vagal firing rate based on the last beat 
    record_P_sym.append(P_sym)
    record_P_vag.append(P_vag)
    T_beat = heart_period(P_sym,P_vag)  # NEW
    print(f"Reflex: mean P_RB = {meanPB:.1f} mmHg -> next T = {T_beat:.3f} s ({60/T_beat:.1f} bpm)")  # NEW

    fes_hist_t.append(cycle_offset+sol_filling.t[-1]); fes_hist.append(sym_ursino(meanPB))       # NEW ADDITION: store this beat's firing
    fes_D = fes_delayed(cycle_offset+sol_filling.t[-1], D_R)                                     # NEW ADDITION: firing D_R seconds ago
    dRsp = lag(dRsp, sigma_R(G_Rsp, fes_D), sol_filling.t[-1], tau_R)              # NEW ADDITION: 
    dRep = lag(dRep, sigma_R(G_Rep, fes_D), sol_filling.t[-1], tau_R)              # NEW ADDITION: 
    kR = TPR_u(dRsp, dRep)/TPR_ref
    cycle_offset += sol_filling.t[-1]
    
# --------- Configurations for the plot ---------

PHASE_COLORS = {
    "contraction": "#eb6834",
    "ejection":    "#2a78d6",
    "relaxation":  "#e87ba4",
    "filling":     "#1baf7a",
}
PHASE_LABELS = {
    "contraction": "Isovolumic contraction",
    "ejection":    "Ejection",
    "relaxation":  "Isovolumic relaxation",
    "filling":     "Filling",
}

# Making the time axis
t_segments = []
for t in record_t:
    t_segments.append(t)

# --------- Plotting x1-x4 ---------

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

plt.tight_layout(rect=[0, 0, 1, 0.93]) 

handles = [plt.Line2D([0], [0], color=PHASE_COLORS[p], lw=2, label=PHASE_LABELS[p])
           for p in ["contraction", "ejection", "relaxation", "filling"]]
fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))

# --------- Plotting the Ventricular and Aortic pressures ---------

fig_pressure, ax = plt.subplots(figsize=(10, 6))

for t, pv_values, pa_values, phase in zip(
    t_segments, record_Pv, record_PA, record_phase
):
    ax.plot(t, pa_values, "r--", lw=1.5, zorder=3)
    ax.plot(t, pv_values, color=PHASE_COLORS[phase], lw=1)
    ax.plot(t, [PAS_R(x)+PAS_L for x in t], "k-", lw=1.5, zorder=2)

ax.set_title("Ventricular and aortic pressure")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Pressure [mmHg]")
ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.9])

handles = [
    plt.Line2D([0], [0], color=PHASE_COLORS[p], lw=2, label=PHASE_LABELS[p])
    for p in ["contraction", "ejection", "relaxation", "filling"]
]
handles.append(plt.Line2D([0], [0], color="red", lw=2, label="Aortic pressure"))
handles.append(plt.Line2D([0], [0], color="black", lw=2, label="PAS_R"))
# ax.legend(handles=handles)
fig_pressure.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0))
              
# --------- Plotting the P-V diagram ---------

fig_pv, ax = plt.subplots(figsize=(10, 6))

for x2_values, pv_values, phase in zip(
    record_x2, record_Pv, record_phase
):
    ax.plot(x2_values, pv_values, color=PHASE_COLORS[phase], lw=1)

ax.set_title("Pressure-Volume diagram")
ax.set_xlabel("Ventricular volume (l)")
ax.set_ylabel("Ventricular pressure [mmHg]")
ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.93]) 

handles = [plt.Line2D([0], [0], color=PHASE_COLORS[p], lw=2, label=PHASE_LABELS[p])
           for p in ["contraction", "ejection", "relaxation", "filling"]]
fig_pv.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))

fig, ax = plt.subplots(figsize=(10,6))

for t, dp, phase in zip(
        t_segments,
        record_PB,
        record_phase):

    ax.plot(t, dp,
            color=PHASE_COLORS[phase])

ax.set_title("Pressure drop across RB")
ax.set_xlabel("Time [s]")
ax.set_ylabel("ΔP [mmHg]")
ax.grid(True)

# pulsen gemmen hele systemet 
Puls= np.concatenate(record_PB)
print("Mean pressure drop across RB: ", np.mean(Puls), "mmHg")

record_T_beat = np.asarray(record_T_beat)
fig_u, axu = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

# Mean pressure
axu[0].plot(record_beat_start, record_meanPB, "o-b")
axu[0].set_ylabel("Mean P (mmHg)")
axu[0].grid(True)

# Sympatisk og vagal aktivitet
axu[1].plot(record_beat_start, record_P_sym,
            "o-r", label="Sympathetic")
axu[1].plot(record_beat_start, record_P_vag,
            "o-g", label="Vagal")
axu[1].set_ylabel("Firing rate")
axu[1].legend()
axu[1].grid(True)

# HR
axu[2].plot(record_beat_start, 60/record_T_beat,
            "o-k")
axu[2].set_ylabel("HR (bpm)")
axu[2].grid(True)

# PAS-ændring
axu[3].step(record_beat_start,
            record_PAS,
            where="post",
            color="purple")
axu[3].set_ylabel("PAS (mmHg)")
axu[3].set_xlabel("Time (s)")
axu[3].grid(True)
axu[0].set_title("Baroreflex response")

fig_u.tight_layout()
fig_u.savefig("greene_ursino_HR.png", dpi=150)
plt.show()
"""
record_meanPB = np.array(record_meanPB)
record_T_beat = np.array(record_T_beat)

record_P_sym = np.array(record_P_sym)
record_P_vag = np.array(record_P_vag)

record_beat_start = np.array(record_beat_start)
record_PAS = np.array(record_PAS)
print(" beat   mean P_RB    N     f_sym  f_vagal   T        HR")  # NEW
for c in range(len(record_T_beat)):  # NEW
    P = record_meanPB[c]  # NEW
    print(f"  {c+1:3d}  PAS={record_PAS[c]:4.1f}  {P:7.1f}  {N_ursino(P):5.1f}  {sym_ursino(P):6.2f}  {para_ursino(P):6.2f}   "  # NEW
          f"{record_T_beat[c]:.3f} s  {60/np.array(record_T_beat[c]):5.1f} bpm")  # NEW

fig_u, axu = plt.subplots(4, 1, figsize=(8, 10), sharex=True)  # NEW
axu[3].step(record_beat_start, record_PAS, where="post"); axu[3].set_ylabel("PAS (mmHg)")  # NEW
axu[0].plot(record_beat_start, record_meanPB, "o-"); axu[0].set_ylabel("mean P at RB (mmHg)")  # NEW
axu[1].plot(record_beat_start, sym_ursino(record_P_sym), "o-", label="sympathetic")  # NEW
axu[1].plot(record_beat_start, para_ursino(record_P_vag), "o-", label="parasympathetic (vagal)")  # NEW
axu[1].set_ylabel("firing (spikes/s)"); axu[1].legend()  # NEW
axu[2].plot(record_beat_start, 60/record_T_beat, "o-"); axu[2].set_ylabel("heart rate (bpm)")  # NEW
axu[3].set_xlabel("Time (s)")  # NEW
t_lo = record_beat_start[pert_beats[0]]; t_hi = record_beat_start[pert_beats[-1] + 1]  # NEW
for a in axu: a.axvspan(t_lo, t_hi, color="#f3e6d6", alpha=0.6); a.grid(alpha=0.3)  # NEW
axu[0].set_title("Shaded: PAS lowered (reduced venous return)")  # NEW
fig_u.tight_layout()  # NEW
fig_u.savefig("greene_ursino_HR.png", dpi=150)  # NEW
"""
plt.show()  # NEW
sys.stdout = sys.__stdout__  # Reset standard output to console
log_file.close()  # Close the log file

