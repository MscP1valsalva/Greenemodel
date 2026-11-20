"""
model_simulation.py

Simulates the left ventricle and the arterial tree, beat by beat, during a
Valsalva manoeuvre (VM). The heart period and the peripheral resistance are
set by a carotid baroreflex.

The model is built from two papers:
    [G] Greene, Clark, Mohr & Bourland (1973). A mathematical model of
        left-ventricular function. Medical and Biological Engineering, 11,
        126-134.
        -> the heart (elastance), the four phases of the heart cycle and the
           arterial load (the circuit in Fig. 2, the equations 3-32).
    [U] Ursino (1998). Interaction between carotid baroregulation and the
        pulsating heart: a mathematical model. Am. J. Physiol. 275
        (Heart Circ. Physiol. 44), H1733-H1747.
        -> the baroreflex: afferent firing (Eq. 31), sympathetic and vagal
           firing (Eqs. 33-34), peripheral resistance (Eqs. 35-37) and heart
           period (Eqs. 38-42), with the numbers from Table 3.
    In the comments, [G] eqn 12 means equation 12 in Greene, and [U] Eq. 33
    means equation 33 in Ursino (in the APPENDIX of the paper).

How the simulation works:
    Greene splits every heart beat into four phases, because the valves open
    and close ([G] Fig. 3). In each phase the circuit is different, so each
    phase has its own set of differential equations:
        1. isovolumic contraction : both valves closed, Pv rises
        2. ejection               : aortic valve open, blood goes to the aorta
        3. isovolumic relaxation  : both valves closed, Pv falls
        4. filling                : mitral valve open, blood comes from the atrium
    Each phase is solved with solve_ivp, and an "event" stops it when the
    valve opens or closes. The end state of one phase is the start state of
    the next. After the filling, the baroreflex looks at the mean pressure of
    the beat and sets the length and resistance of the next beat.

The four state variables (the same as in [G], below eqn 6):
    x1 : pressure across the distal compliance CR [mmHg]
    x2 : volume of the ventricle Vv [l]
    x3 : pressure across the proximal compliance CL [mmHg]
    x4 : flow through the inertance L [l/s]

Units: litres, seconds and mmHg, the same as Greene. So a resistance is in
mmHg*s/l and a compliance in l/mmHg.

Functions in this file:
    PAS_R, ITP              - the Valsalva inputs: atrial pressure and intrathoracic pressure
    E_p, Pv                 - passive (diastolic) elastance and pressure of the ventricle
    A1 ... C3, E_H          - active (systolic) elastance, a sum of three Gaussian curves
    filling_rhs, ejection,
    isovolumic              - the differential equations of the four phases
    P_A                     - aortic root pressure
    contraction_end_event,
    ejection_end_event,
    relaxation_end_event    - when the valves open and close (stops each phase)
    N_ursino, sym_ursino,
    para_ursino, heart_period - the baroreflex control of the heart period
    sigma_R, TPR_u,
    fes_delayed, lag        - the baroreflex control of the peripheral resistance
    store                   - saves the result of one phase, with Pv, PA and PB

The file runs from top to bottom, without functions to call. load_model.py
runs it with runpy and reads the lists record_t, record_PA, ... afterwards.
Run this file directly to plot the result (the part at the bottom).
"""

import numpy as np
from scipy.integrate import solve_ivp
import scipy
import matplotlib.pyplot as plt
import  sys

k_VM = 0.1  # fraction of PAS left during the strain (tune to get ~30-50 % SV drop)
def PAS_R(solved_x4):
        return solved_x4 * L
        

def PAS_total(t_global, x4):
    """
        Atrial source pressure [mmHg]. Reduced to k_VM of its value while
        ITP > 0 (the strain), to model the reduced venous return.
    """
    k = np.where(ITP(t_global) > 0, k_VM, 1.0)
    l=ITP(t_global)
    return PAS_L+l + PAS_R(x4)*k


def last_x4():
    """
        Last recorded sample of x4 (flow through L) [l/s].
        Returns 0 before anything is recorded (first beat), otherwise the
        final value of the most recent segment.
    """
    if not record_x4:
        return 0
    return record_x4[-1][-1]


def ITP(t_global):
    """
        t_global : global time [s] (a number or a numpy array) \n

        Intrathoracic pressure [mmHg]. Not from the papers.
        During the strain the subject blows against a closed airway with
        40 mmHg, and this pressure is on the outside of the heart and the
        thoracic aorta. It is added to the ventricle pressure in every phase.
        np.where is used (and not if), so it also works on a whole array of
        times, which store() needs.

        Examples:
            ITP(10) -> 0     (no strain)
            ITP(60) -> 40    (during the strain, 58.4-73.4 s)
    """
    u = np.where((t_global>=start_time_VM)&(t_global<=end_time_VM),40,0)
    return u

def Pv(t_global,Vv):
    """
        t_global : global time [s] \n
        Vv : ventricle volume [l] \n

        Ventricle pressure during the filling phase [mmHg].
        [G] eqn 18: Pv = h2*Vv^2 + h1*Vv + h0 (a parabola fitted to the static
        pressure-volume curve of a dog heart).
        Two changes:
          - ITP is added, because the pressure outside the heart pushes on it
          - everything is multiplied by Fix_factor_passive (human volumes, see above)
        Note: the ITP is inside the bracket, so only 40*0.0045 = 0.18 mmHg of
        it reaches Pv during filling.

        Examples:
            Pv(0, 0.10)  -> 4.84 mmHg
            Pv(0, 0.12)  -> 6.95 mmHg
            Pv(60, 0.10) -> 5.02 mmHg   (during the strain, ITP = 40)
    """
    return (h2*Vv**2 + h1*Vv + h0) * Fix_factor_passive #TODO figuring out how to use fixfactor on PV part (fixed)

# ======================================================================
# Active elastance ([G] eqns 17, 20-29) - the ventricle during systole
# ======================================================================

def A1(Ved): #TODO make the boundary based on humans. Maybe a variable that can be changed
    """ [G] eqn 20. Example: A1(0.14) -> 10.59 """
    if Ved <= 0.1: # changed from 0.055 to 0.1 cite svendens, lisbeth
        return 2.1e-3*Ved**2 - 0.2338*Ved + 10.6
    else:
        return -1.0133e-3*Ved**2 - 0.0626*Ved + 10.6
def A2(Ved): return 150.0                                   # [G] eqn 21
def A3(Ved):
    """ [G] eqn 22. Example: A3(0.14) -> 10.17 """
    if Ved <= 0.1: # changed from 0.055 to 0.1 cite svendens, lisbeth
        return 2.8e-3*Ved**2 - 0.3*Ved + 10.2
    else:
        return 6.476e-4*Ved**2 - 0.1816*Ved + 10.2
def B1(Ved): return -1e-4*Ved + 0.0157                      # [G] eqn 23
def B2(Ved): return 45.0                                    # [G] eqn 24
def B3(Ved): return -4e-4*Ved + 0.0431                      # [G] eqn 25
def C1(Ved): return 9.7e-6*Ved**2 - 2.8e-4*Ved + 0.12       # [G] eqn 26
def C2(Ved): return 0.06                                    # [G] eqn 27
def C3(Ved): return -40.2e-6*Ved**2 + 4.78e-3*Ved + 0.06    # [G] eqn 28

def E_H(Ved, t): #TODO move all variables/constants to the top, then functions then main (fixed)
    """
        Ved : end-diastolic volume of this beat [l] \n
        t : time since the start of this beat's contraction [s] (local time,
            a number or a numpy array) \n

        Active (systolic) elastance [mmHg/l], [G] eqn 17 with the scale factors above.
        The ventricle pressure in systole is Pv = E_H * Vv (+ ITP).

        Not in [G] or [U]: the contraction also depends on the baroreflex.
        fs is the afferent firing N_ursino scaled to 0-1 (0 = minimum firing,
        1 = maximum firing, 0.5 at P = Pn):
            afs : multiplies the amplitude (from 1 at fs = 0 to 1.15 at fs = 1)
            bfs : multiplies the time, so a bfs > 1 makes the curve come earlier
                  and shorter (from 1 at fs = 0 to 1.0 at fs = 1, max 1.025 at fs = 0.5)
        P_reflex is a global variable, set in the loop after every beat.

        Example (Ved = 0.14 l, P_reflex = Pn = 92 mmHg):
            t [s]    0     0.1    0.2    0.3    0.4    0.5
            E_H      6.9   6.9    14.1   969    149    6.9    mmHg/l
            -> the peak of about 1000 mmHg/l is at t = 0.33/1.025 = 0.32 s
    """
    fs = (sym_ursino(P_reflex) - f_es_inf)/(f_es0 - f_es_inf) # from 0 to 1  #TODO Change the N_urisino to the symp function (fixed)
    afs = 1 + 0.5*fs - 0.35*fs**2 + 0.00075*fs**3 # Increase = harder contraction #TODO Change it based on Greene innervated left ventricle (fixed)
    bfs = 1 + 0.1*fs - 0.007*fs**2 + 0.00015*fs**3 # increase = faster contraction #TODO Change it based on Greene innervated left ventricle (fixed)

    # Add up the three Gaussians of [G] eqn 17. The for loop takes one
    # (A, B, C) triple at a time.
    s = 0.0
    for A, B, C in [(A1(Ved),B1(Ved),C1(Ved)), (A2(Ved),B2(Ved),C2(Ved)), (A3(Ved),B3(Ved),C3(Ved))]:
        s += afs*scale_A*A * np.exp(-((scale_B*B)**2) * (t*bfs - scale_C*C)**2)
    return s + D


def filling_rhs(t, x, Ved,cycle_offset):
    """
        Filling phase, Pv < PAS: mitral valve open, aortic valve closed.
        [G] eqns 3-6, [G] Fig. 3a.

        x : [x1, x2, x3, x4] (see the top of the file) \n
        Ved : not used in this phase, but solve_ivp gives the same args to every phase \n

        dx1 : [G] eqn 3 - the distal compliance CR
        dx2 : [G] eqn 4 - flow into the ventricle through the mitral valve:
              dVv/dt = (PAS - Pv) / Rm   (Ohm's law across Rm)
              [G] writes it as -Ep*x2/Rm + PAS/Rm, which is the same since Ep*x2 = Pv.
        dx3 : [G] eqn 5 - the proximal compliance CL. The aortic valve is
              closed, so no flow comes from the heart.
        dx4 : [G] eqn 6 - the flow through L: L*dx4/dt = (pressure before L) - (pressure after Rb)
    """
    x1, x2, x3, x4 = x   # x1=P_CR, x2=Vv, x3=P_CL, x4=flow through L
    dx1 = (-x1 + RR * x4) / (CR * (R2 + RR))
    dx2 = (PAS - Pv(t+cycle_offset, x2)+ITP(t+cycle_offset)) / Rm
    dx3 = (-x3 - RL * x4) / (CL * (R1 + RL))
    dx4 = (-x4 / L) * (Rb + R1*RL/(R1+RL) + R2*RR/(R2+RR)) \
          + RL * x3 / (L * (R1 + RL)) \
          - RR * x1 / (L * (R2 + RR))
    return [dx1, dx2, dx3, dx4]

def ejection(t, x, Ved, cycle_offset):
    """
        Ejection phase, Pv > PA: aortic valve open, mitral valve closed.
        [G] eqns 11-14, [G] Fig. 3c.

        x : [x1, x2, x3, x4] \n
        Ved : end-diastolic volume of this beat, used by E_H \n

        The ventricle pressure is E_H*x2 + ITP. It is connected to the aortic
        root through Ra, so Ra, R1 and RL meet at the root, which is why RT
        ([G] eqn 15) appears in the denominators.

        dx1 : [G] eqn 11 (same as eqn 3)
        dx2 : [G] eqn 12 - minus the flow out through the aortic valve
        dx3 : [G] eqn 13
        dx4 : [G] eqn 14

        About the sign before RL*x3 in dx2: solving the circuit gives
            flow out = ((R1+RL)*Pv - RL*x3 + R1*RL*x4) / RT
        so dx2 = -(flow out) has +RL*x3, which is what the code does.
        The printed eqn 12 is hard to read in the scan; the derivation
        confirms the plus sign. The same derivation gives eqns 13 and 14 exactly.
    """
    x1, x2, x3, x4 = x   # x1=P_CR, x2=Vv, x3=P_CL, x4=flow through L
    dx1 = (-x1 + RR * x4) / (CR * (R2 + RR))
    dx2 = (-(R1+RL)*((E_H(Ved,t)*x2)+ITP(t+cycle_offset))+RL*x3-R1*RL*x4) / RT # +RL*x3 is correct, see the docstring
    dx3 = (RL * (E_H(Ved,t)*x2+ITP(t+cycle_offset))-(RL+Ra)*x3-RL*Ra*x4) / (CL * RT)
    dx4 = (-RR* x1 / (L*(R2+RR))) + (R1*RL*((E_H(Ved,t)*x2)+ITP(t+cycle_offset))+RL*Ra*x3-(R1*RL*Ra+RG*RT)*x4) / (L*RT)
    return [dx1, dx2, dx3, dx4]

def isovolumic(t, x, Ved, cycle_offset):
    """
        Isovolumic contraction AND relaxation, PAS < Pv < PA: both valves closed.
        [G] eqns 7-10, [G] Fig. 3b.

        Both valves are closed, so the volume cannot change: dx2 = 0 ([G] eqn 8).
        The arterial tree is then on its own, so dx1, dx3 and dx4 are the same
        as in the filling phase ([G] eqns 7, 9, 10 = eqns 3, 5, 6).
        The pressure in the ventricle still changes, because E_H changes with
        time, but that is calculated outside (in the events and in store).
    """
    x1, x2, x3, x4 = x   # x1=P_CR, x2=Vv, x3=P_CL, x4=flow through L
    dx1 = (-x1 + RR * x4) / (CR * (R2 + RR))
    dx2 = 0
    dx3 = (-x3-RL*x4) / (CL * (R1 + RL))
    dx4 = (-x4 / L) * (Rb + R1*RL/(R1+RL) + R2*RR/(R2+RR)) \
          + RL * x3 / (L * (R1 + RL)) \
          - RR * x1 / (L * (R2 + RR))
    return [dx1, dx2, dx3, dx4]

def P_A(x3, dx3):
    """
        x3 : pressure across CL [mmHg] \n
        dx3 : dx3/dt from the phase that is running [mmHg/s] \n

        Aortic root pressure [mmHg]. Not written out in [G], but it follows from
        the circuit: PA is the pressure across CL plus the drop across R1.
        The flow through R1 is the flow into CL, CL*dx3/dt, so
            PA = x3 + R1*CL*dx3/dt
        This is the pressure that is compared with the measured blood pressure.
    """
    return dx3 * CL *R1 +x3

def contraction_end_event(t, x, Ved, cycle_offset):
    """
        End of isovolumic contraction: the aortic valve opens when Pv reaches PA.
        Returns PA - Pv. It is positive at the start (Pv low) and becomes
        negative when Pv passes PA, so direction = -1.
        dx3 is from the isovolumic equations ([G] eqn 9), the phase that is running.
    """
    x1, x2, x3, x4 = x   # x1=P_CR, x2=Vv, x3=P_CL, x4=flow through L
    dx3 = (-x3-RL*x4) / (CL * (R1 + RL))
    Pv_active = (E_H(Ved, t) * x[1]+ ITP(t+cycle_offset))
    #print("Contraction phase: Pv_active =", Pv_active, "P_A =", P_A(x3, dx3), "x1 =", x1, "x2 =", x2, "x3 =", x3, "x4 =", x4)
    return P_A(x3, dx3) - Pv_active   # 0 when Pv reaches P_A


def ejection_end_event(t, x, Ved, cycle_offset):
    """
        End of ejection: the aortic valve closes when Pv falls below PA.
        Returns PA - Pv. It is negative during ejection (Pv > PA) and becomes
        positive when Pv falls below PA, so direction = +1.
        dx3 is from the ejection equations ([G] eqn 13).
    """
    x1, x2, x3, x4 = x   # x1=P_CR, x2=Vv, x3=P_CL, x4=flow through L
    dx3 = (RL * (E_H(Ved,t)*x2+ ITP(t+cycle_offset))-(RL+Ra)*x3-RL*Ra*x4) / (CL * RT)
    Pv_active = (E_H(Ved, t) * x[1] + ITP(t+cycle_offset))
    #print("Ejection phase: Pv_active =", Pv_active, "P_A =", P_A(x3, dx3), "x1 =", x1, "x2 =", x2, "x3 =", x3, "x4 =", x4)
    return P_A(x3, dx3) - Pv_active   # 0 when Pv reaches P_A


def relaxation_end_event(t, x, Ved, cycle_offset): #TODO make changes to the these to make it work during Valsalva 
    """
        End of isovolumic relaxation: the mitral valve opens when Pv falls below PAS.
        Returns (PAS - 0.1) - Pv, which goes from negative to positive, so
        direction = +1.
        The 0.1 mmHg margin makes the valve open just below PAS. Then
        PAS - Pv > 0 at the start of filling, so the ventricle really fills.
        PAS follows PAS_R(t), so the event also works during the strain.
    """
    x1, x2, x3, x4 = x   # x1=P_CR, x2=Vv, x3=P_CL, x4=flow through L
    Pv_active = (E_H(Ved, t) * x[1]+ITP(t+cycle_offset))
    #print("Relaxation phase: Pv_active =", Pv_active, "x1 =", x1, "x2 =", x2, "x3 =", x3, "x4 =", x4)
    return (PAS) - Pv_active   # 0 when Pv reaches P_AS. NEW: was 9.9; now follows PAS so it still works when PAS is lowered


def N_ursino(P):
    """
        P : pressure [mmHg] \n

        Afferent firing of the carotid baroreceptors, f_cs [spikes/s].
        [U] Eq. 31, a sigmoid:
            f_cs = (f_min + f_max*exp((P - Pn)/ka)) / (1 + exp((P - Pn)/ka))
        At low pressure it goes to f_min, at high pressure to f_max, and at
        P = Pn it is exactly halfway.

        Two safety checks (not in [U]):
          - if P is nan or inf (the solver has gone wrong), stop with an error
            message, instead of continuing with nonsense
          - the exponent is limited to +-50, so np.exp does not overflow

        Examples:
            N_ursino(60)  -> 5.31
            N_ursino(92)  -> 25.15   (= (2.52 + 47.78)/2, the middle)
            N_ursino(130) -> 46.06
    """
    x = (P - Pn) / ka

    # Stop the program on invalid values #? why are we doing this? how does it overflow
    if not np.isfinite(x):
        raise ValueError(
            f"Invalid value in N_ursino: "
            f"P={P}, Pn={Pn}, ka={ka}, x={x}"
        )

    # Limit the exponent, so np.exp does not overflow #? Why? 
    x = np.clip(x, -50.0, 50.0) 

    e = np.exp(x) 

    return (f_min + f_max * e) / (1.0 + e)

def sym_ursino(P): #? why are we inputting pressure again? shoundn't it be firing rate?
    """
        P : pressure [mmHg] \n

        Efferent sympathetic firing f_es [spikes/s], [U] Eq. 33:
            f_es = f_es_inf + (f_es0 - f_es_inf)*exp(-k_es*f_cs)
        More afferent firing -> less sympathetic firing (exponential fall).

        Examples:
            sym_ursino(60)  -> 11.89   (low pressure, high sympathetic drive)
            sym_ursino(92)  -> 4.67
            sym_ursino(130) -> 2.73
    """
    return f_es_inf + (f_es0 - f_es_inf)*np.exp(-k_es*N_ursino(P)) 

def para_ursino(P): #? why are we inputting pressure again? shoundn't it be firing rate?
    """
        P : pressure [mmHg] \n

        Efferent vagal (parasympathetic) firing f_ev [spikes/s], [U] Eq. 34:
            f_ev = (f_ev0 + f_ev_inf*exp((f_cs - f_cs0)/k_ev)) / (1 + exp((f_cs - f_cs0)/k_ev))
        More afferent firing -> more vagal firing (a sigmoid, from f_ev0 to f_ev_inf).

        Examples:
            para_ursino(60)  -> 3.38
            para_ursino(92)  -> 4.77
            para_ursino(130) -> 6.15
    """
    e = np.exp((N_ursino(P) - f_cs0)/k_ev)
    return (f_ev0 + f_ev_inf*e)/(1 + e)

def heart_period(P_sym,P_vag):
    """
        P_sym : pressure that drives the sympathetic part [mmHg] (from a few beats ago) \n #? how many sampels is that?
        P_vag : pressure that drives the vagal part [mmHg] (from the last beat) \n #? i know that we are sampling the model when calculatiing it. how many samples are 0.2s ago

        Heart period T [s], static version of [U] Eqs. 38-42:
            dT_s = G_Ts * ln(f_es - f_es_min + 1)  if f_es >= f_es_min, else 0   [U] Eq. 38
            dT_v = G_Tv * f_ev                                                  [U] Eq. 40
            T    = T0 + dT_s + dT_v                                             [U] Eq. 42
        G_Ts is negative: sympathetic firing makes the period shorter.
        G_Tv is positive: vagal firing makes the period longer.
        Two pressures are used, because the two parts have different delays
        in [U] Table 3 (2 s for the sympathetic, 0.2 s for the vagal).

        Examples (same pressure in both):
            P [mmHg]    60     75     92     110    130
            T [s]       0.785  0.864  1.152  1.430  1.494
            HR [bpm]    76.5   69.5   52.1   42.0   40.2
    """
    if sym_ursino(P_sym) >= f_es_min:
        dT_s = G_Ts * np.log(
            sym_ursino(P_sym) - f_es_min + 1.0
        )
    else:
        dT_s = 0.0

    dT_v = G_Tv*para_ursino(P_vag)                                        # vagal: lengthens
    return T0 + dT_s + dT_v  #? Where is the delay here?

def sigma_R(G, fes):
    """
        G : gain of the bed (G_Rsp or G_Rep) [mmHg*s/ml] \n
        fes : sympathetic firing (already delayed) [spikes/s] \n

        Static change of resistance, [U] Eq. 35.
        np.maximum(..., 0) makes the change 0 when fes < f_es_min, which is
        the "else 0" part of Eq. 35 (ln(0 + 1) = 0).

        Examples:
            sigma_R(G_Rsp, 2.0) -> 0      (below f_es_min = 2.66)
            sigma_R(G_Rsp, 5.0) -> 0.84 mmHg*s/ml
    """
    return G*np.log(np.maximum(fes - f_es_min, 0) + 1) #? i would like the fork function to be there

def TPR_u(dRsp, dRep):
    """
        dRsp, dRep : change of the splanchnic and extrasplanchnic resistance [mmHg*s/ml] \n

        Total peripheral resistance of Ursino's two beds [mmHg*s/ml].
        R = R0 + dR for each bed ([U] Eq. 37), and the two beds are in
        parallel, so  1/TPR = 1/Rsp + 1/Rep  ->  TPR = Rsp*Rep/(Rsp + Rep).

        Example: at rest (P = Pn) TPR_u = 0.961 mmHg*s/ml (= TPR_ref).
    """
    Rsp, Rep = Rsp0_u + dRsp, Rep0_u + dRep
    return Rsp*Rep/(Rsp + Rep)


def fes_delayed(t, D):
    """
        t : global time now [s] \n
        D : delay [s] \n

        The sympathetic firing D seconds ago, f_es(t - D), the delay in [U] Eq. 35.
        The firing is saved once per beat in fes_hist, so np.interp draws a
        straight line between the saved values and reads off the value at t - D.
        left=fes_ref: before the first saved time, use the resting value.

        Example: if fes_hist_t = [0, 1, 2] and fes_hist = [4, 5, 6], then
            fes_delayed(3.5, 2) -> 5.5   (the value at t = 1.5)
    """
    return np.interp(t - D, fes_hist_t, fes_hist, left=fes_ref)

def lag(Delta, s, T, tau): #? where does this come from? 
    """
        Delta : the value now (dRsp or dRep) \n
        s : the target value, the static curve sigma (held constant over the beat) \n
        T : length of the beat [s] \n
        tau : time constant [s] \n

        One beat of the first-order low-pass [U] Eq. 36, solved exactly.
        If sigma is constant over the beat, the solution of
            d(Delta)/dt = (s - Delta) / tau
        after a time T is  Delta(T) = s + (Delta(0) - s)*exp(-T/tau).
        Why exact and not Euler: one beat (about 1 s) is not small compared
        with tau = 6 s, and this way it is correct for any beat length.

        Example: lag(0, 1, 0.8, 6) -> 0.125
            (after 0.8 s, 12.5 % of the way from 0 to 1, because 1 - exp(-0.8/6) = 0.125)
    """
    return s + (Delta - s)*np.exp(-T/tau)


def store(sol, phase, cycle_offset):
    """
        sol : the result from solve_ivp for one phase \n
        phase : "contraction", "ejection", "relaxation" or "filling" \n
        cycle_offset : global time when this beat started [s] \n

        Saves the time and the four states of one phase in the record lists,
        and calculates three pressures from them:
            Pv : ventricle pressure
            PA : aortic root pressure (P_A), the one compared with the measurement
            PB : pressure after Rb, PA - Rb*x4
        The equations for Pv and dx3 depend on the phase (which valve is open),
        so they are chosen with if/elif.
        sol.t is the local time; plot_time = sol.t + cycle_offset is the global time.

        Note on PB: it leaves out the drop across L (L*dx4/dt). Over a whole
        beat this averages to about 0 (x4 is about the same at the start and
        end of a beat), so the mean of PB over a beat, which the reflex uses,
        is not affected.
    """
    plot_time = sol.t + cycle_offset
    record_t.append(plot_time)
    record_x1.append(sol.y[0])
    record_x2.append(sol.y[1])
    record_x3.append(sol.y[2])
    record_x4.append(sol.y[3])
    record_phase.append(phase)

    if phase == "filling":
        pv_values = Pv(plot_time, sol.y[1])                     # passive pressure, [G] eqn 18
        dx3 = (-sol.y[2] - RL * sol.y[3]) / (CL * (R1 + RL))    # [G] eqn 5
        pa_values = P_A(sol.y[2], dx3)
    elif phase == "ejection":
        pv_values = E_H(Ved, sol.t) * sol.y[1]+ITP(sol.t+cycle_offset)   # active pressure, [G] eqn 17
        dx3 = (RL * (E_H(Ved,sol.t)*sol.y[1]+ITP(sol.t+cycle_offset))-(RL+Ra)*sol.y[2]-RL*Ra*sol.y[3]) / (CL * RT)  # [G] eqn 13
        pa_values = P_A(sol.y[2], dx3)
    else:  # contraction or relaxation
        pv_values = E_H(Ved, sol.t) * sol.y[1]+ITP(sol.t+cycle_offset)   # active pressure, [G] eqn 17
        dx3 = (-sol.y[2]-RL*sol.y[3]) / (CL * (R1 + RL))        # [G] eqn 9
        pa_values = P_A(sol.y[2], dx3)
    ## pressure across RB
    delta_PB = Rb * sol.y[3]
    Pb = pa_values - delta_PB
    record_PB.append(Pb)
    record_Pv.append(pv_values)
    record_PA.append(pa_values)

# ======================================================================
# Heart parameters ([G] eqn 31) and the Valsalva manoeuvre
# ======================================================================
Rm  = 12    # mitral valve resistance, mmHg*s/l (altered from 0.33 to 12 to match the correct physiological range)
PAS_L=5
start_time_VM = 40
end_time_VM = start_time_VM + 15
Ra = 0.6*20
Fix_factor_passive = 0.0045
# ======================================================================
# Arterial-load parameters ([G] eqn 32, circuit in [G] Fig. 2)
# ======================================================================
CL = 2.35e-4 * 5 # the lower it is, the more pressure in the ascending aorta, but it reduces stroke volume
R1 = 60 *3# sets the height of the notch, and also the height of the curve.
CR = 1.33e-4 *8# the lower it is, the more pressure in the descending aorta, but it reduces stroke volume
R2 = 300.0*0.1 # a high value removes the notch, increases amplitude, lowers the cutting point. A 0.x value raises the cutting point, but also the pressure
RL = 12640.0 * 0.2 # the lower it is, the more volume ejected per beat.
Rb = 60.0 # if it increases, the pressure rises but the cut points do not.
L  = 21.9  # inertance of the blood [mmHg*s^2/l], unchanged from [G]
RT = R1* Ra + R1*RL + Ra*RL   # total resistance, mmHg*s/l
h0, h1, h2 = 0.234, 147.0, 106000.0
# ======================================================================
# Active elastance ([G] eqns 17, 20-29) - the ventricle during systole
# ======================================================================
D = -150
scale_A = 6.5
scale_B = 0.4
scale_C = 5.5

# ======================================================================
# The events for the sovle IVP of the four phases ([G] eqns 3-14)
# ======================================================================
relaxation_end_event.terminal = True
relaxation_end_event.direction = +1 # (PAS - 0.1) - Pv goes from negative to positive when Pv becomes less than PAS
contraction_end_event.terminal = True
contraction_end_event.direction = -1 # PA - Pv goes from positive to negative when Pv becomes greater than PA
ejection_end_event.terminal = True
ejection_end_event.direction = +1 # PA - Pv goes from negative to positive when Pv becomes less than PA
# ======================================================================
# Ursino (1998) baroreflex: heart period ([U] Eqs. 31, 33, 34, 38-42)
# ======================================================================
Pn = 92
ka = 11.758
f_min = 2.52
f_max = 47.78
f_es_inf = 2.10
f_es0 = 16.11
k_es = 0.0675
f_es_min = 2.66
f_ev0 = 3.2
f_ev_inf = 6.3 
k_ev = 7.06 
f_cs0 = 25.0
G_Ts = -0.13
G_Tv = 0.15
T0 = 0.58 
_trapz = getattr(np, "trapezoid", None) or np.trapz #? this seems extensive. maybe just make sure you have the right numpy
T_beat = heart_period(Pn,Pn)       # heart period of the first beat (resting value, 1.152 s) #? so the first heartbeat is 52 bpm. We should just make this dependent on the subject. 

# ======================================================================
# Ursino (1998) baroreflex: peripheral resistance ([U] Eqs. 35-37, Table 3)
# ======================================================================
Rsp0_u = 2.49        # splanchnic gain and resistance without innervation, mmHg s/ml ([U] Table 3)
G_Rsp = 0.695 
G_Rep = 0.53         # extrasplanchnic gain and resistance without innervation, mmHg s/ml ([U] Table 3)
Rep0_u = 0.78
tau_R = 3              # time constant and pure delay, s ([U] Table 3, the same for sp and ep)
D_R = 1

fes_ref = sym_ursino(Pn)                                       # sympathetic firing at the set point (4.67 spikes/s)
dRep = sigma_R(G_Rep, fes_ref)  # start at the resting values (no transient at t = 0)
dRsp = sigma_R(G_Rsp, fes_ref)
TPR_ref = TPR_u(dRsp, dRep)                                    # resting Ursino TPR, the reference for kR
RR_base = 8550*0.2                                             # resting RR. CHANGED FROM [G]: 8550 -> 8550*0.2 #? but why?
kR = 1.0                                                       # RR scale factor used by the next beat
fes_hist_t = [0.0]                      # sympathetic firing history (time, value), for the delay #? i think the delay should be in samples. the model is being calculated at 1000Hz. 
fes_hist = [fes_ref]

# ======================================================================
# Solving the equations
# ======================================================================

record_x1, record_x2, record_x3, record_x4 = [], [], [], []
record_t, record_phase = [], []
record_Pv, record_PA = [], []
record_PB,record_Pdistal = [],[]
record_P_sym, record_P_vag = [], []  # pressures used for the sympathetic and vagal part of the reflex
record_T_beat, record_meanPB, record_beat_start = [], [], []  # one value per beat
record_RR = []  # peripheral resistance RR of each beat

# solve_ivp settings
t_span = (0, 1)   # A maximal cutoff for every phase (this should never be reached, the events stop it first)
max_step = 1e-3   # The largest step solve_ivp may take [s]. Small steps give a smooth curve
                  # and make sure the events (valve openings) are not stepped over
cycle_offset = 0  # global time when the current beat starts [s]

#PAS_base, PAS_low = 10.0, 6.0          # mmHg
#pert_beats = range(8, 18)              # beats 9-18 (0-based index)
n_beats = 100  # number of heart beats to simulate. 180 beats at about 1 s cover the whole measurement
record_PAS = []  # atrial source pressure at the start of each beat
P_reflex = Pn    # pressure the reflex sees, used by E_H. Starts at the set point #? constant = constant. we should just use it

t_segments = []
# ======================================================================
# Main model running
# ======================================================================

for i in range(n_beats): # one heart beat per loop
    #print(f"Cycle {i+1}")
    #PAS = PAS_low if i in pert_beats else PAS_base
      # fraction of venous return left during the strain (tune to get ~30-50 % SV drop)
    PAS = float(PAS_total(cycle_offset, last_x4()))
    record_PAS.append(PAS)

    # The baroreflex changes RR from beat to beat (kR is calculated at the end of the
    # previous beat). RR is kept constant inside a beat.
    RR = RR_base*kR                     # reflex-scaled peripheral resistance for this beat
    RG = Rb + (R2*RR/(RR + R2))         # [G] eqn 16. RG depends on RR, so recompute it 
    record_RR.append(RR)


    # --- Start state of this beat ---
    # First beat: a state taken from the end of an earlier run, so the model starts
    # close to a steady beat. Later beats: start where the last filling ended.
    if i == 0:
        x0 = [57.51041422664778, 0.1101278932374876, 61.32586230866999, 0.11973411143947638]
        #x0 = [70, 0.014, 70, -0.000017]  # reset to initial state for subsequent cycles
        Ved = 0.14
    else:
         x0 = [sol_filling.y[0,-1], sol_filling.y[1,-1], sol_filling.y[2,-1], sol_filling.y[3,-1]]  # final state of filling phase

    # --- 1. Isovolumic contraction ([G] eqns 7-10) ---
    # Local time starts at 0 here, because E_H(Ved, t) needs the time since the start of systole.
    # Stops when Pv reaches PA (contraction_end_event)
    #print("Ved: ",Ved)
    sol_isovolumic = solve_ivp(isovolumic, t_span, x0, args=(Ved,cycle_offset), max_step=max_step,
                                events=contraction_end_event)
    #print("Isovolumic contraction phase completed at t =", sol_isovolumic.t[-1])
    #print(sol_isovolumic.status)
    store(sol_isovolumic, "contraction", cycle_offset)

    # --- 2. Ejection ([G] eqns 11-14) ---
    # Starts at the time where the contraction ended (local time continues, so
    # E_H continues), and stops when Pv falls below PA (ejection_end_event).
    ej_starttime = sol_isovolumic.t[-1]
    ej_span = (ej_starttime, ej_starttime + 1)
    x0 = [sol_isovolumic.y[0,-1], sol_isovolumic.y[1,-1], sol_isovolumic.y[2,-1], sol_isovolumic.y[3,-1]]  # final state of contraction phase
    sol_ejection = solve_ivp(ejection, ej_span, x0, args=(Ved,cycle_offset), max_step=max_step,
                             events=ejection_end_event)
    #print("Ejection phase completed at t =", sol_ejection.t[-1])
    #print(sol_ejection.status)
    #print("RR",RR)
    store(sol_ejection, "ejection", cycle_offset)

    # --- 3. Isovolumic relaxation ([G] eqns 7-10, the same equations as the contraction) ---
    # Stops when Pv falls below PAS (relaxation_end_event), at the latest after 0.25 s
    isor_starttime = sol_ejection.t[-1]
    isor_span = (isor_starttime, isor_starttime + 0.25)
    x0 = [sol_ejection.y[0,-1], sol_ejection.y[1,-1], sol_ejection.y[2,-1], sol_ejection.y[3,-1]]  # final state of ejection phase
    sol_isovolumic = solve_ivp(isovolumic, isor_span, x0, args=(Ved,cycle_offset), max_step=max_step,
                                events=relaxation_end_event)
    #print("Isovolumic relaxation phase completed at t =", sol_isovolumic.t[-1])
    #print(sol_isovolumic.status)
    store(sol_isovolumic, "relaxation", cycle_offset)

    # --- 4. Filling ([G] eqns 3-6) ---
    # The beat ends at the heart period T_beat set by the reflex (local time = T_beat),
    # instead of when Pv reaches PAS. So the filling lasts T_beat minus the time used
    # by the other three phases. If systole is longer than T_beat, the filling still
    # gets at least 0.01 s, so the beat never has zero filling.
    x0 = [sol_isovolumic.y[0,-1], sol_isovolumic.y[1,-1], sol_isovolumic.y[2,-1], sol_isovolumic.y[3,-1]]  # final state of relaxation phase
    fill_starttime = sol_isovolumic.t[-1]
    t_max=T_beat-fill_starttime
    if t_max < 0.01:
        t_max = 0.01
    fill_span = (fill_starttime, fill_starttime + t_max)
    sol_filling = solve_ivp(filling_rhs, fill_span, x0, args=(Ved,  cycle_offset),
                             max_step=max_step)
    Ved = sol_filling.y[1,-1] # End-diastolic volume = the volume at the end of filling, used by E_H in the next beat (Frank-Starling)
    #print("Filling phase completed at t =", sol_filling.t[-1])
    #print(sol_filling.status)
    store(sol_filling, "filling", cycle_offset)

    # --- Baroreflex: heart period of the next beat ---
    # Mean pressure at Rb over this beat. The last 4 entries in the record lists
    # are the 4 phases of this beat. The mean is the area under the curve
    # (trapezoid rule) divided by the length of the beat.
    tt = np.concatenate(record_t[-4:]); pp = np.concatenate(record_PB[-4:])
    meanPB = _trapz(pp, tt)/(tt[-1] - tt[0])
    P_reflex = meanPB
    record_beat_start.append(cycle_offset)
    record_T_beat.append(sol_filling.t[-1])
    record_meanPB.append(meanPB)
    # Delays ([U] Table 3), made by counting beats:
    #   sympathetic, D_T,s = 2 s: use the mean pressure from 3 beats before this one
    #     (record_meanPB[-1] is this beat, [-4] is 3 beats back, about 2.5-3.5 s).
    #     Until there are 4 beats, use the set point Pn.
    #   vagal, D_T,v = 0.2 s: shorter than a beat, so use this beat.
    if len(record_meanPB) >=4:  # make sure we have at least 4 beats to compute the sympathetic firing rate #TODO figure out how to do it on 4*1000 samples before instead of seconds
        P_sym = record_meanPB[-4]
    else:
        P_sym = Pn
    P_vag = meanPB # calculate vagal firing rate based on the last beat #TODO make it based on 0.2*1000 sampels before 
    record_P_sym.append(P_sym)
    record_P_vag.append(P_vag)
    T_beat = heart_period(P_sym,P_vag)
    #print(f"Reflex: mean P_RB = {meanPB:.1f} mmHg -> next T = {T_beat:.3f} s ({60/T_beat:.1f} bpm)")

    # --- Baroreflex: peripheral resistance of the next beat ([U] Eqs. 35-37) ---
    # 1. save this beat's sympathetic firing, with the global time at the end of the beat
    # 2. read the firing D_R = 2 s ago
    # 3. move dRsp and dRep one beat towards the static curve (low-pass, tau_R = 6 s)
    # 4. kR = how much the total resistance has changed compared with rest
    fes_hist_t.append(cycle_offset+sol_filling.t[-1]); fes_hist.append(sym_ursino(meanPB))       # store this beat's firing
    fes_D = fes_delayed(cycle_offset+sol_filling.t[-1], D_R)                                     # firing D_R seconds ago
    dRsp = lag(dRsp, sigma_R(G_Rsp, fes_D), sol_filling.t[-1], tau_R)              # [U] Eq. 36, splanchnic
    dRep = lag(dRep, sigma_R(G_Rep, fes_D), sol_filling.t[-1], tau_R)              # [U] Eq. 36, extrasplanchnic
    kR = TPR_u(dRsp, dRep)/TPR_ref
    # The next beat starts where this one ended. sol_filling.t[-1] is the local
    # end time of the beat = the length of the beat
    cycle_offset += sol_filling.t[-1]

for t in record_t:
    t_segments.append(t)


"""
Run the file: simulate the model and plot the aortic pressure, the heart
rate and the peripheral resistance, with the Valsalva strain marked.
(load_model.py runs this file with runpy, which does not run this part.)
"""
if __name__ == "__main__":
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
    # This part only runs when you run this file directly

    # Put the phases after each other into one long array (the same as load_model.py)
    t_all = np.concatenate(record_t)
    pa_all = np.concatenate(record_PA)

    # One value per beat, plotted at the start time of each beat
    beat_start = np.array(record_beat_start)
    heart_rate = 60 / np.array(record_T_beat)          # [bpm]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    linewidth = 0.5

    ax1.plot(t_all, pa_all, linewidth=linewidth, label="PA (aortic root)")
    ax1.plot(beat_start, record_meanPB, ".-", linewidth=linewidth, label="mean PB per beat")
    ax1.set_ylabel("Pressure [mmHg]")

    ax2.plot(beat_start, heart_rate, ".-", linewidth=linewidth)
    ax2.set_ylabel("Heart rate [bpm]")

    ax3.plot(beat_start, record_RR, ".-", linewidth=linewidth)
    ax3.set_ylabel("RR [mmHg*s/l]")
    ax3.set_xlabel("Time [s]")

    for ax in (ax1, ax2, ax3):
        # Grey band during the strain
        ax.axvspan(start_time_VM, end_time_VM, color="grey", alpha=0.2, linewidth=0)
    ax1.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    plt.show()
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

for t, pv_values, pa_values, x4, phase in zip(
    t_segments, record_Pv, record_PA, record_x4, record_phase
):
    ax.plot(t, pa_values, "r--", lw=1.5, zorder=3)
    ax.plot(t, pv_values, color=PHASE_COLORS[phase], lw=1)

ax.step(record_beat_start, record_PAS, where="post", color="k", lw=1.5, zorder=2)
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
handles.append(plt.Line2D([0], [0], color="black", lw=2, label="PAS"))
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




"""
NEUROCONTROL: a simplified baroreflex that controls the heart period only.

This block runs the same Greene model a second time, but the Ursino reflex is
replaced by one empirical curve that maps blood pressure straight to heart rate.
No afferent firing rate, no separate sympathetic / vagal pathways, no delays.

Every beat:
    0. phases 1-3 run until their valve events (lengths t1, t2, t3); the filling
       then lasts t_fill = T - t1 - t2 - t3, so T sets the length of the beat
    1. RB measures the pressure: mean pressure after Rb over the beat (meanPB)
    2. the logistic carotid-HR curve maps meanPB to a heart rate
       (Kent et al. 1972, as used by Ogoh et al. 2003, J Physiol 550:317):
           HR = HR_min + (HR_max - HR_min) / (1 + exp(k*(meanPB - P_c)))
             P > P_c: HR falls (parasympathetic side)
             P < P_c: HR rises (sympathetic side)
    3. heart period of the NEXT beat: T = 60/HR (the filling phase ends at T)

Only the heart period is controlled here. RR stays at RR_base and the
contractility is held at its set-point value (P_reflex = Pn), so the effect of
heart rate alone can be seen.

Set point: P_c must be this model's own resting pressure at RB, otherwise the
reflex starts away from rest. So the first beats run open loop (HR fixed at
HR_rest), P_c is measured as the mean RB pressure of the last of those beats,
and then the loop is closed, well before the strain starts.

The Ursino results above are saved and put back at the end, so load_model.py
still reads the Ursino run in record_t, record_PA, ... The neurocontrol results
are kept in the neuro_* lists.
"""

# ---------------- NEUROCONTROL parameters ----------------
HR_rest_nc  = 64.0              # resting heart rate, bpm (Ogoh 2003, Table 1, rest)
HR_range_nc = 26.0              # HR_max - HR_min, bpm (Ogoh 2003, rest)
G_max_nc    = -0.35             # max gain at P_c, bpm/mmHg (typical resting value, assumed)
HR_max_nc   = HR_rest_nc + HR_range_nc/2     # highest HR (low pressure), 77 bpm
HR_min_nc   = HR_rest_nc - HR_range_nc/2     # lowest HR (high pressure), 51 bpm
k_nc        = -4*G_max_nc/HR_range_nc        # steepness, 1/mmHg (from G_max = -(HR_max - HR_min)*k/4)
t_close_nc  = 20.0              # s: open loop before this time, closed loop after (strain starts at start_time_VM)
n_setpoint_nc = 5               # number of beats averaged to find P_c
n_beats_nc  = n_beats           # same number of beats as the Ursino run

def HR_neurocontrol(P):
    """NEUROCONTROL: heart rate (bpm) for mean RB pressure P (mmHg), logistic carotid-HR curve."""
    return HR_min_nc + (HR_max_nc - HR_min_nc)/(1 + np.exp(k_nc*(P - P_c_nc)))

# ---------------- save the Ursino results ----------------
ursino_results = dict(record_t=record_t, record_x1=record_x1, record_x2=record_x2, record_x3=record_x3,
                      record_x4=record_x4, record_phase=record_phase, record_Pv=record_Pv, record_PA=record_PA,
                      record_PB=record_PB, record_T_beat=record_T_beat, record_meanPB=record_meanPB,
                      record_beat_start=record_beat_start, record_RR=record_RR, record_PAS=record_PAS)

# ---------------- fresh record lists (store() and last_x4() write/read these names) ----------------
record_x1, record_x2, record_x3, record_x4 = [], [], [], []
record_t, record_phase = [], []
record_Pv, record_PA, record_PB = [], [], []
record_T_beat, record_meanPB, record_beat_start, record_RR, record_PAS = [], [], [], [], []
neuro_HR_target = []            # NEUROCONTROL: HR asked for by the curve after each beat
neuro_t1, neuro_t2, neuro_t3, neuro_t_fill = [], [], [], []   # NEUROCONTROL: length of each phase per beat, s

# ---------------- NEUROCONTROL: start values ----------------
RR = RR_base                    # no resistance reflex in this run
RG = Rb + (R2*RR/(RR + R2))     # [G] eqn 16
P_reflex = Pn                   # contractility held at its set-point value
P_c_nc = None                   # centring pressure, measured before the loop is closed
T_beat = 60/HR_rest_nc          # first beats at the resting heart period
cycle_offset = 0

# ---------------- NEUROCONTROL: main loop (same four phases as above) ----------------
for i in range(n_beats_nc):
    PAS = float(PAS_total(cycle_offset, last_x4()))
    record_PAS.append(PAS)
    record_RR.append(RR)

    if i == 0:
        x0 = [57.51041422664778, 0.1101278932374876, 61.32586230866999, 0.11973411143947638]
        Ved = 0.14
    else:
        x0 = list(sol_filling.y[:, -1])

    # 1. isovolumic contraction (ends on a valve event)
    sol_contraction_nc = solve_ivp(isovolumic, t_span, x0, args=(Ved, cycle_offset), max_step=max_step,
                                   events=contraction_end_event)
    store(sol_contraction_nc, "contraction", cycle_offset)
    # 2. ejection (ends on a valve event)
    t0 = sol_contraction_nc.t[-1]
    sol_ejection_nc = solve_ivp(ejection, (t0, t0 + 1), list(sol_contraction_nc.y[:, -1]), args=(Ved, cycle_offset),
                                max_step=max_step, events=ejection_end_event)
    store(sol_ejection_nc, "ejection", cycle_offset)
    # 3. isovolumic relaxation (ends on a valve event, at most 0.25 s)
    t0 = sol_ejection_nc.t[-1]
    sol_relaxation_nc = solve_ivp(isovolumic, (t0, t0 + 0.25), list(sol_ejection_nc.y[:, -1]), args=(Ved, cycle_offset),
                                  max_step=max_step, events=relaxation_end_event)
    store(sol_relaxation_nc, "relaxation", cycle_offset)
    # 4. filling: NEUROCONTROL sets the whole beat T, the filling gets the rest
    #        t_fill = T - t1 - t2 - t3   (at least 0.01 s)
    t1 = sol_contraction_nc.t[-1] - sol_contraction_nc.t[0]   # isovolumic contraction, s
    t2 = sol_ejection_nc.t[-1] - sol_ejection_nc.t[0]         # ejection, s
    t3 = sol_relaxation_nc.t[-1] - sol_relaxation_nc.t[0]     # isovolumic relaxation, s
    t_fill = max(T_beat - t1 - t2 - t3, 0.01)                 # filling, s
    neuro_t1.append(t1); neuro_t2.append(t2); neuro_t3.append(t3); neuro_t_fill.append(t_fill)
    t0 = sol_relaxation_nc.t[-1]                              # filling starts here (= t1 + t2 + t3)
    sol_filling = solve_ivp(filling_rhs, (t0, t0 + t_fill), list(sol_relaxation_nc.y[:, -1]), args=(Ved, cycle_offset),
                            max_step=max_step)
    Ved = sol_filling.y[1, -1]
    store(sol_filling, "filling", cycle_offset)

    # ---- NEUROCONTROL step 1: RB measures the mean pressure of this beat ----
    tt = np.concatenate(record_t[-4:]); pp = np.concatenate(record_PB[-4:])
    meanPB = _trapz(pp, tt)/(tt[-1] - tt[0])
    record_beat_start.append(cycle_offset)
    record_T_beat.append(sol_filling.t[-1])
    record_meanPB.append(meanPB)
    t_end = cycle_offset + sol_filling.t[-1]

    if t_end < t_close_nc:
        # open loop: HR fixed at rest while the model settles
        HR_next = HR_rest_nc
    else:
        if P_c_nc is None:
            # set point = this model's resting RB pressure (mean of the last open-loop beats)
            P_c_nc = float(np.mean(record_meanPB[-n_setpoint_nc:]))
            print(f"NEUROCONTROL: loop closed at t = {t_end:.1f} s, set point P_c = {P_c_nc:.1f} mmHg")
        # ---- NEUROCONTROL step 2: map blood pressure to heart rate ----
        HR_next = HR_neurocontrol(meanPB)
    neuro_HR_target.append(HR_next)

    # ---- NEUROCONTROL step 3: heart period of the next beat ----
    T_beat = 60/HR_next

    cycle_offset += sol_filling.t[-1]

# ---------------- keep the NEUROCONTROL results, put the Ursino results back ----------------
neuro_t, neuro_PA, neuro_PB, neuro_Pv = record_t, record_PA, record_PB, record_Pv
neuro_phase, neuro_x2 = record_phase, record_x2
neuro_T_beat, neuro_meanPB = np.array(record_T_beat), np.array(record_meanPB)
neuro_beat_start, neuro_PAS = np.array(record_beat_start), np.array(record_PAS)
globals().update(ursino_results)

# ---------------- NEUROCONTROL plots (only plt) ----------------
if __name__ == "__main__":
    t_nc  = np.concatenate(neuro_t)
    pa_nc = np.concatenate(neuro_PA)
    pb_nc = np.concatenate(neuro_PB)

    plt.figure(figsize=(9, 9))

    plt.subplot(4, 1, 1)
    plt.plot(t_nc, pa_nc, lw=0.5, label="PA (aortic root)")
    plt.plot(t_nc, pb_nc, lw=0.5, label="PB (after Rb, RB input)")
    plt.axvspan(start_time_VM, end_time_VM, color="grey", alpha=0.2, lw=0)
    plt.ylabel("Pressure [mmHg]")
    plt.legend(loc="upper right", fontsize=8)
    plt.title("NEUROCONTROL: RB pressure -> logistic curve -> heart period")
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 2)
    plt.plot(neuro_beat_start, neuro_meanPB, ".-")
    plt.axhline(P_c_nc, color="grey", ls=":", label=f"set point P_c = {P_c_nc:.1f} mmHg")
    plt.axvline(t_close_nc, color="k", ls="--", lw=0.8, label="loop closed")
    plt.axvspan(start_time_VM, end_time_VM, color="grey", alpha=0.2, lw=0)
    plt.ylabel("mean PB [mmHg]")
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 3)
    plt.plot(neuro_beat_start, 60/neuro_T_beat, ".-", color="purple")
    plt.axhline(HR_rest_nc, color="grey", ls=":", label="resting HR")
    plt.axvline(t_close_nc, color="k", ls="--", lw=0.8)
    plt.axvspan(start_time_VM, end_time_VM, color="grey", alpha=0.2, lw=0)
    plt.ylabel("Heart rate [bpm]")
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 4)
    plt.step(neuro_beat_start, neuro_PAS, where="post", color="k")
    plt.axvspan(start_time_VM, end_time_VM, color="grey", alpha=0.2, lw=0)
    plt.ylabel("PAS [mmHg]")
    plt.xlabel("Time [s]")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("neurocontrol_HR.png", dpi=150)

    # the curve itself, with the simulated beats on it
    P_axis = np.linspace(min(neuro_meanPB.min(), P_c_nc - 2/k_nc) - 10, max(neuro_meanPB.max(), P_c_nc + 2/k_nc) + 10, 400)
    plt.figure(figsize=(7, 5))
    plt.plot(P_axis, HR_neurocontrol(P_axis), "k", lw=2, label="logistic carotid-HR curve")
    closed = neuro_beat_start[:-1] + neuro_T_beat[:-1] >= t_close_nc
    plt.plot(neuro_meanPB[:-1][closed], 60/neuro_T_beat[1:][closed], "o", color="purple", ms=4,
             label="simulated beats (P of beat n, HR of beat n+1)")
    plt.plot(P_c_nc, HR_rest_nc, "s", color="green", ms=7, label="set point")
    plt.plot(P_c_nc - 2/k_nc, HR_neurocontrol(P_c_nc - 2/k_nc), "v", color="red", ms=8, label="threshold")
    plt.plot(P_c_nc + 2/k_nc, HR_neurocontrol(P_c_nc + 2/k_nc), "^", color="blue", ms=8, label="saturation")
    plt.xlabel("mean pressure at RB [mmHg]")
    plt.ylabel("heart rate [bpm]")
    plt.title("NEUROCONTROL: carotid-HR curve")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("neurocontrol_curve.png", dpi=150)

    # T = t1 + t2 + t3 + t_fill: length of each phase per beat (stacked)
    plt.figure(figsize=(9, 4))
    plt.stackplot(neuro_beat_start, neuro_t1, neuro_t2, neuro_t3, neuro_t_fill,
                  colors=["#eb6834", "#2a78d6", "#e87ba4", "#1baf7a"],
                  labels=["t1 contraction", "t2 ejection", "t3 relaxation", "t_fill = T - t1 - t2 - t3"])
    plt.plot(neuro_beat_start, neuro_T_beat, "k.-", lw=1, label="T (set by NEUROCONTROL)")
    plt.axvline(t_close_nc, color="k", ls="--", lw=0.8)
    plt.axvspan(start_time_VM, end_time_VM, color="grey", alpha=0.2, lw=0)
    plt.xlabel("Time [s]")
    plt.ylabel("Duration [s]")
    plt.title("NEUROCONTROL: heart period split into the four phases")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("neurocontrol_durations.png", dpi=150)

    plt.show()
