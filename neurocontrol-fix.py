"""
model_simulation.py

Simulates the left ventricle and the arterial tree, beat by beat, during a
Valsalva manoeuvre (VM). The heart period is set by NEUROCONTROL, a simplified
carotid baroreflex: one logistic curve that maps the blood pressure at RB
straight to heart rate. (The Ursino reflex has been removed.)

The model is built from two papers:
    [G] Greene, Clark, Mohr & Bourland (1973). A mathematical model of
        left-ventricular function. Medical and Biological Engineering, 11,
        126-134.
        -> the heart (elastance), the four phases of the heart cycle and the
           arterial load (the circuit in Fig. 2, the equations 3-32).
    [K] Kent et al. (1972), the four-parameter logistic baroreflex curve, with the
        resting values from Ogoh et al. (2003), J Physiol 550:317 (Table 1).
        -> NEUROCONTROL: mean RB pressure -> heart rate -> heart period.
    In the comments, [G] eqn 12 means equation 12 in Greene.

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
    the next. After the filling, NEUROCONTROL looks at the mean pressure at RB
    over the beat and sets the length of the next beat.

NEUROCONTROL, every beat:
    1. RB measures the pressure: mean pressure after Rb over the beat (meanPB)
    2. the logistic carotid-HR curve maps meanPB to a heart rate:
           HR = HR_min + (HR_max - HR_min) / (1 + exp(k*(meanPB - P_c)))
             P > P_c: HR falls (parasympathetic side)
             P < P_c: HR rises (sympathetic side)
    3. heart period of the NEXT beat: T = 60/HR
       Phases 1-3 end on valve events (lengths t1, t2, t3), and the filling
       gets the rest of the beat: t_fill = T - t1 - t2 - t3
    Only the heart period is controlled. RR and the contractility are constant.
    The first beats run open loop (HR = HR_rest) so the model can settle; then
    P_c is set to the model's own resting RB pressure and the loop is closed.
    Every line in the main code that NEUROCONTROL changes is marked "NEUROCONTROL".f

VALSALVA FIX (every change is marked "VALSALVA FIX"):
    The strain pressure (ITP) now acts on everything in the chest: the
    ventricle (systole and filling), the atrium (PAS) and the ascending aorta
    (CL). Before, it only pushed on the ventricle, so the ventricle emptied
    itself into an aorta that did not feel the ITP (negative volumes, 1 s
    phases). The mitral valve stops backflow, the venous drive is cut to k_VM
    during the strain, and the resting circuit was retuned to normal values
    (BP ~138/80, MAP 93, SV 84 ml, CO 5.4 l/min; it was 87/17, MAP 33, SV 30 ml).

The four state variables (the same as in [G], below eqn 6):
    x1 : pressure across the distal compliance CR [mmHg]
    x2 : volume of the ventricle Vv [l]
    x3 : pressure across the proximal compliance CL [mmHg]
    x4 : flow through the inertance L [l/s]

Units: litres, seconds and mmHg, the same as Greene. So a resistance is in
mmHg*s/l and a compliance in l/mmHg.

Functions in this file:
    PAS_R, PAS_total, ITP   - the Valsalva inputs: atrial pressure and intrathoracic pressure
    E_p, Pv                 - passive (diastolic) elastance and pressure of the ventricle
    A1 ... C3, E_H          - active (systolic) elastance, a sum of three Gaussian curves
    filling_rhs, ejection,
    isovolumic              - the differential equations of the four phases
    P_A                     - aortic root pressure
    contraction_end_event,
    ejection_end_event,
    relaxation_end_event    - when the valves open and close (stops each phase)
    HR_neurocontrol         - NEUROCONTROL: the logistic carotid-HR curve
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

k_VM = 0.35  # fraction of the venous drive left during the strain. VALSALVA FIX: was 0.1; 0.35 gives a ~40 % SV drop (84 -> 50 ml)
def PAS_R(solved_x4):
        return solved_x4 * L


def PAS_total(t_global, x4):
    """
        Atrial source pressure [mmHg], PAS = PAS_L*k + PAS_R + ITP.
        VALSALVA FIX (PAS): the atrium is in the chest, so its pressure rises
        with the ITP (+l). The ventricle during filling also gets +ITP (see Pv),
        so the ITP itself cancels in PAS - Pv and does not change the filling.
        What lowers the venous return is the drive term PAS_L: during the strain
        it is cut to PAS_L*k_VM (8 -> 2.8 mmHg). Physiologically, PAS_L is the
        pressure that pushes venous blood from the veins outside the chest into
        the atrium. Those veins do not get the +40 mmHg, so this push falls.
    """
    k = np.where(ITP(t_global) > 0, k_VM, 1.0)
    l=ITP(t_global)
    return PAS_L*k + PAS_R(x4) + l   # VALSALVA FIX (PAS): drive PAS_L cut to k_VM in the strain, atrium pressure + ITP   # VALSALVA FIX: venous drive (PAS_L + PAS_R) cut to k_VM in the strain, + ITP because the atrium is in the chest


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
        VALSALVA FIX: the ITP is now added outside the bracket, so all 40 mmHg
        reach Pv (the ventricle sits in the chest).

        Examples:
            Pv(0, 0.10)  -> 4.84 mmHg
            Pv(0, 0.12)  -> 6.95 mmHg
            Pv(45, 0.10) -> 44.84 mmHg  (during the strain, ITP = 40)
    """
    return (h2*Vv**2 + h1*Vv + h0) * Fix_factor_passive + ITP(t_global) # VALSALVA FIX: + ITP, the ventricle is in the chest #TODO figuring out how to use fixfactor on PV part (fixed)

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

        NEUROCONTROL: the contraction no longer depends on a reflex. afs and bfs
        are constants (set with the elastance parameters below). They are the
        values the old Ursino formula gave at its set point (P = 92 mmHg):
            afs = 1.080 : multiplies the amplitude
            bfs = 1.018 : multiplies the time (> 1 = earlier, shorter curve)

        Example (Ved = 0.14 l):
            t [s]    0     0.1    0.2    0.3    0.4    0.5
            E_H      142   173    719    2147   1441   299    mmHg/l
            -> the peak of about 2250 mmHg/l (2.25 mmHg/ml) is at t = 0.32 s
    """
    # NEUROCONTROL: afs and bfs are now constants (see the parameters), the
   

    # Add up the three Gaussians of [G] eqn 17. The for loop takes one
    # (A, B, C) triple at a time.
    s = 0.0
    for A, B, C in [(A1(Ved),B1(Ved),C1(Ved)), (A2(Ved),B2(Ved),C2(Ved)), (A3(Ved),B3(Ved),C3(Ved))]:
        s += afs*scale_A*A * np.exp(-((scale_B*B)**2) * (t*bfs - scale_C*C)**2)
    return np.maximum(s + D, 0.0)   # VALSALVA FIX: elastance cannot be negative (with D = -150 the baseline was -4 mmHg/l)


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
    x3 = x3 + ITP(t+cycle_offset)   # VALSALVA FIX: CL is in the chest anatomically (must discuss with group), so the pressure at its node is x3 (transmural) + ITP. Must be on in EVERY phase, also filling
    dx1 = (-x1 + RR * x4) / (CR * (R2 + RR))
    PAS_t = float(PAS_total(t+cycle_offset, x4_beat))   # VALSALVA FIX: PAS follows the ITP during the beat, not only at its start
    dx2 = max(PAS_t - Pv(t+cycle_offset, x2), 0) / Rm      # VALSALVA FIX: max(...,0) = mitral valve, no backflow into the atrium
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
    x3 = x3 + ITP(t+cycle_offset)   # VALSALVA FIX: CL is in the chest, so the pressure at its node is x3 (transmural) + ITP
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
    x3 = x3 + ITP(t+cycle_offset)   # VALSALVA FIX: CL is in the chest, so the pressure at its node is x3 (transmural) + ITP
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
    x3 = x3 + ITP(t+cycle_offset)   # VALSALVA FIX: CL is in the chest, so the pressure at its node is x3 (transmural) + ITP
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
    x3 = x3 + ITP(t+cycle_offset)   # VALSALVA FIX: CL is in the chest, so the pressure at its node is x3 (transmural) + ITP
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
    return float(PAS_total(t+cycle_offset, x4_beat)) - Pv_active   # VALSALVA FIX: PAS at this moment. 0 when Pv reaches P_AS. NEW: was 9.9; now follows PAS so it still works when PAS is lowered


def HR_neurocontrol(P):
    """
        P : mean pressure at RB over one beat [mmHg] \n

        NEUROCONTROL: heart rate [bpm] from the logistic carotid-HR curve ([K]):
            HR = HR_min + (HR_max - HR_min) / (1 + exp(k*(P - P_c)))
        Steepest at P_c (max gain = -(HR_max - HR_min)*k/4). Above P_c the heart
        slows (parasympathetic side), below P_c it speeds up (sympathetic side).

        Examples (P_c = 32 mmHg):
            HR_neurocontrol(32) -> 64.0 bpm  (set point = resting HR)
            HR_neurocontrol(12) -> 72.4 bpm
            HR_neurocontrol(52) -> 55.6 bpm
    """
    return HR_min_nc + (HR_max_nc - HR_min_nc)/(1 + np.exp(k_nc*(P - P_c_nc)))

def RR_neurocontrol(EDV, ESV, HR, MAP):
    CO = (EDV - ESV) * HR/60
    print(MAP/CO)
    return MAP/CO


def store(sol, phase, cycle_offset):
    """
        sol : the result from solve_ivp for one phase \n
        phase : "contraction", "ejection", "relaxation" or "filling" \n
        cycle_offset : global time when this beat started [s] \n

        Saves the time and the four states of one phase in the record lists,
        and calculates three pressures from them:
            Pv : ventricle pressure
            PA : aortic root pressure (P_A), the one compared with the measurement
            PB : pressure after Rb, PA - Rb*x4  (what RB measures for NEUROCONTROL)
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

    x3_abs = sol.y[2] + ITP(plot_time)   # VALSALVA FIX: pressure at the CL node = x3 (transmural) + ITP
    if phase == "filling":
        pv_values = Pv(plot_time, sol.y[1])                     # passive pressure, [G] eqn 18
        dx3 = (-x3_abs - RL * sol.y[3]) / (CL * (R1 + RL))    # [G] eqn 5
        pa_values = P_A(x3_abs, dx3)
    elif phase == "ejection":
        pv_values = E_H(Ved, sol.t) * sol.y[1]+ITP(sol.t+cycle_offset)   # active pressure, [G] eqn 17
        dx3 = (RL * (E_H(Ved,sol.t)*sol.y[1]+ITP(sol.t+cycle_offset))-(RL+Ra)*x3_abs-RL*Ra*sol.y[3]) / (CL * RT)  # [G] eqn 13
        pa_values = P_A(x3_abs, dx3)
    else:  # contraction or relaxation
        pv_values = E_H(Ved, sol.t) * sol.y[1]+ITP(sol.t+cycle_offset)   # active pressure, [G] eqn 17
        dx3 = (-x3_abs-RL*sol.y[3]) / (CL * (R1 + RL))        # [G] eqn 9
        pa_values = P_A(x3_abs, dx3)
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
PAS_L=8   # VALSALVA FIX: was 5. 8 mmHg is a normal left-atrial pressure and gives EDV ~130 ml
start_time_VM = 40
end_time_VM = start_time_VM + 15
Ra = 0.6*20
Fix_factor_passive = 0.0045
# ======================================================================
# Arterial-load parameters ([G] eqn 32, circuit in [G] Fig. 2)
# ======================================================================
CL = 2.35e-4 * 10 # VALSALVA FIX: was *5 (lowers the pulse pressure). the lower it is, the more pressure in the ascending aorta, but it reduces stroke volume
R1 = 60 # VALSALVA FIX: was 60*3 (resting BP ~138/80 instead of 87/17). Sets the height of the notch, and also the height of the curve.
CR = 1.33e-4 *8# the lower it is, the more pressure in the descending aorta, but it reduces stroke volume
R2 = 300.0*0.1 # a high value removes the notch, increases amplitude, lowers the cutting point. A 0.x value raises the cutting point, but also the pressure
RL = 12640.0 * 0.2 # the lower it is, the more volume ejected per beat.
Rb = 60.0 # if it increases, the pressure rises but the cut points do not.
L  = 5.0   # SHAPE FIX: was 21.9 ([G], dog). Inertance of the blood [mmHg*s^2/l]. With 21.9 the L-CL/CR circuit rang at ~1.3 Hz
           # (a hump in diastole, backflow in x4); with 5 it rings at ~2.6 Hz and is damped, so it gives a dicrotic notch
RT = R1* Ra + R1*RL + Ra*RL   # total resistance, mmHg*s/l
h0, h1, h2 = 0.234, 147.0, 106000.0
# ======================================================================
# Active elastance ([G] eqns 17, 20-29) - the ventricle during systole
# ======================================================================
D = -150
scale_A = 13    # VALSALVA FIX: was 6.5. Stronger contraction, SV 30 -> 84 ml
scale_B = 0.17  # SHAPE FIX: was 0.2 (0.4 originally). Slightly wider elastance curve; a wider one (0.12) made the valves fail during the strain
scale_C = 4.7   # SHAPE FIX: was 5.5. Elastance peak earlier, so isovolumic contraction is ~0.12 s instead of ~0.19 s
afs = 1.080   # NEUROCONTROL: constant amplitude factor of E_H (was the  afs(fs); value at P = 92 mmHg)
bfs = 1.018   # NEUROCONTROL: constant time factor of E_H (was the Ursino bfs(fs); value at P = 92 mmHg)

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
# Peripheral resistance (NEUROCONTROL: constant, no reflex on RR)
# ======================================================================                # RR. CHANGED FROM [G]: 8550 -> 8550*0.2 #? but why?
RR = 8550*0.2                         # NEUROCONTROL: RR is constant (the Ursino RR reflex was removed)
RG = Rb + (R2*RR/(RR + R2))         # [G] eqn 16. Constant now that RR is constant

# ======================================================================
# NEUROCONTROL parameters ([K], resting values from Ogoh 2003)
# ======================================================================
HR_rest_nc  = 64.0              # resting heart rate, bpm (Ogoh 2003, Table 1, rest)
HR_range_nc = 26.0              # HR_max - HR_min, bpm (Ogoh 2003, rest)
G_max_nc    = -0.35             # max gain at P_c, bpm/mmHg (typical resting value, assumed)
HR_max_nc   = HR_rest_nc + HR_range_nc/2     # highest HR (low pressure), 77 bpm
HR_min_nc   = HR_rest_nc - HR_range_nc/2     # lowest HR (high pressure), 51 bpm
k_nc        = -4*G_max_nc/HR_range_nc        # steepness, 1/mmHg (from G_max = -(HR_max - HR_min)*k/4)
D_HR_nc     = 2.0               # HR DELAY: pure delay between the RB pressure and the heart rate, s
t_close_nc  = 20.0              # s: open loop before this time, closed loop after (strain starts at start_time_VM)
n_setpoint_nc = 5               # number of beats averaged to find P_c
P_c_nc = None                   # centring pressure, measured just before the loop is closed
_trapz = getattr(np, "trapezoid", None) or np.trapz #? this seems extensive. maybe just make sure you have the right numpy
T_beat = 60/HR_rest_nc          # NEUROCONTROL: heart period of the first beat = resting HR (0.9375 s)

# ======================================================================
# Solving the equations
# ======================================================================

record_x1, record_x2, record_x3, record_x4 = [], [], [], []
record_t, record_phase = [], []
record_Pv, record_PA = [], []
record_PB,record_Pdistal = [],[]
record_T_beat, record_meanPB, record_beat_start = [], [], []  # one value per beat
record_meanPB_time = []  # HR DELAY: global time at the end of each beat, when its meanPB was measured
record_HR_target = []  # NEUROCONTROL: HR asked for by the curve after each beat
record_t1, record_t2, record_t3, record_t_fill = [], [], [], []  # NEUROCONTROL: length of each phase, per beat [s]
record_Ved, record_Ves = [], [] # EDV and ESV 

# solve_ivp settings
t_span = (0, 1)   # A maximal cutoff for every phase (this should never be reached, the events stop it first)
max_step = 1e-3   # The largest step solve_ivp may take [s]. Small steps give a smooth curve
                  # and make sure the events (valve openings) are not stepped over
cycle_offset = 0  # global time when the current beat starts [s]

#PAS_base, PAS_low = 10.0, 6.0          # mmHg
#pert_beats = range(8, 18)              # beats 9-18 (0-based index)
n_beats = 100  # number of heart beats to simulate. 180 beats at about 1 s cover the whole measurement
record_PAS = []  # atrial source pressure at the start of each beat

t_segments = []
# ======================================================================
# Main model running
# ======================================================================

for i in range(n_beats): # one heart beat per loop
    #print(f"Cycle {i+1}")
    #PAS = PAS_low if i in pert_beats else PAS_base
      # fraction of venous return left during the strain (tune to get ~30-50 % SV drop)
    x4_beat = last_x4()   # VALSALVA FIX: x4 used by PAS_R during this beat (PAS itself is now computed at every time step)
    PAS = float(PAS_total(cycle_offset, x4_beat))
    record_PAS.append(PAS)
    # NEUROCONTROL: RR is constant now (set with the parameters), so the per-beat

    # --- Start state of this beat ---
    # First beat: a state taken from the end of an earlier run, so the model starts
    # close to a steady beat. Later beats: start where the last filling ended.
    if i == 0:
        x0 = [83.4146, 0.12793, 87.49025, 0.02377]   # SHAPE FIX: resting end-of-filling state with the new L, scale_B, scale_C
        Ved = 0.12793
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
    sol_contraction = sol_isovolumic   # NEUROCONTROL: keep the contraction result, t1 is needed for t_fill

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
    Ves = sol_ejection.y[1,-1] # ESV after the ejection 
    record_Ves.append(Ves) # stored the ESV for later use

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
    # NEUROCONTROL: the heart period T_beat sets the length of the filling.
    # Phases 1-3 end on valve events, so their lengths t1, t2, t3 are known once
    # they have run. The filling gets the rest of the beat:
    #     t_fill = T - t1 - t2 - t3
    # A higher HR (shorter T) therefore shortens the filling (diastole), as in vivo.
    # If t1 + t2 + t3 is longer than T, the filling still gets at least 0.01 s.
    x0 = [sol_isovolumic.y[0,-1], sol_isovolumic.y[1,-1], sol_isovolumic.y[2,-1], sol_isovolumic.y[3,-1]]  # final state of relaxation phase
    t1 = sol_contraction.t[-1] - sol_contraction.t[0]   # NEUROCONTROL: isovolumic contraction, s
    t2 = sol_ejection.t[-1] - sol_ejection.t[0]         # NEUROCONTROL: ejection, s
    t3 = sol_isovolumic.t[-1] - sol_isovolumic.t[0]     # NEUROCONTROL: isovolumic relaxation, s
    t_fill = max(T_beat - t1 - t2 - t3, 0.01)           # NEUROCONTROL: filling, s
    record_t1.append(t1); record_t2.append(t2); record_t3.append(t3); record_t_fill.append(t_fill)
    fill_starttime = sol_isovolumic.t[-1]               # local time where the filling starts (= t1 + t2 + t3)
    fill_span = (fill_starttime, fill_starttime + t_fill)
    sol_filling = solve_ivp(filling_rhs, fill_span, x0, args=(Ved,  cycle_offset),
                             max_step=max_step)
    Ved = sol_filling.y[1,-1] # End-diastolic volume = the volume at the end of filling, used by E_H in the next beat (Frank-Starling)
    record_Ved.append(Ved) # Store for later use
    #print("Filling phase completed at t =", sol_filling.t[-1])
    #print(sol_filling.status)
    store(sol_filling, "filling", cycle_offset)

    # ---- NEUROCONTROL step 1: RB measures the mean pressure of this beat ----
    # Mean pressure at Rb over this beat. The last 4 entries in the record lists
    # are the 4 phases of this beat. The mean is the area under the curve
    # (trapezoid rule) divided by the length of the beat.
    tt = np.concatenate(record_t[-4:]); pp = np.concatenate(record_PB[-4:])
    meanPB = _trapz(pp, tt)/(tt[-1] - tt[0])
    record_beat_start.append(cycle_offset)
    record_T_beat.append(sol_filling.t[-1])
    record_meanPB.append(meanPB)
    t_end = cycle_offset + sol_filling.t[-1]
    record_meanPB_time.append(t_end)        # HR DELAY: global time at the end of this beat (FIX: this is what was missing)

    #TODO: add RR neural control 
    if t_end < t_close_nc:
        # NEUROCONTROL open loop: HR fixed at rest while the model settles
        HR_next = HR_rest_nc
    else:
        if P_c_nc is None:
            # NEUROCONTROL set point = this model's resting RB pressure (mean of the last open-loop beats)
            P_c_nc = float(np.mean(record_meanPB[-n_setpoint_nc:]))
            #print(f"NEUROCONTROL: loop closed at t = {t_end:.1f} s, set point P_c = {P_c_nc:.1f} mmHg")
        # ---- NEUROCONTROL step 2: map the pressure from D_HR_nc seconds ago to heart rate ----
        # HR DELAY: only closed-loop measurements that are at least D_HR_nc old are used.
        # Search from the newest beat backwards and take the first one that is old enough.
        t_pressure_needed = t_end - D_HR_nc
        HR_next = HR_rest_nc                                             # HR DELAY: hold rest until a delayed value exists
        for j in range(len(record_meanPB) - 1, -1, -1):                  # HR DELAY: newest first
            if t_close_nc <= record_meanPB_time[j] <= t_pressure_needed:
                HR_next = HR_neurocontrol(record_meanPB[j])               # HR DELAY: pressure from >= 2 s ago
                break
    record_HR_target.append(HR_next)

    # ---- NEUROCONTROL step 3: heart period of the next beat ----
    T_beat = 60/HR_next

    if t_end > 6: 
        HR_needed = t_end - 6
        for i in range(len(record_HR_target) -1, -1, -1): 
            if 6 <= record_meanPB_time[i] <= HR_needed:
                RR = RR_neurocontrol(record_Ved[i], record_Ves[i], record_HR_target[i], record_meanPB[i]) # EDV, ESV and heart rate 6 seconds ago for the function to calculate the new RR
                break
  #print(f"NEUROCONTROL: mean P_RB = {meanPB:.1f} mmHg -> next T = {T_beat:.3f} s ({HR_next:.1f} bpm)")

    # The next beat starts where this one ended. sol_filling.t[-1] is the local
    # end time of the beat = the length of the beat
    cycle_offset += sol_filling.t[-1]

for t in record_t:
    t_segments.append(t)


# ======================================================================
# Plots (only when this file is run directly; only plt)
# ======================================================================
if __name__ == "__main__":
    PHASE_COLORS = {"contraction": "#eb6834", "ejection": "#2a78d6", "relaxation": "#e87ba4", "filling": "#1baf7a"}
    PHASE_LABELS = {"contraction": "Isovolumic contraction", "ejection": "Ejection",
                    "relaxation": "Isovolumic relaxation", "filling": "Filling"}
    phase_handles = [plt.Line2D([0], [0], color=PHASE_COLORS[p], lw=2, label=PHASE_LABELS[p]) for p in PHASE_COLORS]

    t_all  = np.concatenate(record_t)
    pa_all = np.concatenate(record_PA)
    pb_all = np.concatenate(record_PB)
    beat_start = np.array(record_beat_start)
    T_beats    = np.array(record_T_beat)
    meanPB_all = np.array(record_meanPB)

    # --------- NEUROCONTROL overview: pressure, mean RB pressure, heart rate, PAS ---------
    plt.figure(figsize=(9, 9))

    plt.subplot(4, 1, 1)
    plt.plot(t_all, pa_all, lw=0.5, label="PA (aortic root)")
    plt.plot(t_all, pb_all, lw=0.5, label="PB (after Rb, RB input)")
    plt.axvspan(start_time_VM, end_time_VM, color="grey", alpha=0.2, lw=0)
    plt.ylabel("Pressure [mmHg]")
    plt.legend(loc="upper right", fontsize=8)
    plt.title("NEUROCONTROL: RB pressure -> logistic curve -> heart period")
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 2)
    plt.plot(beat_start, meanPB_all, ".-")
    plt.axhline(P_c_nc, color="grey", ls=":", label=f"set point P_c = {P_c_nc:.1f} mmHg")
    plt.axvline(t_close_nc, color="k", ls="--", lw=0.8, label="loop closed")
    plt.axvspan(start_time_VM, end_time_VM, color="grey", alpha=0.2, lw=0)
    plt.ylabel("mean PB [mmHg]")
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 3)
    plt.plot(beat_start, 60/T_beats, ".-", color="purple")
    plt.axhline(HR_rest_nc, color="grey", ls=":", label="resting HR")
    plt.axvline(t_close_nc, color="k", ls="--", lw=0.8)
    plt.axvspan(start_time_VM, end_time_VM, color="grey", alpha=0.2, lw=0)
    plt.ylabel("Heart rate [bpm]")
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 4)
    plt.step(beat_start, record_PAS, where="post", color="k")
    plt.axvspan(start_time_VM, end_time_VM, color="grey", alpha=0.2, lw=0)
    plt.ylabel("PAS [mmHg]")
    plt.xlabel("Time [s]")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("neurocontrol_HR.png", dpi=150)

    # --------- NEUROCONTROL curve, with the simulated beats on it ---------
    P_axis = np.linspace(min(meanPB_all.min(), P_c_nc - 2/k_nc) - 10, max(meanPB_all.max(), P_c_nc + 2/k_nc) + 10, 400)
    closed = beat_start[:-1] + T_beats[:-1] >= t_close_nc
    plt.figure(figsize=(7, 5))
    plt.plot(P_axis, HR_neurocontrol(P_axis), "k", lw=2, label="logistic carotid-HR curve")
    plt.plot(meanPB_all[:-1][closed], 60/T_beats[1:][closed], "o", color="purple", ms=4,
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

    # --------- T = t1 + t2 + t3 + t_fill: length of each phase per beat (stacked) ---------
    plt.figure(figsize=(9, 4))
    plt.stackplot(beat_start, record_t1, record_t2, record_t3, record_t_fill,
                  colors=[PHASE_COLORS["contraction"], PHASE_COLORS["ejection"],
                          PHASE_COLORS["relaxation"], PHASE_COLORS["filling"]],
                  labels=["t1 contraction", "t2 ejection", "t3 relaxation", "t_fill = T - t1 - t2 - t3"])
    plt.plot(beat_start, T_beats, "k.-", lw=1, label="T (set by NEUROCONTROL)")
    plt.axvline(t_close_nc, color="k", ls="--", lw=0.8)
    plt.axvspan(start_time_VM, end_time_VM, color="grey", alpha=0.2, lw=0)
    plt.xlabel("Time [s]")
    plt.ylabel("Duration [s]")
    plt.title("Heart period split into the four phases")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("neurocontrol_durations.png", dpi=150)

    # --------- States x1-x4 ---------
    titles  = ["x1: pressure across $C_R$ (descending & abdominal aorta)", "x2: ventricular volume",
               "x3: pressure across $C_L$ (ascending aorta)", "x4: flow through $L$"]
    ylabels = ["Pressure (mmHg)", "Volume (L)", "Pressure (mmHg)", "Flow (L/s)"]
    data    = [record_x1, record_x2, record_x3, record_x4]
    plt.figure(figsize=(10, 6))
    for n, (title, ylabel, values) in enumerate(zip(titles, ylabels, data)):
        plt.subplot(2, 2, n + 1)
        for t, y, phase in zip(t_segments, values, record_phase):
            plt.plot(t, y, color=PHASE_COLORS[phase], lw=1)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.grid(True, color="#e1e0d9")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.figlegend(handles=phase_handles, loc="upper center", ncol=4)

    # --------- Ventricular and aortic pressure ---------
    plt.figure(figsize=(10, 6))
    for t, pv_values, pa_values, phase in zip(t_segments, record_Pv, record_PA, record_phase):
        plt.plot(t, pa_values, "r--", lw=1.5, zorder=3)
        plt.plot(t, pv_values, color=PHASE_COLORS[phase], lw=1)
    plt.step(record_beat_start, record_PAS, where="post", color="k", lw=1.5, zorder=2)
    plt.title("Ventricular and aortic pressure")
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure [mmHg]")
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.figlegend(handles=phase_handles + [plt.Line2D([0], [0], color="red", lw=2, label="Aortic pressure"),
                                           plt.Line2D([0], [0], color="black", lw=2, label="PAS")],
                  loc="upper center", ncol=5)

    # --------- Pressure-volume diagram ---------
    plt.figure(figsize=(10, 6))
    for x2_values, pv_values, phase in zip(record_x2, record_Pv, record_phase):
        plt.plot(x2_values, pv_values, color=PHASE_COLORS[phase], lw=1)
    plt.title("Pressure-Volume diagram")
    plt.xlabel("Ventricular volume (l)")
    plt.ylabel("Ventricular pressure [mmHg]")
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.figlegend(handles=phase_handles, loc="upper center", ncol=4)

    # --------- Pressure after Rb (what RB measures) ---------
    plt.figure(figsize=(10, 6))
    for t, pb, phase in zip(t_segments, record_PB, record_phase):
        plt.plot(t, pb, color=PHASE_COLORS[phase])
    plt.title("Pressure after Rb (RB input)")
    plt.xlabel("Time [s]")
    plt.ylabel("P [mmHg]")
    plt.grid(True)
    print("Mean pressure after Rb: ", np.mean(pb_all), "mmHg")

    plt.show()
