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

# ======================================================================
# Heart parameters ([G] eqn 31) and the Valsalva manoeuvre
# ======================================================================
# Greene's values were fitted to dogs. Several of them are changed so the
# model gives human pressures and volumes. Every change is marked with
# "CHANGED FROM [G]".

# Mitral valve resistance [mmHg*s/l].
# CHANGED FROM [G]: 0.33 -> 12. With 0.33 the ventricle fills almost at once.
# A larger resistance makes the filling slower, so the filling phase becomes
# longer and more like a human heart.
Rm  = 12    # mitral valve resistance, mmHg*s/l (altered from 0.33 to 12 to match the correct physiological range)

# Atrial source pressure. In [G] there is one constant PAS = 10 mmHg ([G] eqn 31).
# Here it is split in two: PAS = PAS_L + PAS_R(t)
#   PAS_L : a constant part [mmHg]
#   PAS_R : a part that changes during the Valsalva manoeuvre (function below)
# Before and after the strain PAS = 5 + 5 = 10 mmHg, the same as Greene.
PAS_L=5

# Time of the Valsalva strain [s]. The strain starts at 58.4 s and lasts 15 s.
# The same start time is used in model_validation.py (VM_START), so the model
# and the measurement have the strain at the same time.
start_time_VM = 58.4
end_time_VM = start_time_VM + 15

def PAS_R(t):
    """
        t : global time [s] (time since the start of the simulation, not of the beat) \n

        The part of the atrial source pressure that changes during the strain [mmHg].
        Not from the papers: this is our own description of the venous return
        during the Valsalva manoeuvre.

          - before the strain: 5 mmHg
          - when the strain starts it jumps to 15 mmHg, and then falls
            exponentially (time constant tau = 2 s) towards 10 mmHg at the end
            of the strain.
          - after the strain: 5 mmHg again

        The exponential is normalised so it is exactly 15 at the start and
        exactly 10 at the end of the strain, so there is no jump at the end:
            exp_now = exp(-(t - start) / tau)      1 at the start, small at the end
            exp_end = exp(-(end - start) / tau)    the value of exp_now at the end
            (exp_now - exp_end) / (1 - exp_end)    goes from 1 (start) to 0 (end)

        Examples:
            PAS_R(0)    -> 5.0    (before the strain)
            PAS_R(58.4) -> 15.0   (start of the strain)
            PAS_R(60.4) -> 11.84  (2 s = one time constant into the strain)
            PAS_R(73.4) -> 10.0   (end of the strain)
            PAS_R(80)   -> 5.0    (after the strain)
    """
    tau = 2.0

    if t < start_time_VM:
        return 5.0

    elif t <= end_time_VM:
        # Normalised exponential decay (from 1 at the start to 0 at the end)
        exp_now = np.exp(-(t - start_time_VM) / tau)
        exp_end = np.exp(-(end_time_VM - start_time_VM) / tau)

        return 10.0 + (15.0 - 10.0) * (
            (exp_now - exp_end) / (1.0 - exp_end)
        )

    else:
        return 5.0

# Aortic valve resistance [mmHg*s/l].
# CHANGED FROM [G]: 0.6 -> 0.6*20 = 12, tuned so the pressure curve fits the measurements.
Ra = 0.6*20

# Scale factor for the passive pressure-volume curve (see Pv below).
# CHANGED FROM [G]: Greene's curve ([G] eqn 30) is for a dog heart (volumes of
# about 0.02-0.05 l). A human ventricle is about 0.1-0.15 l, and with
# Greene's numbers that would give a diastolic pressure of more than
# 1000 mmHg. Multiplying by 0.0045 gives a few mmHg at human volumes, e.g.
# Pv(0.11 l) = 5.8 mmHg.
Fix_factor_passive = 0.0045

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
    u=np.where((t_global>=start_time_VM)&(t_global<=end_time_VM),40,0)
    return u

# ======================================================================
# Arterial-load parameters ([G] eqn 32, circuit in [G] Fig. 2)
# ======================================================================
# The arterial tree is a pi-type circuit ([G] p. 128):
#
#   PA --+-- R1 -- CL          proximal part: ascending aorta and aortic arch
#        +-- RL -- ground      (RL = cerebral and coronary resistance)
#        +-- L -- Rb --+-- R2 -- CR      distal part: descending aorta,
#                      +-- RR -- ground  RR = systemic peripheral resistance
#
# The comments behind each value describe what we saw when tuning it.
CL = 2.35e-4 * 5 # the lower it is, the more pressure in the ascending aorta, but it reduces stroke volume
R1 = 60 *3# sets the height of the notch, and also the height of the curve.
#R2=R1
CR = 1.33e-4 *8# the lower it is, the more pressure in the descending aorta, but it reduces stroke volume
R2 = 300.0*0.1 # a high value removes the notch, increases amplitude, lowers the cutting point. A 0.x value raises the cutting point, but also the pressure
RL = 12640.0 * 0.2 # the lower it is, the more volume ejected per beat.
Rb = 60.0 # if it increases, the pressure rises but the cut points do not.
# RR is not set here: it is set at the start of every beat in the loop below
# (RR = RR_base*kR), because the baroreflex changes it from beat to beat.
#RR = 8550.0 * 0.5 # the lower it is, the more volume ejected per beat
#RR=1710*0.8 #use for RR part later maybe
#RL=RR+1641
L  = 21.9  # inertance of the blood [mmHg*s^2/l], unchanged from [G]

# RT, [G] eqn 15: appears when the circuit is solved for the ejection phase,
# where Ra, R1 and RL all meet at the aortic root.
RT = R1* Ra + R1*RL + Ra*RL   # total resistance, mmHg*s/l
# RG, [G] eqn 16: Rb in series with R2 and RR in parallel. Also set in the
# loop, because RR changes every beat.
# Note: the printed eqn 16 reads Rb*(1 + R2/(R2+RR)); the version used here
# is the one that comes out when the circuit is solved (and matches eqns 6 and 10).
#RG = Rb+(R2*RR/(RR+R2))   # total resistance, mmHg*s/l (RR changed from Rb)
# when RL is larger than RR, the end of the notch moves up.

# ======================================================================
# Passive elastance ([G] eqns 18, 30) - the ventricle during filling
# ======================================================================
# Coefficients of Greene's passive pressure-volume curve, [G] eqn 30
h0, h1, h2 = 0.234, 147.0, 106000.0

def E_p(Vv):
    """
        Vv : ventricle volume [l] \n

        Passive (diastolic) elastance [mmHg/l]. Elastance is pressure divided
        by volume ([G] eqn 1), so dividing [G] eqn 18 by Vv gives
            E_p = Pv/Vv = h2*Vv + h1 + h0/Vv
        Not used in the simulation (Pv below is used instead), kept for reference.

        Example:
            E_p(0.1) -> 10749 mmHg/l
    """
    return h2*Vv + h1 + h0/Vv
    #return 2+h2*Vv + h1

def Pv(t_global, Vv):
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
    return ((h2*Vv**2 + h1*Vv + h0)+ITP(t_global)) * Fix_factor_passive

# ======================================================================
# Active elastance ([G] eqns 17, 20-29) - the ventricle during systole
# ======================================================================
# Greene describes the contraction as a sum of three Gaussian curves:
#     E_H(Ved, t) = sum_i A_i(Ved) * exp(-B_i(Ved)^2 * (t - C_i(Ved))^2) + D     [G] eqn 17
#   A_i = amplitude, B_i = width (bandwidth), C_i = time of the peak,
#   D   = a constant that sets the small elastance in diastole.
# One Gaussian fits an isovolumic beat. The ejecting beat is skewed, so two
# more Gaussians are added ([G] Fig. 4).
# A, B and C depend on the end-diastolic volume Ved (Frank-Starling). Greene
# fits them with 2nd-order polynomials, with a switch at Ved = 0.055 l for
# A1 and A3 ([G] p. 130).

def A1(Ved):
    """ [G] eqn 20. Example: A1(0.14) -> 10.59 """
    if Ved <= 0.055:
        return 2.1e-3*Ved**2 - 0.2338*Ved + 10.6
    else:
        return -1.0133e-3*Ved**2 - 0.0626*Ved + 10.6
def A2(Ved): return 150.0                                   # [G] eqn 21
def A3(Ved):
    """ [G] eqn 22. Example: A3(0.14) -> 10.17 """
    if Ved <= 0.055:
        return 2.8e-3*Ved**2 - 0.3*Ved + 10.2
    else:
        return 6.476e-4*Ved**2 - 0.1816*Ved + 10.2
def B1(Ved): return -1e-4*Ved + 0.0157                      # [G] eqn 23
def B2(Ved): return 45.0                                    # [G] eqn 24
def B3(Ved): return -4e-4*Ved + 0.0431                      # [G] eqn 25
def C1(Ved): return 9.7e-6*Ved**2 - 2.8e-4*Ved + 0.12       # [G] eqn 26
def C2(Ved): return 0.06                                    # [G] eqn 27
def C3(Ved): return -40.2e-6*Ved**2 + 4.78e-3*Ved + 0.06    # [G] eqn 28

# CHANGED FROM [G]: D = 100 mmHg/l in [G] eqn 29.
# Here D = -150. With Ved in litres and t in seconds, B1 and B3 are so small
# (about 0.016 and 0.043) that Gaussian 1 and 3 are almost flat: together they
# add about +157 mmHg/l all the time. D = -150 cancels most of this, so the
# elastance in diastole is small (about 7 mmHg/l), and the contraction
# itself comes from Gaussian 2.
D = -150

# Not in [G]: scale factors for the amplitudes (A), widths (B) and peak
# times (C), tuned so the pressure curve has a human shape and timing.
# E.g. Gaussian 2 peaks at C2*scale_C = 0.06*5.5 = 0.33 s instead of 0.06 s.
scale_A = 6.5
scale_B = 0.4
scale_C = 5.5

def E_H(Ved, t):
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
    fs = (N_ursino(P_reflex) - f_min)/(f_max - f_min) # from 0 to 1
    afs = 1 + 0.5*fs - 0.35*fs**2
    bfs = 1 + 0.1*fs - 0.1*fs**2

    # Add up the three Gaussians of [G] eqn 17. The for loop takes one
    # (A, B, C) triple at a time.
    s = 0.0
    for A, B, C in [(A1(Ved),B1(Ved),C1(Ved)), (A2(Ved),B2(Ved),C2(Ved)), (A3(Ved),B3(Ved),C3(Ved))]:
        s += afs*scale_A*A * np.exp(-((scale_B*B)**2) * (t*bfs - scale_C*C)**2)
    return s + D

# ======================================================================
# The differential equations of the four phases ([G] eqns 3-14)
# ======================================================================
# All three functions have the form solve_ivp wants: f(t, x, *args) returns
# dx/dt. The extra arguments (Ved, cycle_offset) are passed with args=(...).
#   t            : local time, since the start of this beat [s]
#   cycle_offset : the global time when this beat started [s]
#   t + cycle_offset is the global time, used for ITP and PAS_R, which follow
#   the clock of the experiment and not the beat.
#
# How the equations come from the circuit (Kirchhoff's current law):
#   Distal node (after L and Rb): the flow x4 splits into R2-CR and RR, which gives
#       dx1/dt = (-x1 + RR*x4) / (CR*(R2 + RR))
#   This is the same in all phases, because the distal part never sees the valves.
#   Aortic root node PA: what happens here depends on whether the aortic valve
#   is open (ejection) or closed (the other phases).

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
    dx2 = (PAS_L+PAS_R(t+cycle_offset) - Pv(t+cycle_offset, x2) ) / Rm
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

# ======================================================================
# The ends of the phases (events for solve_ivp)
# ======================================================================
# solve_ivp checks the event function at every step. When it crosses zero in
# the given direction, the phase stops:
#   .terminal = True  -> stop the integration at the zero crossing
#   .direction = -1   -> only count crossings from positive to negative
#   .direction = +1   -> only count crossings from negative to positive
# The valves are ideal ([G] p. 128): they open and close exactly when the
# pressures on the two sides are equal, so the events compare pressures.
# There is no event for the filling phase: it ends when the heart period
# T_beat set by the baroreflex is over (see the loop).

def contraction_end_event(t, x, Ved, cycle_offset):
    """
        End of isovolumic contraction: the aortic valve opens when Pv reaches PA.
        Returns PA - Pv. It is positive at the start (Pv low) and becomes
        negative when Pv passes PA, so direction = -1.
        dx3 is from the isovolumic equations ([G] eqn 9), the phase that is running.
    """
    x1, x2, x3, x4 = x   # x1=P_CR, x2=Vv, x3=P_CL, x4=flow through L
    dx3 = (-x3-RL*x4) / (CL * (R1 + RL))
    Pv_active = (E_H(Ved, t) * x[1]+ITP(t+cycle_offset))
    #print("Contraction phase: Pv_active =", Pv_active, "P_A =", P_A(x3, dx3), "x1 =", x1, "x2 =", x2, "x3 =", x3, "x4 =", x4)
    return P_A(x3, dx3) - Pv_active   # 0 when Pv reaches P_A
contraction_end_event.terminal = True
contraction_end_event.direction = -1 # PA - Pv goes from positive to negative when Pv becomes greater than PA


def ejection_end_event(t, x, Ved, cycle_offset):
    """
        End of ejection: the aortic valve closes when Pv falls below PA.
        Returns PA - Pv. It is negative during ejection (Pv > PA) and becomes
        positive when Pv falls below PA, so direction = +1.
        dx3 is from the ejection equations ([G] eqn 13).
    """
    x1, x2, x3, x4 = x   # x1=P_CR, x2=Vv, x3=P_CL, x4=flow through L
    dx3 = (RL * (E_H(Ved,t)*x2+ITP(t+cycle_offset))-(RL+Ra)*x3-RL*Ra*x4) / (CL * RT)
    Pv_active = (E_H(Ved, t) * x[1]+ITP(t+cycle_offset))
    #print("Ejection phase: Pv_active =", Pv_active, "P_A =", P_A(x3, dx3), "x1 =", x1, "x2 =", x2, "x3 =", x3, "x4 =", x4)
    return P_A(x3, dx3) - Pv_active   # 0 when Pv reaches P_A
ejection_end_event.terminal = True
ejection_end_event.direction = +1 # PA - Pv goes from negative to positive when Pv becomes less than PA


def relaxation_end_event(t, x, Ved, cycle_offset):
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
    return (PAS_L+PAS_R(t+cycle_offset) - 0.1) - Pv_active   # 0 when Pv reaches P_AS. NEW: was 9.9; now follows PAS so it still works when PAS is lowered
relaxation_end_event.terminal = True
relaxation_end_event.direction = +1 # (PAS - 0.1) - Pv goes from negative to positive when Pv becomes less than PAS


# ======================================================================
# Ursino (1998) baroreflex: heart period ([U] Eqs. 31, 33, 34, 38-42)
# ======================================================================
# The chain of the reflex:
#   pressure -> afferent firing f_cs ([U] Eq. 31)
#            -> sympathetic firing f_es ([U] Eq. 33) and vagal firing f_ev ([U] Eq. 34)
#            -> heart period T ([U] Eqs. 38-42)
# High pressure -> more afferent firing -> less sympathetic, more vagal firing
#               -> longer heart period (lower heart rate). And the other way round.
#
# Simplifications compared with [U] (because this model goes beat by beat):
#   - [U] uses the carotid sinus pressure, filtered by a dynamic block ([U] Eq. 30).
#     Here the mean pressure at Rb over one beat is used, without Eq. 30.
#   - [U] has time constants for the heart period (tau_T,s = 2 s, tau_T,v = 1.5 s,
#     [U] Eqs. 39 and 41). Here the static values are used directly.
#   - The pure delays (D_T,s = 2 s, D_T,v = 0.2 s, [U] Table 3) are made by
#     using the pressure from earlier beats (see the loop).

# Parameters from [U] Table 3:
#   Pn    : pressure at the middle of the sigmoid [mmHg]
#   ka    : slope parameter of the sigmoid [mmHg]
#   f_min, f_max : lower and upper saturation of the afferent firing [spikes/s]
Pn, ka, f_min, f_max = 92, 11.758, 2.52, 47.78              # carotid afferent ([U] Eq. 31, Table 3)
# Note: Pn is still Ursino's 92 mmHg. Earlier it was moved to 74.7 mmHg (this
# model's resting mean pressure at Rb, without the reflex). If the baseline
# changes, Pn can be changed here.
f_es_inf, f_es0, k_es, f_es_min = 2.10, 16.11, 0.0675, 2.66   # sympathetic ([U] Eq. 33, Table 3)
f_ev0, f_ev_inf, k_ev, f_cs0 = 3.2, 6.3, 7.06, 25.0           # vagal ([U] Eq. 34, Table 3)
# CHANGED FROM [U]: G_Tv = 0.09 s/v in [U] Table 3, here 0.15 (a stronger vagal effect).
G_Ts, G_Tv, T0 = -0.13, 0.15, 0.58                            # heart period ([U] Eqs. 38, 40, 42, Table 3)

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

    # Stop the program on invalid values
    if not np.isfinite(x):
        raise ValueError(
            f"Invalid value in N_ursino: "
            f"P={P}, Pn={Pn}, ka={ka}, x={x}"
        )

    # Limit the exponent, so np.exp does not overflow
    x = np.clip(x, -50.0, 50.0)

    e = np.exp(x)

    return (f_min + f_max * e) / (1.0 + e)

def sym_ursino(P):
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

def para_ursino(P):
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
        P_sym : pressure that drives the sympathetic part [mmHg] (from a few beats ago) \n
        P_vag : pressure that drives the vagal part [mmHg] (from the last beat) \n

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
    return T0 + dT_s + dT_v

# np.trapz was renamed np.trapezoid in numpy 2.0. This line uses the new name
# if it exists, and the old one otherwise, so the code runs on both versions.
_trapz = getattr(np, "trapezoid", None) or np.trapz
T_beat = heart_period(Pn,Pn)       # heart period of the first beat (resting value, 1.152 s)

# ======================================================================
# Ursino (1998) baroreflex: peripheral resistance ([U] Eqs. 35-37, Table 3)
# ======================================================================
# In [U] the systemic circulation has two beds in parallel ([U] Fig. 1):
# splanchnic (sp: gut, liver, spleen) and extrasplanchnic (ep: the rest).
# The resistance of each bed follows the sympathetic firing:
#   static curve   sigma = G*ln(f_es(t - D) - f_es_min + 1)   [U] Eq. 35
#   low-pass       d(dR)/dt = (sigma - dR) / tau             [U] Eq. 36
#   resistance     R = R0 + dR                               [U] Eq. 37
# Greene's model has only one peripheral resistance RR. So the Ursino
# resistance is not used directly: only its relative change is used, and RR
# is scaled by it:
#   kR = TPR(now) / TPR(rest)   and   RR = RR_base*kR
# So kR = 1 at rest, and kR = 1.2 means 20 % more peripheral resistance.

G_Rsp, Rsp0_u = 0.695, 2.49        # splanchnic gain and resistance without innervation, mmHg s/ml ([U] Table 3)
G_Rep, Rep0_u = 0.53, 0.78         # extrasplanchnic gain and resistance without innervation, mmHg s/ml ([U] Table 3)
tau_R, D_R = 6.0, 2.0              # time constant and pure delay, s ([U] Table 3, the same for sp and ep)

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
    return G*np.log(np.maximum(fes - f_es_min, 0) + 1)

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

fes_ref = sym_ursino(Pn)                                       # sympathetic firing at the set point (4.67 spikes/s)
dRsp, dRep = sigma_R(G_Rsp, fes_ref), sigma_R(G_Rep, fes_ref)  # start at the resting values (no transient at t = 0)
TPR_ref = TPR_u(dRsp, dRep)                                    # resting Ursino TPR, the reference for kR
RR_base = 8550*0.2                                             # resting RR. CHANGED FROM [G]: 8550 -> 8550*0.2
kR = 1.0                                                       # RR scale factor used by the next beat
fes_hist_t, fes_hist = [0.0], [fes_ref]                        # sympathetic firing history (time, value), for the delay

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

def lag(Delta, s, T, tau):
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


# ======================================================================
# Solving the equations
# ======================================================================
# Lists that are filled during the simulation. Each phase of each beat adds
# one numpy array, so e.g. record_t = [times of phase 1, times of phase 2, ...].
# load_model.py puts them together with np.concatenate.
record_x1, record_x2, record_x3, record_x4 = [], [], [], []
record_t, record_phase = [], []
record_Pv, record_PA = [], []
record_PB,record_Pdistal = [],[]
record_P_sym, record_P_vag = [], []  # pressures used for the sympathetic and vagal part of the reflex
record_T_beat, record_meanPB, record_beat_start = [], [], []  # one value per beat
record_RR = []  # peripheral resistance RR of each beat

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


# solve_ivp settings
t_span = (0, 1)   # A maximal cutoff for every phase (this should never be reached, the events stop it first)
max_step = 1e-3   # The largest step solve_ivp may take [s]. Small steps give a smooth curve
                  # and make sure the events (valve openings) are not stepped over
cycle_offset = 0  # global time when the current beat starts [s]

#PAS_base, PAS_low = 10.0, 6.0          # mmHg
#pert_beats = range(8, 18)              # beats 9-18 (0-based index)
n_beats = 180  # number of heart beats to simulate. 180 beats at about 1 s cover the whole measurement
record_PAS = []  # atrial source pressure at the start of each beat
P_reflex = Pn    # pressure the reflex sees, used by E_H. Starts at the set point

for i in range(n_beats): # one heart beat per loop
    #print(f"Cycle {i+1}")
    #PAS = PAS_low if i in pert_beats else PAS_base
    PAS=PAS_L+PAS_R(cycle_offset)
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
    if len(record_meanPB) >= 4:  # make sure we have at least 4 beats to compute the sympathetic firing rate
        P_sym = record_meanPB[-4]
    else:
        P_sym = Pn
    P_vag = meanPB # calculate vagal firing rate based on the last beat
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

# Making the time axis
t_segments = []
for t in record_t:
    t_segments.append(t)


"""
Run the file: simulate the model and plot the aortic pressure, the heart
rate and the peripheral resistance, with the Valsalva strain marked.
(load_model.py runs this file with runpy, which does not run this part.)
"""
if __name__ == "__main__":
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
