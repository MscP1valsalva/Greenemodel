import numpy as np
from matplotlib import pyplot as plt

# fix_factor = 0.1 # could be another number
fix_factor = 0.0045 # could be another number

T = 1        # maximum time in seconds
t_step = 0.1e-3  # integration time step in seconds

h0 = 0.234  # mmHg
h1 = 147    # mmHg/l
h2 = 106000 # mmHg/l^2
Rm = 12   # mmHg.s/l
# Rm = 0.33   # mmHg.s/l
Pas = 10    # mmHg

N_max = int(T/t_step)
x2 = np.zeros(N_max)
Pv = np.zeros(N_max)

x2[0] = 0.02 # l (hypothetical end-diastolisk volume from beat before)
Pv[0] = (h2*x2[0]**2 + h1*x2[0] + h0) * fix_factor
for i in range(1,N_max):
    x2[i] = x2[i-1] + t_step * (Pas - Pv[i-1])/Rm
    Pv[i] = (h2*x2[i]**2 + h1*x2[i] + h0) * fix_factor

t = np.linspace(0,T,N_max)
plt.figure()
plt.plot(t, x2)
plt.xlabel('time (s)')
plt.ylabel('Vv (liter)')
plt.figure()
plt.plot(t, Pv)
plt.xlabel('time (s)')
plt.ylabel('Pv (mmHg)')
plt.show()