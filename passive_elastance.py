"""
    Passive elastance curve according to Greene
"""

import numpy as np
from matplotlib import pyplot as plt

h0 = 0.234  # mmHg
h1 = 147    # mmHg/l
h2 = 106000 # mmHg/l^2

Vv = 1e-3 * np.linspace(0.0, 100.0, 100)

Pv = h2*Vv**2 + h1*Vv + h0

plt.plot(Vv, Pv)
plt.show()
plt.xlabel('Vv (liter)')
plt.ylabel('Pv (mmHg)')
plt.title('Greene - Equation 18')