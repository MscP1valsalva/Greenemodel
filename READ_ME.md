# A Model of the cardiovascular response to the Valsalva Maneuver

## Variable descriptions

## Heart parameters
- `Rm  = 12`    Is the mitral valve resistance in mmHg*s/l (altered from 0.33 to 12 to match the correct physiological range). This alteres the passive elastance function, so that the filling phase becomes longer.
- `Ra = 0.6`    Is the aortic valve resistance in mmHg*s/l, and is unaltered from Greenes et. al.'s values.
- `PAS = 10.0`  Is the atrial source pressure in mmHg. This variable is constant at this point, but in further work, this parameter should be variable and can be used to represent the changes due to intrathorasic pressure. This parameter is used as a threshold for the beginning and end of the filling phase.

## Arterial and peripheral load
The parameters inside the thorax is:
- `CL = 2.35e-4 *0.1`  # l/mmHg
- `R1 = 60*12`       # mmHg*s/l
- `CR = 1.33e-4`    # l/mmHg
- `R2 = 300.0 *2.4`     # mmHg*s/l
- `L  = 21.9`     # mmHg*s^2/l

The parameters outside of the thorax is:
- `RL = 12640.0 *2.5`    # mmHg*s/l
- `Rb = 60.0`       # mmHg*s/l
- `RR = 8550.0 *3`    # mmHg*s/l

- `RT = R1* Ra + R1*RL + Ra*RL`   # total resistance, mmHg*s/l
- `RG = Rb+(R2*RR/(RR+R2))`   # total resistance, mmHg*s/l (RR changed from Rb)

- `h0, h1, h2 = 0.234, 147.0, 106000.0`