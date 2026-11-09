"""
load_model.py

Runs the Greene model (model_simulation.py) and makes its output ready to be
compared with a measurement.

Functions in this file:
    run_greene_model - run the model and get its time and aortic pressure
    sample           - sample a signal at a fixed rate, like the measuring equipment

Run this file directly to plot the model on top of one measurement.
"""

import runpy
from pathlib import Path

import numpy as np

# The model file lies in the same folder as this file
MODEL_FILE = Path(__file__).resolve().parent / "model_simulation.py"


def run_greene_model():
    """
        Runs model_simulation.py and returns: \n
        t : time [s] (numpy array) \n
        pa : aortic pressure [mmHg] (numpy array)

        The number of heart cycles is set in model_simulation.py (range(...) in its loop).

        The time steps in t are uneven, because the ODE solver (solve_ivp)
        chooses its own step size. Use sample() to get even time steps.
    """
    # runpy.run_path runs model_simulation.py from top to bottom, just as if you
    # ran it yourself, and gives back all the variables it made as a dictionary.
    # So model["record_t"] is the variable record_t from the model file
    model = runpy.run_path(str(MODEL_FILE))

    # The model stores each phase of each heartbeat (contraction, ejection,
    # relaxation, filling) as a separate array in a list:
    #     record_t = [phase 1 times, phase 2 times, phase 3 times, ...]
    # np.concatenate puts them after each other into one long array
    t = np.concatenate(model["record_t"])
    pa = np.concatenate(model["record_PA"])

    # One phase ends at exactly the same time as the next one starts, so some
    # times appear twice. np.unique keeps each time once (sorted), and index
    # tells which positions were kept, so we keep the matching pressures
    t, index = np.unique(t, return_index=True)
    return t, pa[index]


def sample(t, signal, fs):
    """
        t : time of the signal [s] (numpy array) \n
        signal : the signal (numpy array) \n
        fs : sampling frequency [Hz] \n

        Samples the signal at fs, just like the measuring equipment does: one
        value every 1/fs seconds (every 0.01 s at fs = 100 Hz).
        After this, the model and the measurement have the same time steps.

        Returns the new time axis and the sampled signal.
    """
    # The new, evenly spaced times: from the first to the last time, in steps of 1/fs
    t_sampled = np.arange(t[0], t[-1], 1 / fs)

    # np.interp finds the value of the signal at each new time. If a new time
    # lies between two model points, it draws a straight line between them and
    # reads the value off that line (linear interpolation)
    signal_sampled = np.interp(t_sampled, t, signal)

    return t_sampled, signal_sampled


"""
Run the file: simulate the Greene model and plot it on top of a measurement
"""
if __name__ == "__main__":
    # This part only runs when you run this file directly
    from preprocessing import load_data, systolic_blood_peaks
    import matplotlib.pyplot as plt

    fs = 100                # sampling frequency, the same as the measurements [Hz]
    adc_finapres = 100      # Finapres conversion: 1 V in the file = 100 mmHg
    subject = "subj 2.3.0211_Export.xls"

    # --- Measurement ---
    df = load_data(subject, fs)
    bp = df["Sys BP"].to_numpy() * adc_finapres    # [V] -> [mmHg]
    bp_peaks, rr = systolic_blood_peaks(bp, fs)

    # --- Model (number of cycles is set in model_simulation.py) ---
    t_model, pa_model = run_greene_model()
    t_model, pa_model = sample(t_model, pa_model, fs)
    model_peaks, _ = systolic_blood_peaks(pa_model, fs)

    # --- Line up the start ---
    # Cut out a piece of the measurement that is as long as the simulation,
    # starting 0.2 s before the first measured peak.
    # Then shift the model in time so its first peak lands on the first
    # measured peak. Only the start is shifted; the beats after that are not
    start = bp_peaks[0] - int(0.2 * fs)    # sample where the cut-out starts
    n = len(pa_model)                      # number of samples in the simulation
    bp = bp[start:start + n]
    t = np.arange(n) / fs                  # time axis for the cut-out [s]
    shift = (model_peaks[0] - int(0.2 * fs)) / fs
    t_model = t_model - t_model[0] - shift

    # --- Peaks to mark in the plot ---
    model_rr = np.diff(model_peaks) / fs   # time between model beats [s]

    # The measured peaks that lie inside the cut-out, counted from its start
    measured_peaks = bp_peaks[(bp_peaks >= start) & (bp_peaks < start + n)] - start

    # --- Plot ---
    plt.figure()
    linewidth = 0.5
    plt.plot(t, bp, linewidth=linewidth, label="Measurement")
    plt.plot(t_model, pa_model, linewidth=linewidth, label="Model")
    plt.plot(t[measured_peaks], bp[measured_peaks], "o", label="Measured peaks")
    plt.plot(t_model[model_peaks], pa_model[model_peaks], "x", label="Model peaks")
    plt.show()
