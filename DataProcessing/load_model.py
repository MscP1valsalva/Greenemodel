import runpy
from pathlib import Path
import numpy as np

MODEL_FILE = Path(__file__).resolve().parent.parent / "model_simulation.py"


def run_greene_model():
    """
        Runs model_simulation.py and returns: \n
        t : time [s] (numpy array) \n
        pa : aortic pressure [mmHg] (numpy array)

        The model has uneven time steps, because the ODE solver picks its own steps.
    """
    # model_simulation.py calls plt.show() at the end. Turn it off while the model
    # runs, so its plot windows don't open and stop this program
    model = runpy.run_path(str(MODEL_FILE))   # runs the file and gives back its variables

    # The model saves each phase separately, so put them after each other
    t = np.concatenate(model["record_t"])
    pa = np.concatenate(model["record_PA"])

    # One phase ends at the same time the next one starts: remove the duplicates
    t, index = np.unique(t, return_index=True)
    return t, pa[index]


def sample(t, signal, fs):
    """
        t : time of the signal [s] \n
        signal : the signal \n
        fs : sampling frequency [Hz]

        Samples the signal at fs, just like the measuring equipment.
        Returns the new time axis and the sampled signal.
    """
    t_sampled = np.arange(t[0], t[-1], 1 / fs)
    signal_sampled = np.interp(t_sampled, t, signal)   # value at each new time
    return t_sampled, signal_sampled


"""
Run the file: simulate the Greene model and compare it with a measurement
"""
if __name__ == "__main__":
    from preprocessing import load_data, systolic_blood_peaks
    import matplotlib.pyplot as plt
    

    fs = 100                # same as the measurements [Hz]
    adc_finapres = 100      # mmHg/V
    subject = "subj 2.3.0211_Export.xls"

    t_model, pa_model = run_greene_model()
    t_model, pa_model = sample(t_model, pa_model, fs)
    model_peaks, _ = systolic_blood_peaks(pa_model, fs)

    # --- Measurement ---
    df = load_data(subject, fs)
    bp = df["Sys BP"].to_numpy() * adc_finapres
    bp_peaks, rr = systolic_blood_peaks(bp, fs)

    # Cut out a piece of the measurement as long as the simulation.
    # Start 0.2 s before the first peak, and shift the model so the first peaks line up
    start = bp_peaks[0] - int(0.2 * fs)
    n = len(pa_model)
    bp = bp[start:start + n]
    t = np.arange(n) / fs
    shift = (model_peaks[0] - int(0.2 * fs)) / fs
    t_model = t_model - t_model[0] - shift

    # --- Compare ---
    model_rr = np.diff(model_peaks) / fs
    measured_peaks = bp_peaks[(bp_peaks >= start) & (bp_peaks < start + n)] - start

    plt.figure()
    linewidth = 0.5
    plt.plot(t, bp, linewidth=linewidth, label="Measurement")
    plt.plot(t_model, pa_model, linewidth=linewidth, label="Model")
    plt.plot(t[measured_peaks], bp[measured_peaks], "o", label="Measured peaks")
    plt.plot(t_model[model_peaks], pa_model[model_peaks], "x", label="Model peaks")
    plt.show()
