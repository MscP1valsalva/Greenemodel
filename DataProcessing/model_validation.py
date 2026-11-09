"""
model_validation.py

Compares the Greene model with a measurement, to see how well the model
describes the measured blood pressure.

The model and the measurement are compared on the same time axis: sample 1
with sample 1, sample 2 with sample 2, and so on. The model is NOT adjusted to
fit the measurement. Only the samples during Finapres calibrations are left
out, because the measurement has no real blood pressure there and we dont want to compare the model with a flat line. :)

Functions in this file:
    valid_samples - which samples of the measurement can be used for the comparison
    compare       - Pearson correlation and RMSE between measurement and model. pearson correlation shows if the two curves 
                    rise and fall at the same times, RMSE shows if the level is right.

Run this file directly to choose a subject and compare it with the model.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from load_model import run_greene_model, sample
from preprocessing import load_data, systolic_blood_peaks, choose_subject


def valid_samples(bp_peaks, n):
    """
        bp_peaks : sample numbers of the measured systolic peaks (numpy array) \n
        n : number of samples in the measurement of the systolic peaks \n

        This function returns a True/False array with one value per sample:
        True = the sample can be compared, False = leave it out so that we don't compare the model with
               a flat line during a Finapres calibration.

        Left out are:
          - the samples before the first peak and after the last peak
          - every beat interval that spans a Finapres calibration. During a
            calibration the curve goes flat for a few seconds, so the interval
            between the peaks on each side is much longer than a normal beat.
            The same rule as in systolic_blood_peaks is used: longer than 1.5
            times the typical interval of the 9 beats around it. 

            Example:
            1.45  1.52  1.43  1.34  [4.27]  1.40  1.39  1.38  1.37
            the [4.27] is the calibration because it is much longer than the 8 beats surrounding it. This is then flagged as false
            
            note: 
            We might need to change the amounts of peak intervals we look at and how much longer it is because of the rapid change in VM.
            
    """
    # Number of samples between each pair of neighbouring peaks
    rr = np.diff(bp_peaks) # Outputs the time difference between beats

 
    typical = pd.Series(rr).rolling(9, center=True, min_periods=1).median().to_numpy()  # Typical interval around each beat (median of 9 neighbouring intervals) 
                                                                                        # There is a lot happening in this function, but it is a pandans function that uses a rolling window
                                                                                        # over the intervals and then converts it back into a numpy array. 
                                                                                        # the min_periods help the function at the start and end recording where there is missing intervals for the window
    valid = np.zeros(n, dtype=bool) # a setup for the for loop in the next couple of lines. 

    # Go through the beats one by one. For each beat:
    #   a  = sample of this peak
    #   b  = sample of the next peak
    #   ok = True if the interval is a normal beat, False if it spans a calibration
    # Then set all samples from a up to b to ok
    for a, b, ok in zip(bp_peaks[:-1], bp_peaks[1:], rr < 1.5 * typical):
        valid[a:b] = ok

    return valid


def compare(measured, model, valid):
    """
        measured : measured pressure (numpy array) \n
        model : model pressure on the same time axis as measured (numpy array) \n
        valid : True/False array from valid_samples \n

        Compares the two signals, using only the samples where valid is True.

        Returns: \n
        r : Pearson correlation, between -1 and 1. It shows if the two curves
            rise and fall at the same times (1 = perfectly together,
            0 = no relation, -1 = opposite). It does NOT look at the level:
            a model that is 20 mmHg too high everywhere can still get r = 1. \n
        rmse : Root Mean Square Error [mmHg], the typical size of the difference
            between model and measurement. This DOES show if the level is wrong.
    """
    # measured[valid] keeps only the samples where valid is True.
    # np.corrcoef gives a 2x2 table of correlations; [0, 1] is the one between
    # the two signals (the others are each signal with itself, which is always 1)
    r = np.corrcoef(measured[valid], model[valid])[0, 1]

    # RMSE: take the difference at each sample, square it (so negative and
    # positive differences don't cancel), take the mean, and the square root
    # to get back to mmHg
    rmse = np.sqrt(np.mean((measured[valid] - model[valid]) ** 2))

    return r, rmse


"""
Run the file: the model is simulated once, then you choose a subject in the
terminal and it is compared with the model. Close the figure window to choose
a new subject, or type q to quit.
"""
if __name__ == "__main__":
    # This part only runs when you run this file directly

    fs = 100              # sampling frequency of the measurements [Hz]
    adc_finapres = 100    # Finapres conversion: 1 V in the file = 100 mmHg

    # --- Simulate the model ---
    # The number of cycles is set in model_simulation.py.
    # The model is the same for every subject, so it only runs once, before the
    # menu. Then you don't have to wait for it every time you choose a subject
    print("Running the model...")
    t_model, pa_model = run_greene_model()
    t_model, pa_model = sample(t_model, pa_model, fs)       # same time steps as the measurement
    model_peaks, rr_model = systolic_blood_peaks(pa_model, fs)
    heart_rate_model = 60 / rr_model                         # [bpm] for each beat

    while True:
        # Let the user choose a measurement file. None means they typed q
        file = choose_subject()
        if file is None:
            break

        # --- Load the chosen subject ---
        name = file.stem                                     # used as the plot title
        df = load_data(file.name, fs)
        bp = df["Sys BP"].to_numpy() * adc_finapres          # [V] -> [mmHg]
        bp_peaks, rr = systolic_blood_peaks(bp, fs)
        heart_rate_measurement = 60 / rr                     # [bpm] for each beat

        print(f"Heart rate from measurement: {heart_rate_measurement.mean():.1f} bpm")
        print(f"Heart rate from model: {heart_rate_model.mean():.1f} bpm")

        # --- Same time axis ---
        # The model and the measurement usually have different lengths. Only
        # the time that both cover can be compared, so cut both to the shorter one.
        # These are copies (_cut), so the full model is still there for the
        # next subject you choose
        n = min(len(bp), len(pa_model))
        bp_cut = bp[:n]
        pa_cut = pa_model[:n]
        bp_peaks_cut = bp_peaks[bp_peaks < n]                # only the peaks inside the cut part

        # --- Compare ---
        valid = valid_samples(bp_peaks_cut, n)
        r, rmse = compare(bp_cut, pa_cut, valid)

        # valid.mean() is the fraction of True values, so x100 gives the percentage used
        print(f"Pearson r = {r:.3f}, RMSE = {rmse:.1f} mmHg ({100 * valid.mean():.0f} % of samples used)")

        # --- Plot ---
        t = np.arange(n) / fs                                # time axis [s]

        plt.figure()
        plt.suptitle(name)
        linewidth = 0.5
        plt.plot(t, bp_cut, linewidth=linewidth, label="Measurement")
        plt.plot(t, pa_cut, linewidth=linewidth, label="Model")

        # Grey areas where valid is False (~ means "not"): the samples left out
        plt.fill_between(t, bp_cut.min(), bp_cut.max(), where=~valid, color="grey", alpha=0.3, label="Not used")

        plt.xlabel("Time [s]")
        plt.ylabel("Pressure [mmHg]")
        plt.legend()

        # plt.show() waits here until you close the figure window,
        # then the loop starts again and you can choose a new subject
        plt.show()
