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
    map_curve     - the MAP of each beat, as a curve with one value per sample
    compare_all_subjects - runs the comparison for every subject, so you can get the mean and standard deviation.
                           Also keeps the curves of every subject for plotting
    plot_subject  - plots the blood pressure and MAP of one subject against the model

Run this file directly to compare the model with all subjects, and then choose
a subject to plot.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from load_model import run_greene_model, sample
from preprocessing import (DATA_DIR, load_data, systolic_blood_peaks, choose_subject,
                           mean_artial_pressure, normal_beats)


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
            The rule for this is in normal_beats (preprocessing.py): longer than 1.5
            times the typical interval of the 9 beats around it.

            Example:
            1.45  1.52  1.43  1.34  [4.27]  1.40  1.39  1.38  1.37
            the [4.27] is the calibration because it is much longer than the 8 beats surrounding it. This is then flagged as false

            note:
            We might need to change the amounts of peak intervals we look at and how much longer it is because of the rapid change in VM.
            That is done in normal_beats, and then it changes everywhere.
    """
    valid = np.zeros(n, dtype=bool)   # start with every sample set to False (not used)

    # Go through the beats one by one. For each beat:
    #   a  = sample of this peak
    #   b  = sample of the next peak
    #   ok = True if the interval is a normal beat, False if it spans a calibration
    # Then set all samples from a up to b to ok
    for a, b, ok in zip(bp_peaks[:-1], bp_peaks[1:], normal_beats(bp_peaks)):
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


def map_curve(signal, peaks, t):
    """
        signal : blood pressure (numpy array) \n
        peaks : sample numbers of the systolic peaks (numpy array) \n
        t : time axis of the signal [s] (numpy array) \n

        Calculates the MAP of each beat (mean_artial_pressure) and turns it into
        a curve with one value per sample.

        Why: the model and the measurement have a different number of beats, at
        different times, so their MAP lists can't be paired value by value.
        np.interp draws a straight line between the MAP values of neighbouring
        beats and reads off the value at every time in t. Then the model and the
        measurement both have one MAP value per sample, and sample k of both
        belongs to the same time.

        Returns the MAP curve [mmHg], with the same length as t.
    """
    # One MAP value per beat, and the sample where each beat ends
    map_beat, beat_end = mean_artial_pressure(signal, peaks)

    # t[beat_end] is the time of each MAP value
    return np.interp(t, t[beat_end], map_beat)


def compare_all_subjects(pa_model, fs, adc_finapres):
    """
        pa_model : sampled model pressure [mmHg] (numpy array) \n
        fs : sampling frequency [Hz] \n
        adc_finapres : Finapres conversion [mmHg/V] \n

        Compares the model with every measurement file in DataFolder:
          1. cut the measurement and the model to the time both cover
          2. find the valid samples (no calibrations)
          3. compare the blood pressure curves with compare()
          4. compare the MAP curves with compare()

        Returns: \n
        table : a table (pandas DataFrame) with one row per subject and the
                columns: heart_rate, r, rmse, r_map, rmse_map, used_percent.
                Use table.mean() and table.std() to get the mean and standard
                deviation over all subjects. \n
        curves : a dictionary with the curves of each subject, so they can be
                 plotted later without calculating them again, e.g.
                 curves["subj 2.1"]["t"] is the time axis of subj 2.1.
                 The keys inside are: t, bp, pa, map_measurement, map_model, valid
    """
    # The model is the same for every subject, so find its peaks only once
    model_peaks, _ = systolic_blood_peaks(pa_model, fs)

    results = []    # one dictionary of numbers per subject, turned into a table at the end
    curves = {}     # the curves of each subject, for plotting

    for file in sorted(DATA_DIR.glob("*_Export.xls")):
        # Short name used as the row label and the key in curves.
        # rsplit cuts the name at the last dot: "subj 2.1.0211_Export" -> "subj 2.1"
        name = file.stem.rsplit(".", 1)[0]

        # --- Load the measurement and find its peaks (once) ---
        bp = load_data(file.name, fs)["Sys BP"].to_numpy() * adc_finapres   # [V] -> [mmHg]
        bp_peaks, rr = systolic_blood_peaks(bp, fs)

        # --- Same time axis ---
        # The model and the measurement usually have different lengths. Only
        # the time that both cover can be compared, so cut both to the shorter one.
        # These are copies (_cut), so the full model is still there for the next subject
        n = min(len(bp), len(pa_model))
        bp_cut = bp[:n]
        pa_cut = pa_model[:n]
        t = np.arange(n) / fs                                # time axis of the cut part [s]

        # Keep only the peaks inside the cut part. Then t[...] can never point
        # past the end of t
        bp_peaks_cut = bp_peaks[bp_peaks < n]
        model_peaks_cut = model_peaks[model_peaks < n]

        # --- Compare the blood pressure curves ---
        valid = valid_samples(bp_peaks_cut, n)
        r, rmse = compare(bp_cut, pa_cut, valid)

        # --- Compare MAP ---
        # Same function and the same valid samples as for the blood pressure curves
        map_measurement = map_curve(bp_cut, bp_peaks_cut, t)
        map_model = map_curve(pa_cut, model_peaks_cut, t)
        r_map, rmse_map = compare(map_measurement, map_model, valid)

        # --- Save the numbers of this subject (one row of the table) ---
        results.append({
            "subject": name,
            "heart_rate": (60 / rr).mean(),     # mean heart rate of the whole measurement [bpm]
            "r": r,
            "rmse": rmse,
            "r_map": r_map,
            "rmse_map": rmse_map,
            # valid.mean() is the fraction of True values, so x100 gives the percentage used
            "used_percent": 100 * valid.mean(),
        })

        # --- Save the curves of this subject, for the plot ---
        curves[name] = {
            "t": t,
            "bp": bp_cut,
            "pa": pa_cut,
            "map_measurement": map_measurement,
            "map_model": map_model,
            "valid": valid,
        }

    # Turn the list of dictionaries into a table: one row per subject, and the
    # dictionary keys become the column names. set_index makes the subject name
    # the row label, so mean() and std() only use the number columns
    table = pd.DataFrame(results).set_index("subject")
    return table, curves


def plot_subject(name, c):
    """
        name : subject name, used as the title (string) \n
        c : the curves of this subject, e.g. curves["subj 2.1"] from compare_all_subjects \n

        Two plots above each other with a shared time axis:
        Top: the blood pressure curves, to see the shape of the beats.
        Bottom: the MAP with its own y-axis, so its slow changes are visible
        (in the top plot they would look flat next to the big pulses).
        The samples that were left out (calibrations) are grey in both.
    """
    # sharex=True: zooming in one plot zooms in both
    # figsize = (width, height) of the window in inches
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 7))
    fig.suptitle(name)
    linewidth = 0.5

    ax1.plot(c["t"], c["bp"], linewidth=linewidth, label="Measurement")
    ax1.plot(c["t"], c["pa"], linewidth=linewidth, label="Model")
    ax1.set_ylabel("Pressure [mmHg]")
    ax1.set_title("Blood pressure")

    ax2.plot(c["t"], c["map_measurement"], linewidth=linewidth, label="MAP - Measurement")
    ax2.plot(c["t"], c["map_model"], linewidth=linewidth, label="MAP - Model")
    ax2.set_ylabel("MAP [mmHg]")
    ax2.set_xlabel("Time [s]")
    ax2.set_title("Mean arterial pressure")

    for ax in (ax1, ax2):
        # Grey areas where valid is False (~ means "not"): the samples left out.
        # transform=get_xaxis_transform() makes the grey area go from the bottom
        # (0) to the top (1) of each plot, whatever the y-values are
        ax.fill_between(c["t"], 0, 1, where=~c["valid"], color="grey", alpha=0.3,
                        transform=ax.get_xaxis_transform(), label="Not used")

        # Put the legend just to the right of the plot, so it doesn't cover
        # the curves. (1.01, 0.5) = just past the right edge, halfway up
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    # Make the two plots and their titles not overlap
    fig.tight_layout()

    # plt.show() waits here until you close the figure window
    plt.show()


"""
Run the file: the model is simulated once and compared with all subjects, and
a table with the results is printed. Then you choose a subject in the terminal
to see its plot. Close the figure window to choose a new subject, or type q to quit.
"""
if __name__ == "__main__":
    # This part only runs when you run this file directly

    fs = 100              # sampling frequency of the measurements [Hz]
    adc_finapres = 100    # Finapres conversion: 1 V in the file = 100 mmHg

    # --- Simulate the model ---
    # The number of cycles is set in model_simulation.py.
    # The model is the same for every subject, so it only runs once
    print("Running the model...")
    t_model, pa_model = run_greene_model()
    t_model, pa_model = sample(t_model, pa_model, fs)       # same time steps as the measurement
    _, rr_model = systolic_blood_peaks(pa_model, fs)
    print(f"Heart rate from model: {(60 / rr_model).mean():.1f} bpm")

    # --- Compare the model with all subjects ---
    # All calculations happen here, once. The menu below only shows the results
    print("Comparing the model with all subjects...")
    table, curves = compare_all_subjects(pa_model, fs, adc_finapres)

    # One row per subject, with 2 decimals
    print(table.round(2))

    # Mean and standard deviation over all subjects, for every column at once.
    # pandas std() is the sample standard deviation (divides by N - 1)
    print("\nOver all subjects:")
    print(table.agg(["mean", "std"]).round(2))

    while True:
        # Let the user choose a measurement file. None means they typed q
        file = choose_subject()
        if file is None:
            break

        # Look up the chosen subject (nothing is calculated again).
        # Same short name as used in compare_all_subjects: "subj 2.1"
        name = file.stem.rsplit(".", 1)[0]
        print(f"\n{name}:")
        print(table.loc[name].round(2))        # this subject's row in the table

        # Close the figure window to choose a new subject
        plot_subject(name, curves[name])
