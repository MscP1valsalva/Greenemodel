"""
preprocessing.py

Loads and cleans the measurements in DataFolder, and finds the heartbeats.

Each measurement file (*_Export.xls) contains these columns, sampled at 100 Hz in our experiment:
    Time   : time [s]
    ECG I  : ECG lead I [V]
    ECG II : ECG lead II [V]
    Sys BP : the continuous blood pressure curve from the Finapres [V] - From iWorks this is volts and the ADC conversion is 100 mmHg/V. 
                                                                         Confirmed by Noemi ;)

Functions in this file:
    load_data             - load one measurement file
    load_all_subjects     - load every measurement file in DataFolder
    ecg_processing        - filter an ECG signal with a butterworth filter
    systolic_blood_peaks  - find the heartbeats (systolic peaks) in a blood pressure curve
    choose_subject        - let you choose a measurement file in the terminal

Run this file directly to choose a subject and plot its ECG and blood pressure.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

# The folder with the measurement files. It lies next to this file:
#   DataProcessing/
#       preprocessing.py   <- this file  (Path(__file__))
#       DataFolder/        <- the data   (.parent / "DataFolder")
# Building the path from this file's location means the code works no matter
# which folder you run it from.
DATA_DIR = Path(__file__).resolve().parent / "DataFolder" # this is just a way for us to run the code independenly
                                                          # so we dont have to change the path everytime a different person tries to run it 


def load_data(file_name, fs):
    """
        file_name : name of the file in DataFolder, e.g. "subj 1.1.0211_Export.xls" (string) \n
        fs : sampling frequency of the recording [Hz] (float) \n

        Loads one measurement file and returns it as a pandas DataFrame: 
        Basically a matlab table (a table with one column per signal and one row per sample).

        The file ends in .xls, but it is NOT a real Excel file: it is plain text
        where the columns are separated by tabs. That is why it is read with
        read_csv(sep="\t") and not read_excel. Looove data formatting (shout out Hans) :-)

        One file can hold several recordings after each other. Each recording
        starts with its own header line ("Time  ECG I  ECG II ...") and its Time
        column starts again at 0. Also the reason why we dont use the time of the data. It is just plain useless. 
        
        This function removes those extra header lines and joins all recordings into one long 
        signal with one continuous time axis. This can then be used to figure out the total length of the 
        mesurement with len(T) and the sample rate. 
    """
    # Read the tab-separated text file. The first line becomes the column names. 
    # Would be this for us: (Time	ECG I	ECG II	Sys BP	Dia BP)
    df = pd.read_csv(DATA_DIR / file_name, sep="\t")

    # The extra header lines inside the file end up as rows where the "Time"
    # column contains the text "Time". Keep only the rows that are NOT such a
    # header, and turn all values into numbers (they were read as text)
    df = df[df["Time"] != "Time"].astype(float)

    # After removing rows, the row numbers have holes in them (0, 1, 2, 5, 6, ...).
    # reset_index numbers the rows again from 0 without holes
    df = df.reset_index(drop=True)

    # Make one continuous time axis: row number / sampling frequency.
    # Row 0 -> 0 s, row 1 -> 0.01 s, row 100 -> 1 s (at fs = 100 Hz).
    # This replaces the original Time column, which starts at 0 for every recording
    df["Time"] = df.index / fs
    return df # this is our matlab lookalike dataset for a single subject. 


def load_all_subjects(fs):
    """
        fs : sampling frequency of the recordings [Hz] (float) \n 
        This function requiers that all mesuremnts have the same sampling freqency

        Loads every *_Export.xls file in DataFolder.
        Returns a dictionary where the key is the subject name and the value is
        its DataFrame, e.g. {"subj 1.1.0211_Export": DataFrame, ...}
        Dictionary are just the file path name and the corresponding dataFrame. 
    """
    subjects = {} # dictionary placeholder

    # glob finds all files whose name matches the pattern so just looks for ending with "Expoer.xls" and the sort just sorts it.
    for file in sorted(DATA_DIR.glob("*_Export.xls")):
        # file.stem is the file name without ".xls", file.name is with it
        subjects[file.stem] = load_data(file.name, fs)

    return subjects


def ecg_processing(signal, fs):
    """
        signal : ECG signal (numpy array) \n
        fs : sampling frequency of the recording [Hz] (float) \n

        Filters the ECG with a band-pass filter that keeps 0.5 - 40 Hz:
          - below 0.5 Hz: removes baseline wander (the slow drifting up and down,
            for example from breathing or movement)
          - above 40 Hz: removes high-frequency noise
        The heartbeat itself lies between these, so it is kept.

        Returns the filtered ECG signal.
    """
    # The frequencies to keep [Hz]
    lowcut = 0.5
    highcut = 40.0

    # butter() wants the frequencies as a fraction of the Nyquist frequency.
    # The Nyquist frequency is half the sampling frequency (50 Hz at fs = 100 Hz),
    # and is the highest frequency that can be seen in a signal sampled at fs
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design a 4th order Butterworth band-pass filter.
    # b and a are the filter coefficients that describe the filter
    b, a = butter(4, [low, high], btype="band")

    # Apply the filter. filtfilt runs the filter forwards and then backwards,
    # so the filtered signal is not shifted in time compared to the original
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def systolic_blood_peaks(signal, fs):
    """
        signal : blood pressure signal (numpy array) \n
        fs : sampling frequency of the recording [Hz] (float) \n

        Finds the systolic peaks (the top of each heartbeat) in a blood pressure curve.

        Returns: \n
        peaks : the sample number (index) of each peak. Use t[peaks] to get the
                times and signal[peaks] to get the systolic pressures \n
        rr : time between beats [s], with the gaps from Finapres calibrations removed
    """
    # --- Step 1: how big is the pulse around each sample? ---
    # The pulse (difference between the top and bottom of a beat) changes a lot
    # during a recording, e.g. from 20 to 150 mmHg during a Valsalva manoeuvre.
    # So instead of one size for the whole recording, measure it locally:
    # for every sample, take the highest minus the lowest value within a
    # 4 second window around it.
    s = pd.Series(signal)   # pandas Series, so we can use .rolling()
    n = int(4 * fs)         # 4 seconds in samples
    rolling_max = s.rolling(n, center=True, min_periods=1).max()
    rolling_min = s.rolling(n, center=True, min_periods=1).min()
    local_pulse = (rolling_max - rolling_min).to_numpy()

    # --- Step 2: find all candidate peaks ---
    # distance: two peaks must be at least 0.4 s apart. That allows heart rates
    #           up to 150 bpm (60 / 0.4) and stops small bumps right after a real
    #           peak from being counted as a new beat.
    # prominence=0: find every peak, but also calculate its "prominence", which
    #           is how much the peak stands out above the signal around it.
    #           We use it in step 3.
    peaks, props = find_peaks(signal, distance=int(0.4 * fs), prominence=0)

    # --- Step 3: keep only the real heartbeats ---
    # A real systolic peak stands out by a large part of the local pulse. The
    # small bump after each peak (the dicrotic notch) does not.
    # Keep the peaks whose prominence is at least 30 % of the local pulse.
    keep = props["prominences"] >= 0.3 * local_pulse[peaks]
    peaks = peaks[keep]

    # --- Step 4: time between beats ---
    # np.diff gives the number of samples between each pair of neighbouring
    # peaks. Dividing by fs turns it into seconds.
    rr = np.diff(peaks) / fs

    # The Finapres calibrates itself every so often. Then the curve goes flat for
    # a few seconds and no beats are visible, so the interval across it is far
    # too long to be one heartbeat.
    # Compare each interval with the typical interval of the 9 intervals around
    # it (the median, which is not thrown off by one odd value), and drop the
    # intervals that are more than 1.5 times as long.
    typical = pd.Series(rr).rolling(9, center=True, min_periods=1).median().to_numpy()
    rr = rr[rr < 1.5 * typical]

    return peaks, rr


def choose_subject():
    """
        Shows the measurement files in DataFolder with a number and asks you to
        choose one in the terminal.

        Returns the chosen file (a Path), or None if you type q.

        Use it like this:
            file = choose_subject()
            if file is None:
                ...                       # you typed q
            df = load_data(file.name, fs)
    """
    files = sorted(DATA_DIR.glob("*_Export.xls"))

    # Keep asking until you give a valid answer
    while True:
        # Print the list, e.g. "  0: subj 1.1.0211_Export"
        # enumerate gives both the number (i) and the file
        print("\nSubjects:")
        for i, file in enumerate(files):
            print(f"  {i}: {file.stem}")

        # input() waits for you to type something and press Enter.
        # strip() removes spaces before and after
        choice = input("Choose a subject number (q to quit): ").strip()

        # q means quit
        if choice.lower() == "q":
            return None

        # isdigit() checks that the answer is a whole number (so "abc" is not
        # accepted), and the number must be one of the files in the list
        if choice.isdigit() and int(choice) < len(files):
            return files[int(choice)]

        # Anything else: tell the user, and the loop asks again
        print("Not a valid number, try again.")


"""
Run the file: choose a subject in the terminal, and its ECG and blood pressure
get plotted. Close the figure window to choose a new subject, or type q to quit.
"""
if __name__ == "__main__":
    # This part only runs when you run this file directly, not when another
    # file imports functions from it
    import matplotlib.pyplot as plt

    fs = 100                  # sampling frequency of the measurements [Hz]
    adc_finapres = 100        # Finapres conversion: 1 V in the file = 100 mmHg
    V_mV_conversion = 1000    # 1 V = 1000 mV

    while True:
        # Let the user choose a measurement file. None means they typed q
        file = choose_subject()
        if file is None:
            break

        # --- Load the chosen subject ---
        name = file.stem                   # used as the plot title
        df = load_data(file.name, fs)
        t = df["Time"].to_numpy()          # time axis [s]

        # .to_numpy() turns a column of the table into a plain numpy array
        ecg_lead_1 = df["ECG I"].to_numpy() * V_mV_conversion     # [V] -> [mV]
        ecg_lead_2 = df["ECG II"].to_numpy() * V_mV_conversion    # [V] -> [mV]

        # --- Blood pressure ---
        bp = df["Sys BP"].to_numpy() * adc_finapres               # [V] -> [mmHg]
        bp_peaks, rr = systolic_blood_peaks(bp, fs)
        heart_rate = 60 / rr       # heart rate for each beat [bpm] (60 s / time per beat)

        # --- Filter the ECG signals ---
        # Lead I is multiplied by -1 (flipped upside down) so its R-peaks point up
        ecg_lead_1_processed = -ecg_processing(ecg_lead_1, fs)
        ecg_lead_2_processed = ecg_processing(ecg_lead_2, fs)

        # --- Plot: three plots above each other with a shared time axis ---
        # sharex=True: zooming in one plot zooms in all three
        linewidth = 0.2
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        fig.suptitle(name)

        ax1.plot(t, ecg_lead_1_processed, linewidth=linewidth)
        ax1.set_ylabel("ECG Lead I (mV)")

        ax2.plot(t, ecg_lead_2_processed, linewidth=linewidth)
        ax2.set_ylabel("ECG Lead II (mV)")

        ax3.plot(t, bp, linewidth=linewidth)
        ax3.plot(t[bp_peaks], bp[bp_peaks], "x", label="Systolic Peaks")   # an x on each peak
        ax3.set_ylabel("Blood Pressure (mmHg)")
        ax3.set_xlabel("Time [s]")

        # plt.show() waits here until you close the figure window,
        # then the loop starts again and you can choose a new subject
        plt.show()
