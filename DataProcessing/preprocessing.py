from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "DataFolder"


def load_data(file_name, fs):
    """
        file_name: name of the file to load (string) \n
        fs : sampling frequency of the recording [Hz] (float) \n

        This function loads the data from the specified file, removes any extra header lines, and converts all data to numeric format. 
        The file is expected to be a tab-separated text file. 
        The function returns a DataFrame with a continuous time axis for all recordings.
        Remove the extra header lines inside the file, and make everything numbers. 
        If you look at the file you will see a new header files every so often that has to be removed. 
    """
    # The file is tab-separated text, so use read_csv (not read_excel)
    df = pd.read_csv(DATA_DIR / file_name, sep="\t")

    df = df[df["Time"] != "Time"].astype(float)

    # Put all recordings after each other, with one continuous time axis
    df = df.reset_index(drop=True)
    df["Time"] = df.index / fs
    return df


def load_all_subjects(fs):
    """
        Loads every *_Export.xls file in DataFolder. \n
        Returns a dictionary: {subject name: DataFrame}
    """
    subjects = {}
    for file in sorted(DATA_DIR.glob("*_Export.xls")):
        subjects[file.stem] = load_data(file.name, fs)
    return subjects

def ecg_processing(signal, fs):
    """
        signal: ECG signal (numpy array) \n
        fs : sampling frequency of the recording [Hz] (float) \n

        This function processes the ECG signal by applying a bandpass filter to remove noise and baseline wander. 
        It returns the filtered ECG signal.
    """
    from scipy.signal import butter, filtfilt
    import numpy as np

    # Design a bandpass filter
    lowcut = 0.5  # Hz
    highcut = 40.0  # Hz
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')

    # Apply the filter to the signal
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def systolic_blood_peaks(signal, fs):
    """
        signal: Blood pressure signal (numpy array) \n
        fs : sampling frequency of the recording [Hz] (float) \n

        This function detects the systolic blood pressure peaks in the given signal.
        Returns: \n
        peaks : indices of the detected peaks \n
        rr : time between beats [s], without the gaps from Finapres calibrations
    """
    from scipy.signal import find_peaks
    import numpy as np

    # Size of the pulse in the last few seconds (it changes a lot during a recording)
    s = pd.Series(signal)
    n = int(4 * fs)
    local_pulse = (s.rolling(n, center=True, min_periods=1).max()
                   - s.rolling(n, center=True, min_periods=1).min()).to_numpy()

    # Find all peaks at least 0.4 s apart (max 150 bpm)
    peaks, props = find_peaks(signal, distance=int(0.4 * fs), prominence=0)

    # Keep peaks that stand out by 30 % of the local pulse (ignores the dicrotic notch)
    keep = props["prominences"] >= 0.3 * local_pulse[peaks]
    peaks = peaks[keep]

    # Time between beats. During a calibration the Finapres shows no beats, so
    # the interval across it is far too long. Drop intervals that are more than
    # 1.5x the typical interval of the beats around it
    rr = np.diff(peaks) / fs
    typical = pd.Series(rr).rolling(9, center=True, min_periods=1).median().to_numpy()
    rr = rr[rr < 1.5 * typical]

    return peaks, rr


"""
Run the file: choose a subject in the terminal and it gets plotted
"""
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs = 100
    files = sorted(DATA_DIR.glob("*_Export.xls"))
    adc_finapres = 100 # mmHg/V
    V_mV_conversion = 1000 # mV/V

    while True:
        # Show the subjects with a number 
        print("\nSubjects:")
        for i, file in enumerate(files):
            print(f"  {i}: {file.stem}")

        choice = input("Choose a subject number (q to quit): ").strip()
        if choice.lower() == "q":
            break
        if not choice.isdigit() or int(choice) >= len(files):
            print("Not a valid number, try again.")
            continue

        # Load only the chosen subject
        file = files[int(choice)]
        name = file.stem
        df = load_data(file.name, fs)
        t = df["Time"].to_numpy()


        ecg_lead_1 = df["ECG I"].to_numpy() * V_mV_conversion
        ecg_lead_2 = df["ECG II"].to_numpy() * V_mV_conversion


        bp = (df["Sys BP"].to_numpy() * adc_finapres) 
        bp_peaks, rr = systolic_blood_peaks(bp, fs)
        heart_rate = 60 / rr   # bpm for each beat

        # Process the ECG signal
        ecg_lead_1_processed = -ecg_processing(ecg_lead_1, fs)
        ecg_lead_2_processed = ecg_processing(ecg_lead_2, fs)

        linewidth = 0.2

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        fig.suptitle(name)
        ax1.plot(t, ecg_lead_1_processed, linewidth = linewidth)
        ax1.set_ylabel("ECG Lead I (mV)")
        ax2.plot(t, ecg_lead_2_processed, linewidth = linewidth)
        ax2.set_ylabel("ECG Lead II (mV)")
        ax3.plot(t, bp, linewidth = linewidth)
        ax3.plot(t[bp_peaks], bp[bp_peaks], "x", label="Systolic Peaks")
        ax3.set_ylabel("Blood Pressure (mmHg)")
        ax3.set_xlabel("Time [s]")

        plt.show()   # close the figure window to choose a new subject
