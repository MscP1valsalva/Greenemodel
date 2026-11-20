import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt # This order of importing is important
import pywt

DATA_DIR = Path(__file__).resolve().parent

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
    df = pd.read_csv(DATA_DIR /file_name, sep="\t")

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


fs = 100
adc_finapres = 100
file_name = "subj 2.4.0211_Export.xls"

bp = load_data(file_name, fs = fs)["Sys BP"].to_numpy() 
n = len(bp)
t = np.arange(n) / fs

# Define scales
scales = np.arange(1, 256)

# Compute CWT
coefficients, frequencies = pywt.cwt(
    bp,
    scales,
    'cmor1.5-1.0',           # complex Morlet wavelet: bandwidth 1.5, centre frequency 1.0
    sampling_period=1/fs
)

# 2 rows, 1 column. sharex=True: both plots use the same time axis
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Top: the blood pressure
ax1.plot(t, bp, linewidth=0.5)
ax1.set_ylabel("ECG [V]")
ax1.set_xlim(50, 100)

# Bottom: the CWT. cs is the drawn contour plot, needed for the colorbar
cs = ax2.contourf(t, frequencies, np.abs(coefficients), levels=100, cmap="viridis")
ax2.set_ylabel("Frequency [Hz]")
ax2.set_xlabel("Time [s]")
ax2.set_yscale("log")
#ax2.set_xlim(35, 100)       # sharex: this zooms the top plot too

# Put the colorbar in its own small axis just outside the right edge of ax2.
# [x, y, width, height] is relative to ax2: x = 1.02 is just past its right edge.
# ax2 itself keeps its full width, so the time axes of ax1 and ax2 line up
cax = ax2.inset_axes([1.02, 0, 0.02, 1])
fig.colorbar(cs, cax=cax, label="Magnitude")
fig.subplots_adjust(left=0.15, right=0.8)   # room for the y-label on the left and the colorbar on the right
fig.savefig(DATA_DIR / "CWT_original.png", dpi=300)

coeff = pywt.wavedec(bp, "db4", level = 10)

# One plot more than there are levels: the top one is for the original signal
fig, axs = plt.subplots(len(coeff) + 1, 1, figsize=(6, 10))   # smaller figure, the text keeps its size

# Top: the original signal, against sample number
axs[0].plot(bp, linewidth=0.5)   # thin line (default is 1.5)
axs[0].text(0.01, 0.95, "Original signal", transform=axs[0].transAxes,
            ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, linewidth=0))
axs[0].set_xlabel("Sample number", fontsize=9)
axs[0].tick_params(labelsize=9)

# The rest: the levels, in reversed order. coeff = [A10, D10, ..., D1], and
# axs[1:][::-1] lists the remaining plots from the bottom up. So coeff[0] (A10)
# goes in the bottom plot and D1 in the plot just under the original signal
axs = axs[1:][::-1]
for i, c in enumerate(coeff):
    axs[i].plot(c, linewidth=0.5)

# Write the name and frequency band of each level next to its plot.
# coeff = [A10, D10, D9, ..., D1]. Level j covers fs/2^(j+1) to fs/2^j Hz
level = len(coeff) - 1
for i in range(len(coeff)):
    if i == 0:
        name = f"A{level}"
        band = f"0-{fs / 2**(level + 1):.2f} Hz"
    else:
        j = len(coeff) - i
        name = f"D{j}"
        band = f"{fs / 2**(j + 1):.2f}-{fs / 2**j:.2f} Hz"
    # transform=axs[i].transAxes: x and y from 0 to 1 inside the plot, so the text
    # sits in the top left corner whatever the values are
    axs[i].text(0.01, 0.95, f"{name}: {band}", transform=axs[i].transAxes,
                ha="left", va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, linewidth=0))   # white box so the text is readable
    axs[i].tick_params(labelsize=9)

# Only the bottom plot gets the x-label. axs is reversed, so axs[0] is the bottom one (A10)
axs[0].set_xlabel("Coefficient number", fontsize=9)

# Only 2 numbers on each y-axis (the lowest and the highest), so they do not overlap.
# fig.axes is every plot in this figure, including the original signal at the top.
# get_yticks() can give numbers outside the plot, so keep only the ones inside the y-limits
for ax in fig.axes:
    bottom, top = ax.get_ylim()
    ticks = [y for y in ax.get_yticks() if bottom <= y <= top]
    ax.set_yticks([min(ticks), max(ticks)])

# Space between the levels. This must come after everything is drawn, and
# only once: a later tight_layout() would overwrite it.
# h_pad = space between the levels, in units of the font size (default 1.08)
fig.tight_layout(h_pad=0.1)

# --- Remove whole detail levels and rebuild the signal ---
# Which detail levels to remove. D1 = 25-50 Hz, D2 = 12.5-25 Hz (the fastest wiggles)
remove = [1, 2, 3, 4, 5, 6, 7, 8]

# Copy the coefficients, so the originals are not changed
coeff_clean = [c.copy() for c in coeff]

for j in remove:
    # coeff = [A5, D5, D4, ..., D1], so Dj sits at position len(coeff) - j
    coeff_clean[len(coeff) - j] = np.zeros_like(coeff_clean[len(coeff) - j])

# Show the removed levels in the DWT plot (they become flat lines at 0)
for i, c in enumerate(coeff_clean):
    axs[i].plot(c, linewidth=0.5)
fig.savefig(DATA_DIR / "DWT_levels.png", dpi=300)

# Rebuild the signal. waverec can give one extra sample (padding), so cut to n
bp_clean = pywt.waverec(coeff_clean, "db4")[:n]
"""
plt.figure()
plt.plot(t, bp, linewidth = 0.5)
plt.plot(t, bp_clean, linewidth = 0.5)
plt.legend(["Original", "Reconstructed"])
"""
coefficients, frequencies = pywt.cwt(
    bp_clean,
    scales,
    'cmor1.5-1.0',           # complex Morlet wavelet: bandwidth 1.5, centre frequency 1.0
    sampling_period=1/fs
)

# 2 rows, 1 column. sharex=True: both plots use the same time axis
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Top: the blood pressure
ax1.plot(t, bp_clean, linewidth=0.5)
ax1.set_ylabel("ECG [V]")
ax1.set_xlim(50, 100)

# Bottom: the CWT. cs is the drawn contour plot, needed for the colorbar
cs = ax2.contourf(t, frequencies, np.abs(coefficients), levels=100, cmap="viridis")
ax2.set_ylabel("Frequency [Hz]")
ax2.set_xlabel("Time [s]")
ax2.set_yscale("log")
#ax2.set_xlim(35, 100)       # sharex: this zooms the top plot too

# Put the colorbar in its own small axis just outside the right edge of ax2.
# [x, y, width, height] is relative to ax2: x = 1.02 is just past its right edge.
# ax2 itself keeps its full width, so the time axes of ax1 and ax2 line up
cax = ax2.inset_axes([1.02, 0, 0.02, 1])
fig.colorbar(cs, cax=cax, label="Magnitude")
fig.subplots_adjust(left=0.15, right=0.8)   # room for the y-label on the left and the colorbar on the right
fig.savefig(DATA_DIR / "CWT_cleaned.png", dpi=300)

plt.show()