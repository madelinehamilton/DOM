# Imports
from time_series_smoothing import smoothing
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Matplotlib settings
plt.style.use('default')
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 9
sns.set_context('paper', font_scale=0.9)
plt.rcParams['figure.constrained_layout.use'] = True

"""
bimmuda_bpm.py produces Figure S2. in the paper, which visualizes the cyclic nature of BPM in the top pop songs between
1950 and 2021.

Inputs: .csv of the BiMMuDa metadata (created manually by the first author)
Outputs: .png of Figure S2.
"""

# SPECIFY BASE DIRECTORY
base_dir = "/Users/madelinehamilton/Documents/python_stuff/death_of_melody/"

# Directory of BiMMuDa per-song metadata
per_song_filename = os.path.join(base_dir, "other/bimmuda_per_song_metadata.csv")

# Directory of BPM image
bpm_image_filename = os.path.join(base_dir, "visualizations/bimmuda_bpm_cycles.png")

"""
Obtain the mean BPMs per year from the per-song metadata and smooth the time series
"""
# Import per-song data
df = pd.read_csv("bimmuda_per_song_metadata.csv")

# Get means of the features by year
means = df.groupby('Year').mean()
means = means.reset_index(level=0)

# Mean BPM of the songs by year
bpms = list(means['BPM 1'])

# Some smoothing would make the figure look better
smoothed_bpm = smoothing([bpms])[0]
smoothed_bpm = smoothed_bpm[2:-2]

"""
Create the figure and save it.
"""
years = list(range(1952, 2020))
plt.plot(years, smoothed_bpm)
plt.xlabel("Year")
plt.ylabel("BPM")
plt.ylim(80, 130)

# Save
plt.savefig(bpm_image_filename, bbox_inches='tight')
