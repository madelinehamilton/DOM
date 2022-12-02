# Imports
from krumhansl_key_finder import key_finder
from pitch_standard_deviation import pitch_std
from melodic_interval_size import mis
from onset_density import onset_density
from nPVI import nPVI
import pretty_midi as pm
import pandas as pd
import os

"""
compute_dom_features_no_idyom.py computes all features that do not require IDyOM. This includes the
Tonality, Pitch STD, Melodic Interval Size, Onset Density and nPVI features.

Inputs - directory to the MIDI dataset
Outputs - a .csv with a DataFrame of these features is saved to output_data
"""

# SPECIFY BASE DIRECTORY
base_dir = "/Users/madelinehamilton/Documents/python_stuff/death_of_melody/"

# MIDI directory
input_dir = os.path.join(base_dir, "midis")
# Directory for output DataFrame
csv_name = os.path.join(base_dir, "output_data/dom_no_idyom_features.csv")

def compute_dom_features_no_idyom(directory):
    # Tonality
    tonality_df = key_finder(directory)
    # Pitch STD
    pitch_std_df = pitch_standard_deviation(directory)
    # Melodic Interval Size
    mis_df = mis(directory)
    # Onset Density
    od_df = onset_density(directory)
    # nPVI
    npvi_df = npvi(directory)

    df = pd.merge(tonality_df, pitch_std_df, on="ID")
    df = pd.merge(df, mis_df, on="ID")
    df = pd.merge(df, od_df, on="ID")
    df = pd.merge(df, npvi_df, on="ID")

    # Unwanted column
    df = df.drop(['Predicted Key'], axis=1)

    # Save
    df.to_csv(csv_name, index=False)

def compute_dom_features_no_idyom(directory):

    melody_ids = []
    years = []
    tonalities = []
    pitch_stds = []
    mis_vals = []
    onset_densities = []
    npvis = []

    # Read in each MIDI file as a pm object
    for filename in os.listdir(directory):
        if filename.endswith(".mid"):
            midi = pm.PrettyMIDI(os.path.join(directory, filename))
            # Melody ID is the filename minus ".mid"
            melody_id = filename[:-4]
            # The year is the first four characters of the filename
            year = int(filename[:4])

            # Compute features
            tonality = key_finder(midi)
            std = pitch_std(midi)
            melodic_interval_size = mis(midi)
            od = onset_density(midi)
            npvi_val = nPVI(midi)

            # Store information
            melody_ids.append(melody_id)
            years.append(year)
            tonalities.append(tonality)
            pitch_stds.append(std)
            mis_vals.append(melodic_interval_size)
            onset_densities.append(od)
            npvis.append(npvi_val)

    # Create DataFrame and save
    df_cols = ['ID', 'Year', 'Tonality', 'Pitch STD', 'MIS', 'Onset Density', 'nPVI']
    df = pd.DataFrame(list(zip(melody_ids, years, tonalities, pitch_stds, mis_vals, onset_densities, npvis)), columns = df_cols)
    df.to_csv(csv_name, index=False)

compute_dom_features_no_idyom(input_dir)
