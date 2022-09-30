# Imports
import os
import pretty_midi as pm
import pandas as pd
import numpy as np

"""
pitch_standard_deviation.py contains the functionality for computing the Pitch STD feature.

Inputs - a directory of MIDI melodies
Outputs - (returned) a DataFrame with 'ID' (MIDI file ID) and 'Pitch STD' columns
"""

"""
compute_pitch_std() takes a Pretty MIDI object and computes the pitch standard deviation of the melody it contains.
This is done by removing all temporal information, i.e., obtaining a list of MIDI pitches for each note event, and
taking the standard deviation of the list.

Inputs - a Pretty MIDI object
Outputs - pitch standard deviation
"""
def compute_pitch_std(midi_obj):
    # List of MIDI pitch numbers for each note vent
    notes = midi_obj.instruments[0].notes
    pitches = [x.pitch for x in notes]
    # Compute the standard deviation of this list
    std = np.std(pitches)
    return std

"""
pitch_standard_deviation() takes a directory of MIDIs, computes the pitch standard deviation of each MIDI, and
summarizes the results in a dataset.

Inputs - directory of MIDIs
Outputs - DataFrame with "ID" (file name) and "Pitch STD" columns
"""
def pitch_standard_deviation(directory):

    melody_ids = []
    stds = []

    # Read in each MIDI file as a pm object
    for filename in os.listdir(directory):
        if filename.endswith(".mid"):
            midi = pm.PrettyMIDI(os.path.join(directory, filename))
            std = compute_pitch_std(midi)

            # Store information
            melody_ids.append(filename[:-4])
            stds.append(std)

    # Create and return DataFrame
    df = pd.DataFrame(zip(melody_ids, stds), columns = ['ID', 'Pitch STD'])
    return df
