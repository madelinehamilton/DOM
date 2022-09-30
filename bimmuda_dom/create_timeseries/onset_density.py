# Imports
import os
import pretty_midi as pm
import pandas as pd

"""
onset_density.py contains the functionality for computing the Onset Density feature.

Inputs - a directory of MIDI melodies
Outputs - (returned) a DataFrame with 'ID' (MIDI file ID) and 'Onset Density' columns
"""

"""
compute_onset_density() computes the onset density of a monophonic MIDI object.
Onset density is the average number of note events per second.

Inputs - a Pretty MIDI object
Outputs - the onset density of the melody contained in the MIDI
"""
def compute_onset_density(midi_obj):
    length =  midi_obj.instruments[0].notes[-1].end
    num_notes = len(midi_obj.instruments[0].notes)
    return (num_notes/length)

"""
onset_density() computes the onset density of every MIDI in a given directory and creates a dataset summarizing the results.

Inputs - directory of Pretty MIDI objects
Outputs - a DataFrame with 'ID' (filename) and 'Onset Density' columns
"""
def onset_density(directory):

    melody_ids = []
    onset_densities = []

    # Read in each MIDI file as a pm object
    for filename in os.listdir(directory):
        if filename.endswith(".mid"):
            midi = pm.PrettyMIDI(os.path.join(directory, filename))
            # Compute OD
            od = compute_onset_density(midi)

            # Store information
            onset_densities.append(od)
            melody_ids.append(filename[:-4])

    # Create and return DataFrame
    df = pd.DataFrame(zip(melody_ids, onset_densities), columns = ['ID', 'Onset Density'])
    return df
