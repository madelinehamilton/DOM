import pandas as pd
import os
import pretty_midi as pm

"""
melodic_interval_size.py contains the functionality for computing the MIS feature.

Inputs - a directory of MIDI melodies
Outputs - (returned) a DataFrame with 'ID' (MIDI file ID) and 'MIS' columns
"""

"""
melodic_interval_size() takes a Pretty MIDI object and computes the melodic interval size of the melody contained in it.
Melodic interval size is the average distance, in MIDI pitch note numbers, between consecutive pitch intervals.

Inputs - Pretty MIDI object
Outputs - melodic interval size of the melody
"""
def melodic_interval_size(midi_obj):

    # Note event list
    note_list = midi_obj.instruments[0].notes

    intervals = []

    # Iterate through each note event (except for the last one)
    for i in range(len(note_list)):
        if i == len(note_list)-1:
            continue
        else:
            # For each note event, get the pitch of the current note and the next note
            current_note = note_list[i]
            next_note = note_list[i+1]

            # Compute and store the interval between the two notes
            interval = abs(current_note.pitch - next_note.pitch)
            intervals.append(interval)

    # Average interval
    mis = sum(intervals)/len(intervals)
    return mis

"""
compute_mis() computes the melodic interval size for every MIDI in a given directory and creates and dataset
summarizing the results.

Inputs - MIDI directory
Outputs - DataFrame with 'ID' (filename) and 'MIS' (melodic interval size) columns
"""
def mis(directory):

    melody_ids = []
    melodic_interval_sizes = []

    # Read in each MIDI file as a pm object
    for filename in os.listdir(directory):
        if filename.endswith(".mid"):
            midi = pm.PrettyMIDI(os.path.join(directory, filename))
            mis = melodic_interval_size(midi)
            melodic_interval_sizes.append(mis)
            melody_ids.append(filename[:-4])

    # Create and return DataFrame
    df = pd.DataFrame(zip(melody_ids, melodic_interval_sizes), columns = ['ID', 'MIS'])
    return df
