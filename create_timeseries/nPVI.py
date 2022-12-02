# Imports
import pandas as pd
import os
import pretty_midi as pm

"""
nPVI.py contains the functionality for computing the nPVI feature.

Inputs - a Pretty MIDI object
Outputs - (returned) a DataFrame with 'ID' (MIDI file ID) and 'nPVI' columns
"""

"""
relative_durations() is a util for computing nPVI. It takes a durational vector (list of the durations of each note event)
and normalizes it so the smallest duration becomes 1.0, and the remaining durations are expressed as multiples of the
smallest duration.

Inputs - a Pretty MIDI object containing a monophonic melody.
Outputs - the relative duration vector of the melody.
"""
def relative_durations(midi_obj):
    # Get absolute durations
    notes = midi_obj.instruments[0].notes
    dur_vector = [(x.end - x.start) for x in notes]
    # Find smallest duration
    smallest_dur = min(dur_vector)
    # Divide every element by the smallest duration
    relative_durs = [int(round(x/smallest_dur)) for x in dur_vector]
    return relative_durs

"""
nPVI() computes the nPVI (normalized Pairwise Variability Index) (Patel et al 2003) of a relative durational vector.
The nPVI measures variability between relative durations. From temporal interval to temporal interval, how much variation
is there?

Inputs - a Pretty Midi object
Outputs - the nPVI associated with the relative durational vector
"""
def nPVI(midi):
    # Compute the relative duration vector
    rel_dur_vec = relative_durations(midi)
    # Number of durations (for example, if only quarter notes are used then m = 1. If half notes, quarter notes and
    # eighth notes are used, then m = 3.)
    m = len(rel_dur_vec)
    # Iterate through the intervals
    summ = 0
    for k in range(m-1):
        d_k = rel_dur_vec[k]
        d_k_plus_1 = rel_dur_vec[k+1]
        if d_k == d_k_plus_1:
            continue
        else:
            num = d_k - d_k_plus_1
            denom = (d_k + d_k_plus_1)/2
            summ += abs(num/denom)/(m-1)
    npvi = (100)*summ
    return npvi
