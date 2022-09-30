# Imports
from mido import MidiFile
import os
import pretty_midi as pm
import mido
import shutil

# SPECIFY DIRECTORY OF MIDI DATASET
input_directory = "/Users/madelinehamilton/Documents/bimmuda_new"
# SPECIFY DIRECTORY FOR NEW DATASET (make sure an empty folder with this name is created)
output_directory = "/Users/madelinehamilton/Documents/bimmuda_new_processed"

"""
preprocess_midis performs preprocessing of the MIDIs in the BiMMuDa, to prepare it for IDyOM / DOM paper.

Input - directory of the dataset (enter below this description)
Output - the preprocessed dataset, in the new directory specified

Actions:

- 'Flattens' the file structure (no folders)
- Filters out non-MIDI files and _full files (excludes lyrics, .mscz files)
- Checks file types, ensuring each MIDI is type 0 (only one track)
- Checks for note overlap, ensuring only one note is playing at a time
- TODO: Normalizes velocities
"""

"""
filter_flat() iterates through the directory, moving them to the new directory, excluding _full.mid, _misc.mid,
.mscz files, and .txt files. It also prints a file count.

Inputs - input directory, output directory
Output - nothing returned, file count printed
"""
def filter_flat(input_dir, output_dir):
    count = 0
    # Walk through the directory
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            # Exclude .txt and .mscz files
            if file.endswith(".mid"):
                # Exclude _full and _misc MIDIs
                if "full" in file:
                    continue
                elif "misc" in file:
                    continue
                # Copy MIDIs to new directory
                else:
                    count += 1
                    old_path = os.path.join(subdir, file)
                    new_path = os.path.join(output_dir, file)
                    shutil.copy(old_path, new_path)
    print("File count:", count, "MIDI files.")

"""
check_file_types() checks that all files in the new directory are type 0. It prints the names of all files that are
not type 0. These should be edited manually before proceeding.

Inputs - directory of the new, preprocessed MIDI directory
Outputs - returns 0 if there are no issues, 1 if there is at least one non-Type 0 file. A list of non-Type 0 files is printed.
"""
def check_file_types(directory):
    issues_flag = 0
    # Iterate through directory
    for filename in os.listdir(directory):
        if filename.endswith(".mid"):
            midi = pm.PrettyMIDI(os.path.join(directory, filename))
            # Type 1 MIDIs have more than one track
            if len(midi.instruments) > 1:
                issues_flag = 1
                print("MIDI", filename, "is not Type 0.")
    return issues_flag

"""
check_overlap() checks that, for all the MIDI files in the new preprocessed directory, there is no temporal overlap between
notes. That is, either 0 or 1 notes are playing at all times.

Inputs - directory of the new, preprocessed MIDI directory
Outputs - returns 0 if there are no issues, 1 if there is at least one file with overlapping notes.
          A list of files with overlapping notes is printed.
"""
def check_overlap(directory):
    issues_flag = 0
    # Iterate through directory
    for filename in os.listdir(directory):
        if filename.endswith(".mid"):
            # Read in MIDI, get note list
            midi = pm.PrettyMIDI(os.path.join(directory, filename))
            notes = midi.instruments[0].notes
            i = 0
            # Start with the first onset time and iterate until the second to last onset time
            while i < len(notes) - 1:
                # Get the current and next onset
                current_note = midi.instruments[0].notes[i]
                next_note = midi.instruments[0].notes[i + 1]
                # Check for simultaneous onsets within a threshold of .0001
                if (next_note.start - current_note.start) < .0001:
                    print("Two notes at the same time in file:", filename)
                    issues_flag = 1
                i += 1

            # Now check for overlapping notes.
            # That is, for each note, check that the offset occurs BEFORE the onset of the following note.
            for i in range(len(notes)):
                # Skip the last note
                if i == len(notes) - 1:
                    continue
                else:
                    current_note = midi.instruments[0].notes[i]
                    next_note = midi.instruments[0].notes[i + 1]
                    # If current offset > next onset
                    if current_note.end >= next_note.start:
                        print("Overlapping notes in", filename)
                        issues_flag = 1
    return issues_flag

# MAIN

# Flatten and filter
print("Copying files to new directory...")
filter_flat(input_directory, output_directory)

# Check file types
print("Ensuring all files are Type 0...")
file_types_issue_flag = check_file_types(output_directory)

# Check for overlapping notes
print("Checking for simultaneous/overlapping notes...")
overlapping_notes_issue_flag = check_overlap(output_directory)

# If there are issues, print a message and terminate.
# Otherwise, normalize velocities
if file_types_issue_flag == 1 or overlapping_notes_issue_flag == 1:
    print("Please fix issues; preprocessing can then continue")
