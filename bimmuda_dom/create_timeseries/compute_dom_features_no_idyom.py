# Imports
from krumhansl_key_finder import key_finder
from pitch_standard_deviation import pitch_standard_deviation
from melodic_interval_size import mis
from onset_density import onset_density
from nPVI import npvi
import pandas as pd
import os

"""
compute_dom_features_no_idyom.py computes all Death of Melody features that do not require IDyOM. This includes the
Tonality, Pitch STD, Melodic Interval Size, Onset Density and nPVI features.

Inputs - directory to the (preprocessed) MIDI dataset (make sure you have run preprocess_midis.py first)
Outputs - a .csv with a DataFrame of these features is saved.
"""

# SPECIFY DIRECTORY
input_dir = "/Users/madelinehamilton/Documents/bimmuda_new_processed"
# SPECIFY DIRECTORY FOR OUTPUT DATASET
csv_name = "/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/output_data/dom_no_idyom_features.csv"

"""
melody_ids() creates a base dataset with melody IDs and the year of the song they come from. In this case, the year can be found
in the first four characters of the filename.

Inputs - directory to preprocessed MIDIs
Outputs - DataFrame of with two columns: "ID", the name of the file minus the .mid suffix, and "Year".
"""
def melody_ids(directory):
    melody_ids = []
    years = []
    # Iterate through directory
    for filename in os.listdir(directory):
        if filename.endswith(".mid"):
            # Get the melody ID and year from the filename
            melody_id = filename[:-4] # Everything but .mid
            year = int(filename[:4]) # First four characters

            melody_ids.append(melody_id)
            years.append(year)

    df = pd.DataFrame(zip(melody_ids, years), columns = ['ID', 'Year'])
    return df

"""
compute_dom_features_no_idyom() computes the features in the Death of Melody paper that do not involve IDyOM.

Inputs - directory of the preprocessed MIDIs.
Output - Nothing returned. A .csv containing a dataset is written out. Each row will have 5 columns representing
         5 of the 7 DOM features.

Actions:
- Creates a dataset of melody IDs (filenames) with 5 columns
- Computes each feature (the bulk of this is done with functions written in other files)
- Saves the dataset
"""

def compute_dom_features_no_idyom(directory):
    # Instantiate dataset, which will be added on to
    df = melody_ids(directory)

    # Compute each non-IDyOM feature
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

    # Merge the DataFrames together
    df = pd.merge(df, tonality_df, on="ID")
    df = pd.merge(df, pitch_std_df, on="ID")
    df = pd.merge(df, mis_df, on="ID")
    df = pd.merge(df, od_df, on="ID")
    df = pd.merge(df, npvi_df, on="ID")

    # Save the DataFrame
    df.to_csv(csv_name, index=False)

compute_dom_features_no_idyom(input_dir)
