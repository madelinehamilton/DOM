# Imports
import pandas as pd
import os
from collections import Counter

"""
changepoint_tally.py takes a .csv containing a table of changepoints as generated by changepoint_analysis.R,
tallies the changepoints per feature and applies thresholds to determine the "true" changepoints per feature.
This is necessary for determining the Tiers in the DOM analysis.

Inputs - .csv with the list of changepoints per parameter setting.
Outputs:
    - (Saved) A .csv with the changepoint tallies for visualization later
    - (Printed) A list of the "true" changepoints per feature (plus "true" multivariate changepoints)

"""

# SPECIFY BASE DIRECTORY
base_dir = "/Users/madelinehamilton/Documents/python_stuff/death_of_melody/"

# Name of .csv with changepoints
changepoints_table_name = os.path.join(base_dir, "output_data/changepoints.csv")
# Name of changepoint tally .csv
tally_csv_name = os.path.join(base_dir, "output_data/changepoint_tallies.csv")

"""
apply_tally_thresholds() discards changepoints that do not meet a tally threshold, given lists of changepoints and their
tallies.

Inputs - a DataFrame of changepoint tallies as produced by tally_changepoints() (see below)
Outputs - a DataFrame of changepoint tallies that contains only the "true" changepoints (those that exceed the
          threshold tally)
"""
def apply_tally_thresholds(df, threshold=30):
    # For each feature, loop through the "Positions" and "Tallies" lists simultaneously.
    # Mark which positions are above the threshold
    threshold_df = pd.DataFrame(columns=['Feature','Changepoints'])
    features = ['Tonality', 'MIC', 'Pitch.STD', 'MIS', 'Onset.Density', 'nPVI', 'RIC', 'Multivariate']
    for f in features:
        # Get row associated with feature
        row = df.loc[df['Feature'] == f]
        # Positions of the changepoints
        positions = list(row['Positions'])[0]
        # Tallies of the changepoints
        tallies = list(list(row['Tallies'])[0])
        # Remove positions that do not meet the threshold
        new_positions = [positions[i] for i in range(len(positions)) if tallies[i] >= 30]
        # Store the "true" changepoints
        dict_for_df = {'Feature': f, 'Changepoints': new_positions}
        threshold_df = threshold_df.append(dict_for_df, ignore_index=True)
    print("Changepoints for each feature (plus multivariate changepoints):")
    print(threshold_df)
    return threshold_df

"""
tally_changepoints() takes a DataFrame of changepoints and creates a dataset of value counts for each changepoint
per feature.

Inputs - directory to .csv containing a DataFrame of changepoints, as produced by changepoint_analysis.R
Outputs - a DataFrame of changepoints tallies with 'Feature' (feature name), 'Positions' (list of changepoint
positions for the feature) and 'Tallies' (list of tallies for each changepoint) columns.
"""
def tally_changepoints(file):

    # Import the changepoint data
    column_names = ["feature", "alpha", "k", "min_size", "pos"]
    changept_df = pd.read_csv(changepoints_table_name, names=column_names, header=None)
    changept_df = changept_df.iloc[1:]
    changept_df['k'] = changept_df['k'].fillna("NULL")

    # For each feature, tally the changepoints
    tallies = changept_df.groupby(['feature'])['pos'].value_counts()
    tally_df = pd.DataFrame(columns=['Feature','Positions','Tallies'])

    # How to I turn this into a useful DataFrame? I guess I have to go the long way
    features = ['Tonality', 'MIC', 'Pitch.STD', 'MIS', 'Onset.Density', 'nPVI', 'RIC', 'Multivariate']
    for f in features:
        counts = tallies[f]
        positions = [int(x) for x in list(counts.index)]
        tals = counts.values
        # Convert from time series position to years
        positions = [1951 + x for x in positions]
        dict_for_df = {'Feature': f, 'Positions': positions, 'Tallies': tals}
        tally_df = tally_df.append(dict_for_df, ignore_index=True)

    # Write out the tallies
    tally_df.to_csv(tally_csv_name, index=False)
    return tally_df

tally_info = tally_changepoints(changepoints_table_name)
true_changepoints = apply_tally_thresholds(tally_info)