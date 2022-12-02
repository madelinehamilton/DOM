# Imports
from time_series_smoothing import smoothing
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
produce_full_dom_dataset.py takes a .csv with a dataset of the 5 non-IDyOM features and merges it with two .dat files, each of
which contain one IDyOM feature. It also averages the features by year and smooths the resulting time series to produce the
data needed for the DOM changepoint detection and regressions.

You need to specify the base directory.

Inputs:
       - .csv produced by compute_dom_features_no_idyom.py, which contains the DOM features not computed by IDyOM
       - Two .dat files, each with a DOM feature computed by IDyOM (the RIC and MIC features)
Outputs:
       - a .csv of a DataFrame containing all 7 DOM features computed per MIDI melody
       - a .csv of a DataFrame containing the 7 unsmoothed DOM time series
       - a .csv of a DataFrame containing the 7 smoothed DOM time series
"""

# SPECIFY BASE DIRECTORY
base_dir = "/Users/madelinehamilton/Documents/python_stuff/death_of_melody/"

# SPECIFY NAMES OF RIC AND MIC IDYOM FILES
mic_filename = os.path.join(base_dir, "output_data/mic_from_idyom.dat")
ric_filename = os.path.join(base_dir, "output_data/ric_from_idyom.dat")

# Non-IDyOM feature DataFrame
non_idyom_filename = os.path.join(base_dir, "output_data/dom_no_idyom_features.csv")

# Output Directories
full_dom_df_name = os.path.join(base_dir, "output_data/dom_features.csv")
ts_unsmoothed_df_name = os.path.join(base_dir, "output_data/dom_time_series_unsmoothed.csv")
ts_smoothed_df_name = os.path.join(base_dir, "output_data/dom_time_series_smoothed.csv")

"""
visualize_time_series() takes a DataFrame of time series and plots each column.

Inputs - a DataFrame with 'Year' as the first column and the time series in remaining columns
Outputs - nothing returned, Matplotlib graph shown
"""
def visualize_time_series(time_series_df):
    years = list(time_series_df['Year'])
    # Iterate through time series
    for col in list(time_series_df.columns)[1:]:
        data = list(time_series_df[col])
        plt.plot(years, data)
        plt.title(col)
        plt.show()

"""
full_dom_dataset() produces the full set of time series needed for the DOM analysis (see description at the top)
"""
def full_dom_dataset(visualize=True):
    # Read in the datasets
    non_idyom_features_df = pd.read_csv(non_idyom_filename)
    mic_df = pd.read_csv(mic_filename, sep='\s+', engine="python")
    ric_df = pd.read_csv(ric_filename, sep='\s+', engine="python")

    # Create 'ID' columns for the IDyOM features so we can merge properly
    mic_df['ID'] = mic_df['melody.name'].str[1:-1]
    ric_df['ID'] = ric_df['melody.name'].str[1:-1]

    # Get the columns we need from the IDyOM feature datasets
    mic_df = mic_df[['ID', 'mean.information.content']]
    ric_df = ric_df[['ID', 'mean.information.content']]
    mic_df.columns = ['ID', 'MIC']
    ric_df.columns = ['ID', 'RIC']

    # Merge these with the non_idyom_features DataFrame
    dom_df = pd.merge(non_idyom_features_df, mic_df, on="ID", how ='outer')
    dom_df = pd.merge(dom_df, ric_df, on="ID", how ='outer')

    # Save the full dataset before computing means by year
    dom_df.to_csv(full_dom_df_name, index=False)

    # Time series (non-smoothed)
    time_series_df = dom_df.groupby('Year').mean()
    time_series_df = time_series_df.reset_index(level=0)

    # Save the raw time series
    time_series_df.to_csv(ts_unsmoothed_df_name, index=False)

    # Prepare the data for smoothing by turning DataFrame into list of lists
    series_lists = []
    for column in list(time_series_df.columns[1:]):
        series_lists.append(list(time_series_df[column]))

    # Smooth the time series
    smoothed_lists = smoothing(series_lists)
    smoothed_lists = [list(time_series_df['Year'])] + smoothed_lists
    smoothed_lists = [list(i) for i in zip(*smoothed_lists)]
    smoothed_time_series_df = pd.DataFrame(smoothed_lists, columns = time_series_df.columns)

    # Save
    smoothed_time_series_df.to_csv(ts_smoothed_df_name, index=False)

    # Finally, visualize each of the smoothed time series
    if visualize:
        visualize_time_series(smoothed_time_series_df)

full_dom_dataset()
