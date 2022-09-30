# Imports
from time_series_smoothing import smoothing
import pandas as pd
import matplotlib.pyplot as plt

"""
produce_full_dom_dataset.py takes a .csv with a dataset of the 5 non-IDyOM features and merges it with two .dat files, each of
which contain one IDyOM feature. It also averages the features by year and smooths the resulting time series to produce the
data needed for the DOM changepoint detection and VAR.

You need to specify the filenames taken as input (non-IDyOM feature dataset, MIC data from IDyOM, and RIC data from IDyOM) and
the desired output filenames (dataset with DOM feature values by melody, unsmoothed time series data, and smoothed time series
data).

Inputs:
       - .csv produced by compute_dom_features_no_idyom.py, which contains the DOM features not computed by IDyOM
       - Two .dat files, each with a DOM feature computed by IDyOM (the RIC and MIC features)
Outputs:
       - a .csv of a DataFrame containing all 7 DOM features computed per MIDI melody
       - a .csv of a DataFrame containing the 7 unsmoothed DOM time series
       - a .csv of a DataFrame containing the 7 smoothed DOM time series
Actions:
       - Merges the non-IDyOM and IDyOM data into one DataFrame
       - Averages all features by year to create the 7 DOM time series
       - Smoothes the 7 time series
"""

# SPECIFY INPUT DIRECTORIES
NON_IDYOM_FILENAME = '/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/output_data/dom_no_idyom_features.csv'
MIC_FILENAME = '/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/output_data/mic_from_idyom.dat'
RIC_FILENAME = '/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/output_data/ric_from_idyom.dat'

# SPECIFY DESIRED OUTPUT DIRECTORIES
FULL_DOM_DF = '/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/output_data/dom_features.csv'
TIME_SERIES_UNSMOOTHED_DF = '/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/output_data/dom_time_series_unsmoothed.csv'
TIME_SERIES_SMOOTHED_DF = '/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/output_data/dom_time_series_smoothed.csv'

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
    non_idyom_features_df = pd.read_csv(NON_IDYOM_FILENAME)
    mic_df = pd.read_csv(MIC_FILENAME, sep='\s+', engine="python")
    ric_df = pd.read_csv(RIC_FILENAME, sep='\s+', engine="python")

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

    # We don't need the 'Predicted Key' column, take it out
    dom_df = dom_df[['ID', 'Year', 'Tonality', 'MIC', 'Pitch STD', 'MIS', 'Onset Density', 'nPVI', 'RIC']]

    # Save the full dataset before computing means by year
    dom_df.to_csv(FULL_DOM_DF, index=False)

    # Time series (non-smoothed)
    time_series_df = dom_df.groupby('Year').mean()
    time_series_df = time_series_df.reset_index(level=0)

    # Save the raw time series
    time_series_df.to_csv(TIME_SERIES_UNSMOOTHED_DF, index=False)

    # Prepare the data for smoothing; turn DataFrame into list of lists
    series_lists = []
    for column in list(time_series_df.columns[1:]):
        series_lists.append(list(time_series_df[column]))

    # Smooth the time series
    smoothed_lists = smoothing(series_lists)
    smoothed_lists = [list(time_series_df['Year'])] + smoothed_lists
    smoothed_lists = [list(i) for i in zip(*smoothed_lists)]
    smoothed_time_series_df = pd.DataFrame(smoothed_lists, columns = time_series_df.columns)

    # Save
    smoothed_time_series_df.to_csv(TIME_SERIES_SMOOTHED_DF, index=False)

    # Finally, visualize each of the smoothed time series
    if visualize:
        visualize_time_series(smoothed_time_series_df)

full_dom_dataset()
