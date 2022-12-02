# Imports
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import LinearRegression
from regression_helper import normalize, nrmse_range
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import statsmodels.api as sm

# SPECIFY BASE DIRECTORY
base_dir = "/Users/madelinehamilton/Documents/python_stuff/death_of_melody/"

# Directory of smoothed time series
ts_filename = os.path.join(base_dir, "output_data/dom_time_series_smoothed.csv")

# Directories for residuals
resid_dir_1 = os.path.join(base_dir, "output_data/era_1_residuals.csv")
resid_dir_2 = os.path.join(base_dir, "output_data/era_2_residuals.csv")
resid_dir_3 = os.path.join(base_dir, "output_data/era_3_residuals.csv")

# Table 2 directory
table_2_dir = os.path.join(base_dir, "output_data/table_2.csv")

"""
autoregression_residuals.py fits autoregressive models for each feature, for each era, and saves the residuals for
future regressions. Each era has a maximum lag derived from the number of observations to encourage parsimony. Within
each era, the best (yielding the best nRMSE) lag for each feature is found. An autoregressive model for each feature is
fitted, and the residuals are saved.

Inputs: .csv with the smoothed time series
Outputs:
        - .csv storing the lags and nRMSE values for each AR model (for Table 2 of the paper)
        - .csv storing the residuals from all the autoregression fits, for regression later
"""

# Read in the data
ts_df = pd.read_csv(ts_filename)
ts_df = ts_df[['Tonality', 'MIC', 'Pitch STD', 'MIS', 'Onset Density', 'nPVI', 'RIC']].dropna().reset_index(drop=True)

# Normalize
for col in ts_df.columns:
    ts_df[col] = normalize(list(ts_df[col]))

"""
With changepoint detection, we found three eras: 1950 - 1964, 1965 - 1999, and 2000 - 2021. We need to divide
the time series into these eras and perform autoregression per era.
"""
seg1 = ts_df.iloc[0:13]
seg2 = ts_df.iloc[13:48]
seg3 = ts_df.iloc[48:]
eras = [seg1, seg2, seg3]

"""
Fit an autoregressive model for each feature. We need to determine the optimal lag per feature per era,
with a maximum lag dependent on the length of the era in years.

Then, fit the autoregressive model with optimal lag, compute the nRMSE and residuals and store these in
new DataFrames.

We also need to be storing the best lags and nRMSEs for each feature and era for Table 2 of the paper.
"""

seg1_maxlag = 3
seg2_maxlag = 7
seg3_maxlag = 4

maximum_lags = [seg1_maxlag, seg2_maxlag, seg3_maxlag]

new_eras = []

# Initial storing of results for Table 2
# Idk if this is optimal but computer science is hard
ar_results = {'Tonality': [], 'MIC': [], 'Pitch STD': [], 'MIS': [], 'Onset Density': [], 'nPVI': [], 'RIC': []}

# Iterate through the era time series
for i in range(len(eras)):
    df = eras[i]
    max_lag = maximum_lags[i]
    # Empty DataFrame for the residuals
    new_df = pd.DataFrame(columns=df.columns)
    # Iterate through all the features
    for col in ts_df.columns:
        # Get the feature
        series = df[col]
        series = series.reset_index(drop=True)
        # List to store nRMSEs of the different lags
        nrmses = []
        # Iterate through all possible lags
        for l in range(1, max_lag + 1):
            # Fit model with lag
            model = AutoReg(series, lags = l, old_names = True)
            # Get the fitted values
            results = model.fit()
            fitted = list(results.fittedvalues)
            # Add the first l elements of the original time series to the beginning of the fitted values
            # This is for nRMSE purposes: an AR model of lag l cannot produce fitted values for the first
            # l elements, so the first l elements will just be the original values
            first_l = list(series)[:(l+1)]
            fitted = first_l + fitted
            # Compute nRMSE between fitted and original values
            nrmse = nrmse_range(list(series), fitted)
            nrmses.append(nrmse)

        # Choose the lag with the lowest nRMSE
        best_nrmse = min(nrmses)
        best_lag = nrmses.index(min(nrmses)) + 1

        # Fit a model with this lag
        model = AutoReg(series, lags = best_lag, old_names = True)
        results = model.fit()
        # Get the residuals
        residuals = list(results.resid)
        # Put N/A for the first best_lag values (as explained in the above comment)
        residuals = [np.nan]*best_lag + residuals
        # The "new" version of the feature is now the residuals. Store this
        new_df[col] = residuals

        # Store model info for Table 2
        feature_info = ar_results[col]
        feature_info.append(round(best_nrmse, 2))
        feature_info.append(best_lag)
        ar_results[col] = feature_info

    # Store the residuals for each era
    new_eras.append(new_df)

"""
Finally, write out the residuals for future regression, and the AR model info for Table 2.
"""

era_1_resid = new_eras[0]
era_2_resid = new_eras[1]
era_3_resid = new_eras[2]

era_1_resid.to_csv(resid_dir_1, index=False)
era_2_resid.to_csv(resid_dir_2, index=False)
era_3_resid.to_csv(resid_dir_3, index=False)

# Again this probably isn't optimal but whatever
# Table 2
table_2_rows = []
for tup in ar_results.items():
    feat = tup[0]
    table_2_rows.append([feat] + tup[1])

table_2_df = pd.DataFrame(table_2_rows, columns=['Feature', 'Era 1 nRMSE', 'Era 1 Lag', 'Era 2 nRMSE', 'Era 2 Lag', 'Era 3 nRMSE', 'Era 3 Lag'])
table_2_df.to_csv(table_2_dir, index=False)
