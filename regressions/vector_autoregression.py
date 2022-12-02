# Imports
from regression_helper import normalize, nrmse_range
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
import os
import pandas as pd
import numpy as np
import scipy.stats

"""
vector_autoregression.py performs vector autoregression on the DOM smoothed time series. The time series are divided
into segments per the results of changepoint analysis, and VAR models are fit separately per segment.

Inputs: .csv containing the smoothed time series
Outputs:
        - .csv's with model coefficients for each era
        - .csv with 2022 forecasts generated by the Era 3 VAR

Note: many manual parts to this script. I drop features based on the printed results of the Granger's causality tests manually.
I also do manual insertion of the VAR model coefficients. I couldn't figure out how to extract the
coefficients automatically from the statsmodels object; I could only print them out and type them in a list.
"""

# SPECIFY BASE DIRECTORY
base_dir = "/Users/madelinehamilton/Documents/python_stuff/death_of_melody/"

# Directory of the smoothed time series
ts_filename = os.path.join(base_dir, "output_data/dom_time_series_smoothed.csv")

# Directories to save models (for Figure 2 in another script)
era_1_model_filename = os.path.join(base_dir, "output_data/era_1_var_coefficients.csv")
era_2_model_filename = os.path.join(base_dir, "output_data/era_2_var_coefficients.csv")
era_3_model_filename = os.path.join(base_dir, "output_data/era_3_var_coefficients.csv")

# Directory to save the Era 3 VAR forecasts (for Figure 3 in another script)
forecasts_filename = os.path.join(base_dir, "output_data/era_3_var_forecasts.csv")

# Pandas display settings
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 200)

"""Data preparation"""
# Read in the time series
ts_df = pd.read_csv(ts_filename)
ts_df = ts_df[['Tonality', 'MIC', 'Pitch STD', 'MIS', 'Onset Density', 'nPVI', 'RIC']].dropna().reset_index(drop=True)

# Normalize data
for col in ts_df.columns:
    ts_df[col] = normalize(list(ts_df[col]))

# Segment the data into its eras
era_1 = ts_df.iloc[0:13]
era_2 = ts_df.iloc[13:48]
era_3 = ts_df.iloc[48:]

eras = [era_1, era_2, era_3]

"""
First, we need to perform Granger's causality tests per era, so we can determine which features are suitable
for VAR.

grangers_causation_matrix() computes the Granger causality of all possible combinations of the time series. The rows
are the response variable, columns are predictors. The values in the table are the p-values. p-values lesser than the
significance level (0.05) implies the null hypothesis that the coefficients of the corresponding past values is zero,
that is, the hypothesis that X does not cause Y can be rejected.
"""

def grangers_causation_matrix(data, variables, test='ssr_chi2test'):

    maxlag=3
    test = 'ssr_chi2test'
    # Initialize matrix
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    # Iterate through all combinations of time series
    for c in df.columns:
        for r in df.index:
            # Grangers causality test from statsmodels
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            # Get best p value
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            min_p_value = np.min(p_values)
            # Store p value
            df.loc[r, c] = min_p_value
    # Label rows and columns of matrix
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

# Granger's causality matrix for each era
for i in range(len(eras)):
    print("Causality matrix for Era:", i+1)
    era = eras[i]
    print(grangers_causation_matrix(era, variables = era.columns))
    print()

"""
From there, you need to decide which features to keep for each segment. Here, I keep features in the analysis
if there are at least 2 p-values below .05 in its corresponding row and column in the causality matrix.
All variables will be kept for segments 1 and 3. For the middle segment, I need to remove the Tonality and
Onset Density features because they do not have enough significant p-values.
"""

era_1_columns = ts_df.columns
era_2_columns = ['MIC', 'Pitch STD', 'MIS', 'nPVI', 'RIC']
era_3_columns = ts_df.columns

era_1 = era_1[era_1_columns]
era_2 = era_2[era_2_columns]
era_3 = era_3[era_3_columns]

eras = [era_1, era_2, era_3]

"""
Next we need to find the optimal lag for each VAR. I don't care so much about parsimony here, so I don't impose
a maximum lag.
"""

models = []
# For each era, create a model and call select_order()
for i in range(len(eras)):
    print("Optimal lag for Era", i+1, "model:")
    era = eras[i]
    model = VAR(era)
    models.append(model)
    print(model.select_order())
    print()

"""
It looks like we can't meaningfully analyze Era 1 (1950 - 1964) with VAR. There's probably not enough observations.

For Era 2 the optimal lag is 4, and for Era 3 the optimal lag is 1.
"""

eras = [era_2, era_3]
models = models[1:]
lags = [4, 1]

"""
Fit the models with the optimal lags.
"""

models_fitted = []
for i in range(len(models)):
    model_fitted = models[i].fit(lags[i])
    models_fitted.append(model_fitted)
    #print(model_fitted.summary())

"""
For the Era 3 model, produce the 2022 forecasts.
"""
era_3_model = models_fitted[1]
era_3_lag = lags[1]
initial_values = era_3.values[1:]
forecast_vals = [era_3_model.forecast(initial_values, 3)[2]]
forecast_df = pd.DataFrame(forecast_vals, columns = eras[1].columns)

"""
Finally, save the forecasts and model coefficients.
"""
era_2_ys = ['MIC', 'MIC', 'Pitch STD', 'Pitch STD', 'Pitch STD', 'Pitch STD', 'MIS', 'MIS', 'MIS', 'MIS', 'MIS', 'MIS', 'MIS', 'MIS']
era_2_predictors = ['RIC', 'RIC', 'MIC', 'MIS', 'MIS', 'nPVI', 'RIC', 'MIC', 'MIS', 'nPVI', 'nPVI', 'Pitch STD', 'MIS', 'nPVI']
era_2_lags = [1, 4, 1, 2, 4, 4, 1, 2, 2, 2, 3, 4, 4, 4]
era_2_coefs = [0.7266, 0.7206, 1.042, -.7638, -.5232, -.8178, .8768, .8881, -.6766, -.8314, 1.046, .9774, -.6026, -.5494]

era_3_ys = ['Tonality', 'Tonality', 'Pitch STD', 'Pitch STD', 'Onset Density', 'Onset Density', 'Onset Density', 'Onset Density', 'Onset Density', 'nPVI', 'nPVI', 'RIC', 'RIC', 'RIC']
era_3_predictors = ['nPVI', 'RIC', 'nPVI', 'RIC', 'const', 'Tonality', 'Onset Density', 'nPVI', 'RIC', 'nPVI', 'RIC', 'Tonality', 'nPVI', 'RIC']
era_3_lags = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
era_3_coefs = [.5926, -.6967, 1.044,-1.290, .6881, -.4564,.4602,-1.582,1.777,2.067,-1.948,-.2737,1.697,-1.653]

era_2_df = pd.DataFrame(list(zip(era_2_ys, era_2_predictors, era_2_lags, era_2_coefs)), columns =['Dependent', 'Predictor', 'Lag', 'Coefficient'])
era_3_df = pd.DataFrame(list(zip(era_3_ys, era_3_predictors, era_3_lags, era_3_coefs)), columns =['Dependent', 'Predictor', 'Lag', 'Coefficient'])

era_2_df.to_csv(era_2_model_filename, index = False)
era_3_df.to_csv(era_3_model_filename, index = False)

forecast_df.to_csv(forecasts_filename, index = False)
