# Imports
from regression_helper import normalize, nrmse_range
from sklearn.linear_model import LinearRegression
import math
import os
import pandas as pd
import statsmodels.api as sm

"""
residual_reg.py regresses the autoregression residuals of features computed previously against other features
(per era).

All of the regressions are univariate, with one exception (Tonality residual in Era 3). Through experimentation
(not documented), I figured out that, though the time series are uncorrelated overall, they are pretty correlated within eras,
leading to multicollinearity when I try to do multilinear regression most of the time. Because of this I found only one
significant multilinear regression.

Inputs:
       - a .csv of smoothed time series
       - .csv's of autoregression residuals for each era

Outputs:
       - a .csv containing Table 3 in the paper: results of significant regressions (estimates, p-values, etc.)
"""

# SPECIFY BASE DIRECTORY
base_dir = "/Users/madelinehamilton/Documents/python_stuff/death_of_melody/"

# Directory of smoothed time series
ts_filename = os.path.join(base_dir, "output_data/dom_time_series_smoothed.csv")

# Directories of autoregression residuals
resid_1_filename = os.path.join(base_dir, "output_data/era_1_residuals.csv")
resid_2_filename = os.path.join(base_dir, "output_data/era_2_residuals.csv")
resid_3_filename = os.path.join(base_dir, "output_data/era_3_residuals.csv")

# Directory for Table 3 .csv
table_3_filename = os.path.join(base_dir, "output_data/table_3.csv")

"""Prepare the data"""

# The original time series need to be normalized and segmented
original_ts_df = pd.read_csv(ts_filename)
original_ts_df = original_ts_df[['Tonality', 'MIC', 'Pitch STD', 'MIS', 'Onset Density', 'nPVI', 'RIC']].dropna().reset_index(drop=True)
for col in original_ts_df.columns:
    original_ts_df[col] = normalize(list(original_ts_df[col]))

seg1 = original_ts_df.iloc[0:13]
seg2 = original_ts_df.iloc[13:48]
seg3 = original_ts_df.iloc[48:]
original_eras = [seg1, seg2, seg3]

# Autoregression residuals for each era
era_1_residuals = pd.read_csv(resid_1_filename)
era_2_residuals = pd.read_csv(resid_2_filename)
era_3_residuals = pd.read_csv(resid_3_filename)

new_eras = [era_1_residuals, era_2_residuals, era_3_residuals]

"""
For each era, regress the residuals of one feature against the other 6 features. Keep regressions with R^2 values more
than 0.25 and p-values less than 0.05, and store the model info for Table 3.
"""

# This will store the regression results
reg_info = []
# Iterate through the eras
for i in range(len(new_eras)):
    # Get original time series and residuals for the era
    new_era_df = new_eras[i]
    old_df = original_eras[i]
    features = list(new_era_df.columns)

    # Iterate through each autoregression residual
    for dep in features:
        y_data = new_era_df[dep]

        # Remove NA values. Keep count of them so we know how to adjust the predictor data
        nan_count = sum([math.isnan(x) for x in list(y_data)])
        y_data = y_data.dropna()

        # Iterate through all possible predictors (all features except itself)
        for indep in features:
            if dep == indep:
                continue
            else:
                # Predictor
                x = old_df[indep][nan_count:]

                # Regression
                X2 = sm.add_constant(x)
                est = sm.OLS(list(y_data), X2)
                est2 = est.fit()
                # Get model info
                p_vals = est2.pvalues.to_dict()
                r_squared = est2.rsquared
                coefficient = est2.params[indep]

                # If p < 0.05
                if p_vals[indep] < 0.05:
                    # If R^2 >= 0.25
                    if r_squared >= 0.25:
                        # If the regression is significant, store its info
                        reg_info.append([i+1, dep, indep, round(coefficient, 3), round(p_vals[indep], 4), round(r_squared, 3)])

results_df = pd.DataFrame(reg_info, columns = ['Era', 'Dependent', 'Independent', 'Estimate', 'p-value', 'R^2'])

"""
We're mostly there. We just need to do the multilinear regression (Tonality residual on Pitch STD and Onset Density
in Era 3) and store those results.
"""

# Dependent
tonality_residual = new_eras[2]['Tonality']
nan_count = sum([math.isnan(x) for x in list(tonality_residual)])
tonality_residual = tonality_residual.dropna().reset_index(drop=True)

# Independents
pitch_std = list(original_eras[2]['Pitch STD'][nan_count:])
onset_density = list(original_eras[2]['Onset Density'][nan_count:])

# Regression
x_data = pd.DataFrame(zip(pitch_std, onset_density), columns = ['Pitch STD', 'Onset Density'])
X2 = sm.add_constant(x_data)
est = sm.OLS(tonality_residual, X2)
est2 = est.fit()

# Get model info
p_vals = est2.pvalues.to_dict()
r_squared = est2.rsquared
coefficients = est2.params.to_dict()

pitch_std_row = {'Era': 3, 'Dependent': 'Tonality', 'Independent': 'Pitch STD', 'Estimate': round(coefficients['Pitch STD'], 3), 'p-value': round(p_vals['Pitch STD'], 4), 'R^2': round(r_squared, 3)}
onset_density_row = {'Era': 3, 'Dependent': 'Tonality', 'Independent': 'Onset Density', 'Estimate': round(coefficients['Onset Density'], 3), 'p-value': round(p_vals['Onset Density'], 4), 'R^2': round(r_squared, 3)}

results_df = results_df.append(pitch_std_row, ignore_index=True)
results_df = results_df.append(onset_density_row, ignore_index=True)

# Save the Table 3 DataFrame
results_df.to_csv(table_3_filename, index = False)
