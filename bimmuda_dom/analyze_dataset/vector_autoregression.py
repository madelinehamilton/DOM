# Imports
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, aic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

"""
vector_autoregression.py performs vector autoregression on the DOM smoothed time series segment-wise. That is, the time
series are divided into segments per the results of changepoint analysis, and VAR models are fit separately per segment.

Inputs - directory of the .csv containing the smoothed time series, as computed in produce_full_dom_dataset.py.
Outputs - (printed) results of Granger's causality tests and VAR model order selection, as well as model coefficients
          and nRMSE values for each segment's model.

Note: this analysis was developed in Jupyter Notebooks, hence the slightly odd formatting. As you see the results of
each part of the analysis, you will have to manually type out columns, coefficients, etc. that you wish to keep for
the next part.
"""

# SPECIFY INPUT DIRECTORY OF SMOOTHED TIME SERIES
ts_filename = '/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/output_data/dom_time_series_smoothed.csv'

# SPECIFY DESIRED OUTPUT DIRECTORY FOR VAR MODEL COEFFICIENTS
#seg1_filename = '/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/output_data/segment1_var_coefs.csv'
seg2_filename = '/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/output_data/segment2_var_coefs.csv'
seg3_filename = '/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/output_data/segment3_var_coefs.csv'

# Z-score
def normalize(lst):
    rnge = max(lst) - min(lst)
    lst2 = [(x - min(lst))/rnge for x in lst]
    return lst2

"""
grangers_causation_matrix() computes the Granger causality of all possible combinations of the time series. The rows
are the response variable, columns are predictors. The values in the table are the P-Values. P-Values lesser than the
significance level (0.05), implies the null hypothesis that the coefficients of the corresponding past values is zero,
that is, the X does not cause Y can be rejected.

Inputs - DataFrame of the time series to be tested and their labels/names
Outputs - DataFrame containing the causality matrix
"""

def grangers_causation_matrix(data, variables, test='ssr_chi2test'):

    maxlag=3
    test = 'ssr_chi2test'

    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    # Iterate through all combinations of time series
    for c in df.columns:
        for r in df.index:
            # Grangers causality test from statsmodels
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            min_p_value = np.min(p_values)
            # Store p value
            df.loc[r, c] = min_p_value
    # Label rows and columns of matrix
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

"""
nmrse_range() computes a metric for evaluating the performance of a model. Between the original values and the model's
predicted values, the RMSE is computed value-by-value. These are averaged, and then square root is taken. This value
is then normalized by the range of original values, to compensate for differences in scale between features.

Inputs - two lists, one containing original feature values and one containing the model prediction values.
Outputs - the nMRSE value between the two lists
"""
def nrmse_range(original_vals, model_vals):

    lst1 = original_vals
    lst2 = model_vals
    errors = [0]*len(lst1)

    # Iterate through the two lists simultaneously
    for i in range(len(lst1)):
        x1 = lst1[i]
        x2 = lst2[i]
        # L2 error between the two values
        errors[i] = (x1-x2)**2

    # Take the mean and square root
    avg_error = (sum(errors)/len(errors))
    rmse = np.sqrt(avg_error)
    # Normalize by the range
    minimum = min(original_vals)
    maximum = max(original_vals)
    nmrse = rmse/(maximum-minimum)

    return nmrse

"""
compute_model_values() takes selected VAR coefficients, computes the predicted VAR values of features, and computes
the error between the model values and the actual feature values (nMRSE).
"""
def compute_model_values(ts_data, coef_data, segment_no, plot=True):
    # Get all dependent names
    dependent_names = list(coef_data['Dependent'].unique())
    # Lists to hold computed values
    new_vals = []
    # Length of predictor data
    data_length = len(list(ts_data['RIC']))
    # Loop through each dependent
    for i in range(len(dependent_names)):
        const = 0
        dependent_name = dependent_names[i]
        # Get the original time series
        ts = list(ts_data[dependent_name])
        # Get the coefficient data
        coefs = coef_data[coef_data['Dependent'] == dependent_name]
        # Lag each predictor appropriately, multiply by coefficient
        predictor_names = list(coefs['Predictor'].unique())

        reg_values = []
        for i2 in range(len(predictor_names)):
            predictor = predictor_names[i2]
            coef_info = coefs[coefs['Predictor'] == predictor]
            lag = list(coef_info['Lag'])[0]
            coefficient = list(coef_info['Coefficient'])[0]
            # For lag L, take off the last L elements, and add L nans to the beginning
            x_data = 0
            if lag == 0:
                x_data = [1]*data_length
            else:
                x_data = list(ts_data[predictor])
            lagged_x = [np.nan]*lag
            lagged_x = lagged_x + x_data[:len(x_data)-lag]
            lagged_x = [coefficient*x for x in lagged_x]
            reg_values.append(lagged_x)


        # reg_values is a list of lists. Each list corresponds to one relevant predictor
        # Each component list has the lagged values of the relevant time series multiplied by their appropriate coefficient
        # Add values together (add first value of each list, second value of each list, etc.)
        ys = list(map(sum, zip(*reg_values)))
        new_vals.append(ys)

        # Remove nan
        nans = [0]*len(ys)
        for i in range(len(ys)):
            nans[i] = np.isnan(ys[i])

        number_of_nans = sum(nans)

        ys_no_nan = ys[number_of_nans:]
        ts_no_nan = ts[number_of_nans:]

        # Plot
        if plot:
            plt.plot(ts_no_nan, label='Original Values')
            plt.plot(ys_no_nan, label='Model Values')
            plt.title(dependent_name + " Segment " + str(segment_no))
            plt.legend()
            plt.show()

        # Error between the original and model values
        error = nrmse_range(ts_no_nan, ys_no_nan)
        print("nRMSE for VAR predicting", dependent_name, "in segment", segment_no, ":", error)
    return (list(zip(dependent_names,new_vals)))

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 200)
ts_df = pd.read_csv(ts_filename)
ts_df = ts_df[['Tonality', 'MIC', 'Pitch STD', 'MIS', 'Onset Density', 'nPVI', 'RIC']].dropna().reset_index(drop=True)

# Normalize data
for col in ts_df.columns:
    ts_df[col] = normalize(list(ts_df[col]))

seg1 = ts_df.iloc[0:13]
seg2 = ts_df.iloc[13:48]
seg3 = ts_df.iloc[48:]

# Analyze each segment separately
segments = [seg1, seg2, seg3]
# Granger's causality matrix for each segment
for s in segments:
    granger_mat = grangers_causation_matrix(s, variables = s.columns)

"""
From there, you need to decide which features to keep for each segment. Here, I keep features in the analysis
if there are at least 2 p-values below .05 in its corresponding row and column in the causality matrix.
All variables will be kept for segments 1 and 3. For the middle segment, I need to remove the Tonality and
Onset Density features because they do not have enough significant p-values.
"""

seg1_columns = ts_df.columns
seg2_columns = ['MIC', 'Pitch STD', 'MIS', 'nPVI', 'RIC']
seg3_columns = ts_df.columns

seg1 = seg1[seg1_columns]
seg2 = seg2[seg2_columns]
seg3 = seg3[seg3_columns]

segments = [seg1, seg2, seg3]

# Look at different lags for each segment
models = []
for s in segments:
    model = VAR(s)
    models.append(model)
    #print(model.select_order())

# It looks like we can't meaningfully analyze Segment 1 with VAR. There's probably not enough observations.
segments = [seg2, seg3]
models = models[1:]
lags = [4, 1]

# Fit a VAR per segment, print results
for i in range(len(segments)):
    model_fitted = models[i].fit(lags[i])
    #print(model_fitted.summary())

# For each segment, input the dependents, independents, coefficients and their lags
seg2_ys = ['MIC', 'MIC', 'Pitch STD', 'Pitch STD', 'Pitch STD', 'Pitch STD', 'MIS', 'MIS', 'MIS', 'MIS', 'MIS', 'MIS', 'MIS', 'MIS']
seg2_predictors = ['RIC', 'RIC', 'MIC', 'MIS', 'MIS', 'nPVI', 'RIC', 'MIC', 'MIS', 'nPVI', 'nPVI', 'Pitch STD', 'MIS', 'nPVI']
seg2_lags = [1, 4, 1, 2, 4, 4, 1, 2, 2, 2, 3, 4, 4, 4]
seg2_coefs = [0.7266, 0.7206, 1.042, -.7638, -.5232, -.8178, .8768, .8881, -.6766, -.8314, 1.046, .9774, -.6026, -.5494]

seg3_ys = ['Tonality', 'Tonality', 'Pitch STD', 'Pitch STD', 'Onset Density', 'Onset Density', 'Onset Density', 'Onset Density', 'Onset Density', 'nPVI', 'nPVI', 'RIC', 'RIC', 'RIC']
seg3_predictors = ['nPVI', 'RIC', 'nPVI', 'RIC', 'const', 'Tonality', 'Onset Density', 'nPVI', 'RIC', 'nPVI', 'RIC', 'Tonality', 'nPVI', 'RIC']
seg3_lags = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
seg3_coefs = [.5926, -.6967, 1.044,-1.290, .6881, -.4564,.4602,-1.582,1.777,2.067,-1.948,-.2737,1.697,-1.653]

seg2_df = pd.DataFrame(list(zip(seg2_ys, seg2_predictors, seg2_lags, seg2_coefs)), columns =['Dependent', 'Predictor', 'Lag', 'Coefficient'])
seg3_df = pd.DataFrame(list(zip(seg3_ys, seg3_predictors, seg3_lags, seg3_coefs)), columns =['Dependent', 'Predictor', 'Lag', 'Coefficient'])

seg2_model = compute_model_values(seg2, seg2_df,2)
seg3_model = compute_model_values(seg3, seg3_df,3)

# Save model coefficients
seg2_df.to_csv(seg2_filename, index=False)
seg3_df.to_csv(seg3_filename, index=False)

print("Segment 2 Coefficients")
print(seg2_df)
print()
print("Segment 3 Coefficients")
print(seg3_df)
