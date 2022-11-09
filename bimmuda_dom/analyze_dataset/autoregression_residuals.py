# Imports
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

import statsmodels.api as sm

# Read in the data
ts_filename = '/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/output_data/dom_time_series_smoothed.csv'

# Z-score
def normalize(lst):
    rnge = max(lst) - min(lst)
    lst2 = [(x - min(lst))/rnge for x in lst]
    return lst2

def nrmse_range(original_vals, model_vals):

    lst1 = original_vals
    lst2 = model_vals
    errors = [0]*(len(lst1) - 1)

    # Iterate through the two lists simultaneously
    for i in range(len(lst1)):
        if i == 0:
            continue
        else:
            x1 = lst1[i]
            x2 = lst2[i]
            # L2 error between the two values
            errors[i-1] = (x1-x2)**2

    # Take the mean and square root
    avg_error = (sum(errors)/len(errors))
    rmse = np.sqrt(avg_error)
    # Normalize by the range
    minimum = min(original_vals)
    maximum = max(original_vals)
    nmrse = rmse/(maximum-minimum)

    return nmrse

ts_df = pd.read_csv(ts_filename)
ts_df = ts_df[['Tonality', 'MIC', 'Pitch STD', 'MIS', 'Onset Density', 'nPVI', 'RIC']].dropna().reset_index(drop=True)

# Normalize data
for col in ts_df.columns:
    ts_df[col] = normalize(list(ts_df[col]))

# Segment into eras
seg1 = ts_df.iloc[0:13]
seg2 = ts_df.iloc[13:48]
seg3 = ts_df.iloc[48:]

eras = [seg1, seg2, seg3]

"""
Step 1 is to fit an autoregressive model for each feature.
"""

seg1_maxlag = 3
seg2_maxlag = 7
seg3_maxlag = 4

maximum_lags = [seg1_maxlag, seg2_maxlag, seg3_maxlag]

new_eras = []

# For each era
for i in range(len(eras)):
    df = eras[i]
    #print("Segment:", i+1)
    max_lag = maximum_lags[i]
    # Empty DataFrame
    new_df = pd.DataFrame(columns=df.columns)
    # For each feature
    for col in ts_df.columns:
        aics = []
        bics = []
        hqics = []
        series = df[col]
        series = series.reset_index(drop=True)
        nrmses = []
        for l in range(1, max_lag + 1):
            model = AutoReg(series, lags = l, old_names = True)
            results = model.fit()
            fitted = list(results.fittedvalues)
            # Add the first l elements of the original to the beginning of the fitted values for nRMSE purposes
            first_l = list(series)[:(l+1)]
            fitted = first_l + fitted
            # Compute nRMSE between fitted and original values
            nrmse = nrmse_range(list(series), fitted)
            nrmse_mod = nrmse
            nrmses.append(nrmse_mod)

        # Choose the lag with the lowest modified nRMSE
        best_nrmse = min(nrmses)
        best_lag = nrmses.index(min(nrmses)) + 1
        #print(col, "has best lag", best_lag, "and nRMSE", best_nrmse)

        # Fit a model with this lag
        model = AutoReg(series, lags = best_lag, old_names = True)
        results = model.fit()
        fitted = list(results.fittedvalues)

        # Get the residuals
        residuals = list(results.resid)

        # Put N/A for the first best_lag values
        residuals = [np.nan]*best_lag + residuals

        new_df[col] = residuals

    new_eras.append(new_df)

results_dfs = []
for i in range(len(new_eras)):
    new_era_df = new_eras[i]
    old_df = eras[i]

    dependents = []
    independents = []
    r_2s = []
    intercepts = []
    coefs = []
    for col in new_era_df.columns:
        resid = list(new_era_df[col])
        nan_count = sum([math.isnan(x) for x in resid])
        resid = resid[nan_count:]
        for col2 in old_df.columns:
            if col == col2:
                continue
            else:
                feat = list(old_df[col2])
                feat = feat[nan_count:]
                feat = [[x] for x in feat]
                model = LinearRegression()
                model_fit = model.fit(feat, resid)
                r_sq = model.score(feat, resid)
                intercept = model.intercept_
                coef = model.coef_[0]

                dependents.append(col)
                independents.append(col2)
                r_2s.append(r_sq)
                intercepts.append(intercept)
                coefs.append(coef)

    results_df = pd.DataFrame(list(zip(dependents, independents, r_2s, intercepts, coefs)), columns = ['Dependent', 'Independent', 'R^2', 'Intercept', 'Coefficient'])
    results_dfs.append(results_df)

# Read out the results
era1_results_name = '/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/output_data/era1_single_regression_results.csv'
#results_dfs[0].to_csv(era1_results_name, index=False)

era2_results_name = '/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/output_data/era2_single_regression_results.csv'
#results_dfs[1].to_csv(era2_results_name, index=False)

era3_results_name = '/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/output_data/era3_single_regression_results.csv'
#results_dfs[2].to_csv(era3_results_name, index=False)

r_squared_threshold = 0.2

filtered_regression_results = []
for seg in results_dfs:
    filtered_regression_results.append(seg[seg['R^2'] >= r_squared_threshold].reset_index(drop=True))


print(filtered_regression_results[0])
print(filtered_regression_results[1])
print(filtered_regression_results[2])

# Distribution of R^2
era1_r2s = list(results_dfs[0]['R^2'])
era2_r2s = list(results_dfs[1]['R^2'])
era3_r2s = list(results_dfs[2]['R^2'])

# Best fits
for i in range(len(eras)):
    reg_df = filtered_regression_results[i]
    resid_df = new_eras[i]
    old_df = eras[i]
    if not reg_df.empty:
        for index, row in reg_df.iterrows():
            dependent = row['Dependent']
            independent = row['Independent']
            coef = float(row['Coefficient'])
            intercept = float(row['Intercept'])
            r_sq = float(row['R^2'])
            indep_data = list(old_df[independent])
            # Get the residual
            y = list(resid_df[dependent])
            x_data = [x*coef + intercept for x in indep_data]

            #plt.plot(y, label = 'Residual')
            #plt.plot(x_data, label = 'Regression Line')
            #plt.legend()
            #title_str = "Era " + str(i+1) + " " + dependent + " regressed on " + independent + " (R^2 = " + str(str(round(r_sq, 2))) + ")"
            #plt.title(title_str)
            #plt.show()

dependent_list = ['Tonality', 'MIC', 'Pitch STD', 'MIS', 'Onset Density', 'nPVI', 'RIC']
seg1_predictors = [['Onset Density'], ['Tonality', 'Pitch STD', 'MIS', 'Onset Density', 'nPVI'], ['MIC'], ['Tonality', 'Pitch STD', 'nPVI'], ['MIC'], ['Tonality', 'MIC', 'MIS', 'Onset Density'], ['nPVI']]
seg2_predictors = [['MIC', 'MIS'], ['Onset Density'], ['Tonality', 'Onset Density', 'RIC'], ['Pitch STD', 'nPVI', 'RIC'], ['Pitch STD', 'nPVI', 'RIC'], ['MIC'], ['Pitch STD', 'Onset Density']]
seg3_predictors = [['MIC', 'Pitch STD', 'MIS', 'Onset Density'], ['Pitch STD', 'MIS'], ['Tonality', 'Onset Density', 'RIC'], ['Tonality', 'MIC', 'Pitch STD', 'nPVI', 'RIC'], ['RIC'], ['MIS', 'RIC'], ['nPVI']]

seg1_reg_dict = dict(zip(dependent_list, seg1_predictors))
seg2_reg_dict = dict(zip(dependent_list, seg2_predictors))
seg3_reg_dict = dict(zip(dependent_list, seg3_predictors))

era_dicts = [seg1_reg_dict, seg2_reg_dict, seg3_reg_dict]

# Fit multilinear regressions with selected predictors for each dependent
multi_reg_results = []
for i in range(len(new_eras)):
    seg_residuals = new_eras[i]
    predictor_df = eras[i]
    predictor_dict = era_dicts[i]

    dependents = []
    independents = []
    r_2s = []
    coefs = []
    intercepts = []
    for dep in dependent_list:
        predictor_data = predictor_df[predictor_dict[dep]]
        residual = seg_residuals[dep]
        number_nans = sum([math.isnan(x) for x in residual])
        predictor_data = predictor_data.iloc[number_nans:]
        residual = residual.dropna().reset_index(drop=True)

        X2 = sm.add_constant(predictor_data)
        est = sm.OLS(list(residual), X2)
        est2 = est.fit()
        print(est2.summary())

        #model = LinearRegression()
        #model.fit(predictor_data, residual)
        #r_sq = model.score(predictor_data, residual)
        #intercept = model.intercept_
        #coef = model.coef_

        dependents.append(dep)
        independents.append(str(predictor_dict[dep]))
        r_2s.append(r_sq)
        intercepts.append(intercept)
        coefs.append(coef)

    results_df = pd.DataFrame(list(zip(dependents, independents, r_2s, intercepts, coefs)), columns = ['Dependent', 'Independent', 'R^2', 'Intercept', 'Coefficients'])
    multi_reg_results.append(results_df)

# Filter: R^2 >= 0.5

filtered_multireg_results = []
for seg in multi_reg_results:
    filtered_multireg_results.append(seg[seg['R^2'] >= 0.0].reset_index(drop=True))

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

for i in range(len(filtered_multireg_results)):
    df = filtered_multireg_results[i]
    print("Era", i+1)
    for index, row in df.iterrows():
        print("Dependent:", row['Dependent'])
        print("Independent variables:", row['Independent'])
        print("Coefficients:", row['Coefficients'])
        print("Intercept:", row['Intercept'])
        print("R^2:", row['R^2'])
        print()

# Produce some figures of the regression lines
"""
for i in range(len(eras)):
    reg_df = filtered_multireg_results[i]
    resid_df = new_eras[i]
    old_df = eras[i]
    if not reg_df.empty:
        for index, row in reg_df.iterrows():
            dependent = row['Dependent']
            independent = row['Independent']
            coef = float(row['Coefficient'])
            intercept = float(row['Intercept'])
            r_sq = float(row['R^2'])
            indep_data = list(old_df[independent])
            # Get the residual
            y = list(resid_df[dependent])
            x_data = [x*coef + intercept for x in indep_data]

            #plt.plot(y, label = 'Residual')
            #plt.plot(x_data, label = 'Regression Line')
            #plt.legend()
            #title_str = "Era " + str(i+1) + " " + dependent + " regressed on " + independent + " (R^2 = " + str(str(round(r_sq, 2))) + ")"
            #plt.title(title_str)
            #plt.show()



"""
# Fit multilinear regressions with the selected
multi_reg_results = []
for i in range(len(filtered_regression_results)):
    df = filtered_regression_results[i]
    old_df = eras[i]
    all_dependents = list(df['Dependent'].unique())

    dependents = []
    independents = []
    r_2s = []
    coefs = []
    intercepts = []
    for dep in all_dependents:
        data = df[df['Dependent'] == dep]
        independent_variable_names = list(data['Independent'].unique())
        ind_var = old_df[independent_variable_names]
        dep_data = old_df[dep]
        number_nans = sum([math.isnan(x) for x in dep_data])
        ind_var = ind_var.iloc[number_nans:]
        model = LinearRegression()
        model.fit(ind_var, dep_data)
        r_sq = model.score(ind_var, dep_data)
        intercept = model.intercept_
        coef = model.coef_

        dependents.append(dep)
        independents.append(str(independent_variable_names))
        r_2s.append(r_sq)
        intercepts.append(intercept)
        coefs.append(coef)

    results_df = pd.DataFrame(list(zip(dependents, independents, r_2s, intercepts, coefs)), columns = ['Dependent', 'Independent', 'R^2', 'Intercept', 'Coefficients'])
    multi_reg_results.append(results_df)

"""
# Need to check if the signs of the coefficients are the same (single vs. multilinear regression)
for i in range(len(multi_reg_results)):
    print("Segment:", i+1)
    multi_df = multi_reg_results[i]
    df = filtered_regression_results[i]
    dependents = list(multi_df['Dependent'].unique())
    for dep in dependents:
        single_coef_df = df[df['Dependent'] == dep]
        single_coef_df = single_coef_df[['Independent', 'Coefficient']]
        multi_data = multi_df[multi_df['Dependent'] == dep]
        multi_data = multi_data[['Independent', 'Coefficients']]
        print("Dependent:", dep)
        print("Coefficients for single regression")
        print(single_coef_df)
        print("Coefficients for multiple linear regression")
        print(multi_data)
        print()
        print()
"""



# I need to make sure I have two datasets for R
# 1. Dataset containing predictors (normalized)
# 2. Dataset containing residuals
predictor_df_1 = seg1
predictor_df_2 = seg2
predictor_df_3 = seg3

resid_df_1 = new_eras[0]
resid_df_2 = new_eras[1]
resid_df_3 = new_eras[2]

predictor_df_1.to_csv("/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/output_data/era1_predictors.csv", index=False)
predictor_df_2.to_csv("/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/output_data/era2_predictors.csv", index=False)
predictor_df_3.to_csv("/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/output_data/era3_predictors.csv", index=False)

resid_df_1.to_csv("/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/output_data/era1_residuals.csv", index=False)
resid_df_2.to_csv("/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/output_data/era2_residuals.csv", index=False)
resid_df_3.to_csv("/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/output_data/era3_residuals.csv", index=False)
