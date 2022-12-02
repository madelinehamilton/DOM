# The Death of Melody? Trajectories and Revolutions in Western Pop Music

This is the code base for the Death of Melody project. Python 3 and R are utilized for analysis. For reproduction, please ensure the following Python 
libraries are installed on your machine: pretty_midi, pandas, numpy, collections, json, scipy, statsmodels, sklearn, matplotlib.pyplot, matplotlib.gridspec, and seaborn, as well as the "ecp" and "ggplot2" R libraries.

Below are reproduction tips per folder. To run individual scripts, you'll always need to specify the directory to which you downloaded the code. 

# create_timeseries

1. Unzip the "midis" folder, keeping it at the same level as the other folders. 

2. Run compute_dom_features_no_idyom.py to compute the features that do not require IDyOM. 

3. Compute the IDyOM features (see idyom_feature_instructions.txt). Put the two .dat files in the /output_data folder, one for each feature. Name them "mic_from_idyom.dat" and "ric_from_idyom.dat"

4. Run produce_full_dom_dataset.py. The full time series will be in /output_data.

# changepoint_detection
Make sure you have run the scripts in create_timeseries beforehand. 

1. Run changepoint_analysis.R to obtain the changepoints for the seven time series using the 60 different parameter settings.

2. Run changepoint_tally.py to determine the "true" changepoints of the time series. 

3. Run time_series_changepoint_visual.R to (partially) reproduce Figure 1.

4. Run visualize_changepoint_tallies.py to reproduce Figure S1.

# regressions
Make sure you have run the scripts in create_timeseries beforehand. It is not necessary to also have run the scripts in changepoint_detection, as the
times series are segmented according to the changepoints manually.

1. Run autoregression_residuals.py to fit AR models to the time series and reproduce Table 2.

2. Run residual_reg.py to regress the residuals from autoregression_residuals.py against the other feature values and reproduce Table 3. 
 
3. Run vector_autoregression.py to fit VAR models to the time series and produce forecasts for 2022.

4. Run var_fits.py to reproduce Figure 2, which visualizes the best VAR fits.

5. Run var_forecasts.py to reproduce Figure 3, which visualizes the forecasts yielded the VAR. 

# other
1. Run bimmuda_bpm.py to reproduce Figure S2. 

If you have any questions, feel free to contact the main author at m.a.hamilton@qmul.ac.uk



