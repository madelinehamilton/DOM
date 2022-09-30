1. Ensure you have run all scripts in create_dataset. In particular, make sure that the .csv with the smoothed time series is in the output_data directory.

2. Run changepoint_analysis.R. The .csv with the changepoints should be in the output_data directory.

3. Run changepoint_tally.py, which takes the .csv of changepoints and applies various thresholds and filters to produce the "true' changepoints per feature. It will output a .csv with the changepoint tallies for Figure S1 in the supplementary index of the paper (see the "visualise" directory).