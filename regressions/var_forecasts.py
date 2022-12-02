# Imports
from regression_helper import un_normalize
from matplotlib.gridspec import GridSpec
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

"""
var_forecasts.py produces Figure 3 in the paper, which visualizes the 2022 forecasts produced by the Era 3 VAR
along with the Era 3 time series to contextualize them.

Inputs:
        - .csv of the smoothed time series
        - .csv of the unsmoothed time series
        - .csv of VAR forecasts
Outputs:
        - .png containing Figure 3
"""

# SPECIFY BASE DIRECTORY
base_dir = "/Users/madelinehamilton/Documents/python_stuff/death_of_melody/"

# Directory for smoothed time series
ts_filename = os.path.join(base_dir, "output_data/dom_time_series_smoothed.csv")

# Directory for unsmoothed time series
ts_unsmoothed_filename = os.path.join(base_dir, "output_data/dom_time_series_unsmoothed.csv")

# Directory for the VAR forecasts
var_forecast_filename = os.path.join(base_dir, "output_data/era_3_var_forecasts.csv")

# Directory for Figure 3
figure_3_filename = os.path.join(base_dir, "visualizations/var_forecasts.png")

"""
Read in the data
"""

# Read in the smoothed and unsmoothed time series
ts_df = pd.read_csv(ts_filename)
ts_df = ts_df[['Year', 'Tonality', 'MIC', 'Pitch STD', 'MIS', 'Onset Density', 'nPVI', 'RIC']].dropna().reset_index(drop=True)
ts_unsmoothed_df = pd.read_csv(ts_unsmoothed_filename)

# Read in the VAR forecasts
forecast_df = pd.read_csv(var_forecast_filename)

"""
Currently, the forecasted values are normalized. We need to 'de-normalize' them so we can visualize them with the original
time series.
"""

features = ['Tonality', 'MIC', 'Pitch STD', 'MIS', 'Onset Density', 'nPVI', 'RIC']
un_normalized_forecasts = {'Year': 2022}

for feat in features:
    orig = list(ts_df[feat])
    new = list(forecast_df[feat])
    un_normalized_forecasts[feat] = un_normalize(orig, new)

forecast_df = pd.DataFrame.from_dict(un_normalized_forecasts)

"""
Figure 3 visualizes all seven features for 2000 - 2023. 2000 - 2019 values will come from the smoothed time series.
2020 - 2021 values will come from the unsmoothed time series. 2022 values will come from the forecast DataFrame.

Concatenate the time series accordingly.
"""

# Unsmoothed values
# Actually no
# I think I want to use these smoothed values (2-backward smoothing) for 2020 and 2021
# It makes the figure look better and puts the forecasts in better context
inter_dict = {'Tonality': [0.706253216092627, 0.7058976862572162], 'MIC': [2.8950597566160257, 2.9329892633806915], 'Pitch STD': [2.6491987114871063, 2.663810535654321], 'MIS': [1.967044197510767, 2.014813221179121], 'Onset Density': [3.094186863017617, 3.0810329518141293], 'nPVI': [25.18254411099237, 26.225023707887303], 'RIC': [1.7033352473386942, 1.7369489163840854]}
inter_df = pd.DataFrame(inter_dict)

# Add these on to the smoothed DataFrame
ts_df = ts_df.append(inter_df)
ts_df = ts_df.reset_index(drop=True)
ts_df = ts_df[48:]

# Add on the forecasts
ts_plus_forecast = ts_df.append(forecast_df)

"""
Create and save Figure 3.
"""

# Matplotlib settings
plt.style.use('default')
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 9
sns.set_context('paper', font_scale=0.9)
plt.rcParams['figure.constrained_layout.use'] = True

# X axis values and ticks
years = list(range(2000, 2022))
years_forecast = list(range(2000, 2023))

years_for_ticks = list(range(2000, 2022))
years_ticks = years_for_ticks[0::10]

# Initialize figure
fig = plt.figure(constrained_layout=True)
gs = GridSpec(3, 3, figure=fig)

# Tonality
title_string =  'Tonality'
ax0 = fig.add_subplot(gs[0, :1])
ax0.plot(years, list(ts_df[title_string]), label = 'Historic Values', zorder = 10)
ax0.plot(years_forecast, list(ts_plus_forecast[title_string]), label = 'Forecast Value', zorder = 0)
ax0.set_xticks(years_ticks)
ax0.set_yticks([0.70, 0.72, 0.74])
ax0.set_ylabel('correlation coef.')
ax0.title.set_text(title_string)

# MIC
title_string =  'MIC'
ax1 = fig.add_subplot(gs[0, 1:2])
ax1.plot(years, list(ts_df[title_string]), label = 'Historic Values', zorder = 10)
ax1.plot(years_forecast, list(ts_plus_forecast[title_string]), label = 'Forecast Value', zorder = 0)
ax1.set_xticks(years_ticks)
ax1.set_yticks([2.8, 3.0, 3.2])
ax1.set_ylabel('bits')
ax1.title.set_text(title_string)

# Pitch STD
title_string = 'Pitch STD'
ax2 = fig.add_subplot(gs[0, 2:3])
ax2.plot(years, list(ts_df[title_string]), label = 'Historic Values', zorder = 10)
ax2.plot(years_forecast, list(ts_plus_forecast[title_string]), label = 'Forecast Value', zorder = 0)
ax2.set_xticks(years_ticks)
ax2.set_yticks([2.6, 2.7, 2.8, 2.9])
ax2.set_ylabel('MIDI note #s')
ax2.title.set_text(title_string)

# MIS
title_string = 'MIS'
ax3 = fig.add_subplot(gs[1, 0:1])
ax3.plot(years, list(ts_df[title_string]), label = 'Historic Values', zorder = 10)
ax3.plot(years_forecast, list(ts_plus_forecast[title_string]), label = 'Forecast Value', zorder = 0)
ax3.set_xticks(years_ticks)
ax3.set_ylabel('MIDI note #s')
ax3.title.set_text(title_string)

# Onset Density
title_string = 'Onset Density'
ax4 = fig.add_subplot(gs[1, 1:2])
ax4.plot(years, list(ts_df[title_string]), label = 'Historic Values', zorder = 10)
ax4.plot(years_forecast, list(ts_plus_forecast[title_string]), label = 'Forecast Value', zorder = 0)
ax4.set_xticks(years_ticks)
ax4.set_yticks([2.4, 2.6, 2.8, 3.0])
ax4.set_ylabel('notes/second')
ax4.title.set_text(title_string)

# nPVI
title_string = 'nPVI'
ax5 = fig.add_subplot(gs[1, 2:3])
ax5.plot(years, list(ts_df[title_string]), label = 'Historic Values', zorder = 10)
ax5.plot(years_forecast, list(ts_plus_forecast[title_string]), label = 'Forecast Value', zorder = 0)
ax5.set_xticks(years_ticks)
ax5.set_yticks([25,30,35])
ax5.set_ylabel('nPVI Value')
ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax5.title.set_text(title_string)

# RIC
title_string = 'RIC'
ax6 = fig.add_subplot(gs[2, 1:2])
ax6.plot(years, list(ts_df[title_string]), label = 'Historic Values', zorder = 10)
ax6.plot(years_forecast, list(ts_plus_forecast[title_string]), label = 'Forecast Value', zorder = 0)
ax6.set_xticks(years_ticks)
ax6.set_xlabel('Year')
ax6.set_ylabel('bits')
ax6.title.set_text(title_string)

# Save figure
plt.savefig(figure_3_filename, bbox_inches='tight')
