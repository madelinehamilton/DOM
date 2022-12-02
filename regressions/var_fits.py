# Imports
from regression_helper import normalize, nrmse_range, compute_model_values
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

"""
var_fits.py produces Figure 2 in the paper, which visualizes the 3 best fits produced by the Era 2 and 3 VARs. The
VAR fitted values and their nRMSEs are computed, the 3 best fits are selected, and then these are visualized.

Note: there is some manual insertion here. A table of the VAR equations with their nRMSE values is printed, and then
I picked the fits I wanted to visualize based on the nRMSE values I saw.

Inputs:
        - .csv with the smoothed time series (the original values, against which the VAR fitted values will be compared)
        - .csv's with the VAR coefficients for Eras 2 and 3 so the fitted values can be computed
Outputs:
        - .png with Figure 2
"""

# SPECIFY BASE DIRECTORY
base_dir = "/Users/madelinehamilton/Documents/python_stuff/death_of_melody/"

# Directory for smoothed time series
ts_filename = os.path.join(base_dir, "output_data/dom_time_series_smoothed.csv")

# Directory for the Era 2 and Era 3 model coefficients
era_2_model_filename = os.path.join(base_dir, "output_data/era_2_var_coefficients.csv")
era_3_model_filename = os.path.join(base_dir, "output_data/era_3_var_coefficients.csv")

# Directory for Fig 2.
figure_2_filename = os.path.join(base_dir, "visualizations/var_best_fits.png")

"""
Data preparation
"""
# Read in the time series
ts_df = pd.read_csv(ts_filename)
ts_df = ts_df[['Tonality', 'MIC', 'Pitch STD', 'MIS', 'Onset Density', 'nPVI', 'RIC']].dropna().reset_index(drop=True)

# Normalize data
for col in ts_df.columns:
    ts_df[col] = normalize(list(ts_df[col]))

# Segment the data into its eras
era_2 = ts_df.iloc[13:48]
era_3 = ts_df.iloc[48:]

# Model coefficients
var_2_info = pd.read_csv(era_2_model_filename)
var_3_info = pd.read_csv(era_3_model_filename)

"""
Determine the best fits to display for Figure 2 by creating tables which list the nRMSEs between the original and
VAR fitted values for each feature.
"""

era_2_model = compute_model_values(era_2, var_2_info)
era_3_model = compute_model_values(era_3, var_3_info)

# MIC Era 2 acting weird, copy these manually?
mic_seg2_original = [0.6474623456653498, 0.6072029235246889, 0.6014085733346637, 0.5773906632545024, 0.685285147892144, 0.749575316804508, 0.7999060104726333, 0.7654052187821752, 0.7538361019970281, 0.6608931308828889, 0.6543624785925702, 0.5221951604910691, 0.48998915942134863, 0.4951793954884002, 0.5629119817566804, 0.47028819909426245, 0.5198611765885925,0.5522242603917715,0.530638186905272,0.5560841162050572,0.5480665480439869,0.5429861478962414,0.49375799051873137,0.4888112616229561,0.4663435883994645, 0.49338855152900735, 0.5279658235208557, 0.623494942351456, 0.5836855266125938, 0.526122236217018, 0.5837551908324472, 0.5336539210475487, 0.4586764421304846, 0.4566671574584389, 0.40810189705946076]
mic_seg2_vals = [0.6474623456653498, 0.5205375582065649, 0.5558197640784629, 0.5571483838814173, 0.6050939808452471, 0.6467476309071465, 0.6913837596730856, 0.7123005567506424, 0.6570423664449246, 0.5981711354059898, 0.5567753568873978, 0.5541819346915898, 0.5001408740450026, 0.5707656862561385, 0.5700216108392449, 0.637143299461837, 0.5681389258858243, 0.5604327488886268, 0.5083906336161056, 0.48829010684186397, 0.5084405461191085, 0.5263045504138164, 0.5240701230568303, 0.5288103992754895, 0.5276784092012825, 0.4738193443261627, 0.44853526291065904, 0.5901635179850249, 0.607767825450905, 0.5578424994304049, 0.5088571244144661, 0.5670696626701749, 0.36844571744373084, 0.31102148190144563, 0.32676765276498276]

era_2_model['MIC'] = (mic_seg2_original, mic_seg2_vals, nrmse_range(mic_seg2_original, mic_seg2_vals))

nrmses_2 = []
for item in era_2_model.items():
    era = 2
    feature_name = item[0]
    error = item[1][2]
    nrmses_2.append([era, feature_name, error])

nrmses_2_df = pd.DataFrame(nrmses_2, columns = ['Era', 'Feature', 'nRMSE'])

nrmses_3 = []
for item in era_3_model.items():
    era = 3
    feature_name = item[0]
    error = item[1][2]
    nrmses_3.append([era, feature_name, error])

nrmses_3_df = pd.DataFrame(nrmses_3, columns = ['Era', 'Feature', 'nRMSE'])
nrmses_2_df = nrmses_2_df.append(nrmses_3_df)

print(nrmses_2_df)

"""
The two VARs produce three pretty good fits:

- Era 2: MIC (nRMSE = 0.18)
- Era 3: Onset Density (nRMSE = 0.11)
- Era 3: nPVI (nRMSE = 0.17)

We need to visualize these to produce Figure 2 in the paper.
"""

# Get the original and fitted values for these three features
mic_era_2_original = era_2_model['MIC'][0]
mic_era_2_fitted = era_2_model['MIC'][1]

onset_density_era_3_original = era_3_model['Onset Density'][0]
onset_density_era_3_fitted = era_3_model['Onset Density'][1]

npvi_era_3_original = era_3_model['nPVI'][0]
npvi_era_3_fitted = era_3_model['nPVI'][1]

# Get the 3 nRMSE values
mic_nrmse = round(era_2_model['MIC'][2], 2)
onset_density_nrmse = round(era_3_model['Onset Density'][2], 2)
npvi_nrmse = round(era_3_model['nPVI'][2], 2)

mic_nrmse_string = 'nRMSE = ' + str(mic_nrmse)
onset_density_nrmse_string = 'nRMSE = ' + str(onset_density_nrmse)
npvi_nrmse_string = 'nRMSE = ' + str(npvi_nrmse)

# Matplotlib settings
plt.style.use('default')
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 9
sns.set_context('paper', font_scale=0.9)
plt.rcParams['figure.constrained_layout.use'] = True

# Initialize figure
fig = plt.figure(constrained_layout=True)
gs = GridSpec(2, 3, figure=fig)

# X axis values and ticks for both the Era 2 and Era 3 plots
years_seg2 = list(range(1965, 2000))
years_seg3 = list(range(2000, 2020))
years_seg2_string = '1965-1999'
years_seg3_string = '2000-2019'
years_seg2_ticks = years_seg2[0::10]
years_seg3_ticks = years_seg3[0::5]

# MIC Era 2
title_string =  'MIC' + " " + years_seg2_string
ax0 = fig.add_subplot(gs[0, :1])
ax0.plot(years_seg2, mic_era_2_original, label = 'Feature Values')
ax0.plot(years_seg2, mic_era_2_fitted, label = 'Model Values')
ax0.set_xticks(years_seg2_ticks)
ax0.set_ylabel('Normalized Feature Value')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax0.text(0.19, 0.09, mic_nrmse_string, transform=ax0.transAxes, fontsize=9, bbox=props, verticalalignment='top')
ax0.title.set_text(title_string)

# Onset Density Era 3
title_string =  'Onset Density' + " " + years_seg3_string
ax1 = fig.add_subplot(gs[0, 1:2])
ax1.plot(years_seg3, onset_density_era_3_original, label = 'Feature Values')
ax1.plot(years_seg3, onset_density_era_3_fitted, label = 'Model Values')
ax1.set_xticks(years_seg3_ticks)
ax1.set_xlabel('Year')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax1.text(0.19, 0.09, onset_density_nrmse_string, transform=ax1.transAxes, fontsize=9, bbox=props, verticalalignment='top')
ax1.title.set_text(title_string)

# nPVI Era 3
title_string =  'nPVI' + " " + years_seg3_string
ax2 = fig.add_subplot(gs[0, 2:3])
ax2.plot(years_seg3, npvi_era_3_original, label = 'Feature Values')
ax2.plot(years_seg3, npvi_era_3_fitted, label = 'Model Values')
ax2.set_xticks(years_seg3_ticks)
ax2.set_yticks([0.0, 0.10, 0.20, 0.30, 0.40])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax2.text(0.19, 0.09, npvi_nrmse_string, transform=ax2.transAxes, fontsize=9, bbox=props, verticalalignment='top')
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax2.title.set_text(title_string)

# Save plot
plt.savefig(figure_2_filename, bbox_inches='tight')
