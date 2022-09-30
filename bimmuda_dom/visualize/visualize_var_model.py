# Imports
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
visualize_var_model.py produces Fig 2 in the DOM paper.
"""

# SPECIFY DESIRED OUTPUT DIRECTORY FOR FIGURE
fig_name = "/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/visualize/var_mat_fig.png"

# TO-DO: automate
# I'll just list the VAR coefficients manually
seg2_lst = [0, 1.0420, 0.8881, 0, 0, .9774, 0, -.6435, -.6396, 0, -.8178, -.1116, .7236, 0, .8768]
seg3_lst = [0, 0, -.4564, 0, -.2737, 0, 0, 0, 0, 0, 0, 0, .4602, 0, 0, .5926, 1.044, .4602, 2.0670, 1.6970, -.6967, -1.2900,1.7770,-1.9480,-1.653]

# Create datasets of coefficients for Segment 2 and Segment 3
# Dependent names
seg2_names = ['MIC', 'Pitch STD', 'MIS', 'nPVI', 'RIC']
seg2_df = pd.DataFrame(seg2_lst)
seg2_df.columns = ['Value']
seg2_names_col = ['MIC', 'Pitch STD', 'MIS', 'MIC', 'Pitch STD', 'MIS', 'MIC', 'Pitch STD', 'MIS', 'MIC', 'Pitch STD', 'MIS', 'MIC', 'Pitch STD', 'MIS']
seg2_preds_col = ['MIC', 'MIC', 'MIC', 'Pitch STD', 'Pitch STD', 'Pitch STD', 'MIS', 'MIS', 'MIS', 'nPVI', 'nPVI', 'nPVI', 'RIC', 'RIC', 'RIC']
seg2_df['Predictor'] = seg2_preds_col
seg2_df['Dependent'] = seg2_names_col
seg2_df = seg2_df.pivot("Dependent", "Predictor", "Value")
seg2_df = seg2_df.reindex(seg2_names, columns=seg2_names)
seg2_df = seg2_df.dropna()

seg3_names = ['Tonality', 'Pitch STD', 'Onset Density', 'nPVI', 'RIC']
seg3_df = pd.DataFrame(seg3_lst)
seg3_df.columns = ['Value']
seg3_names_col = ['Tonality', 'Pitch STD', 'Onset Density', 'nPVI', 'RIC', 'Tonality', 'Pitch STD', 'Onset Density', 'nPVI', 'RIC', 'Tonality', 'Pitch STD', 'Onset Density', 'nPVI', 'RIC', 'Tonality', 'Pitch STD', 'Onset Density', 'nPVI', 'RIC', 'Tonality', 'Pitch STD', 'Onset Density', 'nPVI', 'RIC']
seg3_preds_col = ['Tonality', 'Tonality', 'Tonality', 'Tonality', 'Tonality', 'Pitch STD', 'Pitch STD', 'Pitch STD', 'Pitch STD', 'Pitch STD', 'Onset Density', 'Onset Density', 'Onset Density', 'Onset Density', 'Onset Density','nPVI','nPVI','nPVI','nPVI','nPVI','RIC', 'RIC', 'RIC', 'RIC', 'RIC']
seg3_df['Predictor'] = seg3_preds_col
seg3_df['Dependent'] = seg3_names_col
seg3_df = seg3_df.pivot("Dependent", "Predictor", "Value")
seg3_df = seg3_df.reindex(seg3_names, columns=seg3_names)

# Figure settings
plt.style.use('default')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.autolayout'] = False
plt.rcParams['font.family'] = "Arial"
sns.set_context('paper', font_scale=0.9)
plt.rcParams["figure.figsize"] = (50,15)

fig = plt.figure()

# Segment 2
ax2 = fig.add_subplot(1,2,1)
ax2 = sns.heatmap(seg2_df, annot=True, cmap = "Spectral_r", vmin=-2.1, vmax=2.1, center=0, cbar=False, linewidth=.5, annot_kws={"fontsize":20})
ax2.set_yticklabels(('MIC          \n(nMRSE = 0.179)', 'Pitch STD      \n(nMRSE = 0.965)', 'MIS           \n(nMRSE = 0.756)'), rotation=0, fontsize="20", va="center")
ax2.set_xticklabels(('MIC', 'Pitch STD', 'MIS', 'nPVI', 'RIC'), fontsize="20")
ax2.set_ylabel('Dependent', fontsize="25")
ax2.set_xlabel('Predictor', fontsize="25")
plt.title("Segment 2: 1965 - 1996", fontsize="28")

sns.set(font_scale=2)

# Segment 3
ax3 = fig.add_subplot(1,2,2)
ax3 = sns.heatmap(seg3_df, annot=True, cmap = "Spectral_r", vmin=-2.1, vmax=2.1, center=0, linewidth=.5, annot_kws={"fontsize":20})
ax3.set_yticklabels(('Tonality        \n(nMRSE = 0.626)', 'Pitch STD     \n(nMRSE = 0.422)', 'Onset Density   \n(nMRSE = 0.143)', 'nPVI           \n(nMRSE = 0.186)', 'RIC           \n(nMRSE = 0.470)'),rotation=0, fontsize="20", va="center")
ax3.set_xticklabels(('Tonality', 'Pitch STD', 'Onset Density', 'nPVI', 'RIC'), fontsize="15", rotation=0)
ax3.set_ylabel('Dependent', fontsize="25")
ax3.set_xlabel('Predictor', fontsize="25")
plt.title("Segment 3: 1997 - 2019", fontsize="28")

# Save figure
plt.savefig(fig_name, bbox_inches='tight')
