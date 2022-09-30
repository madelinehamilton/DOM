# Imports
library(ggplot2)

# time_series_changepoint_visual.R produces the graphs necessary for
# Fig 1 in the paper. The plots are saved separately because the 
# full figure is assembled elsewhere, but this script will give
# nice visualizations of the individual features and their
# changepoints. 

# You will need to specify the changepoints manually by adding
# geom_vline(xintercept = changepoint, size=.5) lines to the plots

# Inputs 

# Import time series
# SPECIFY DIRECTORY OF THE .CSV OF THE SMOOTHED TIME SERIES
# PRODUCED BY produce_full_dom_dataset.py
df_directory <- '/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/dom_time_series_smoothed.csv'
df <- t(read.csv(file = df_directory))

# SPECIFY DESIRED NAMES OF VISUALIZATIONS
tonality_filename <- "tonality_viz.png"
mic_filename <- "mic_viz.png"
pitch_std_filename <- "pitch_std_viz.png"
mis_filename <- "mis_viz.png"
od_filename <- "od_viz.png"
npvi_filename <- "npvi_viz.png"
ric_filename <- "ric_viz.png"

# Re-configure df (R is hard)
tonality <- c(df['Tonality',])
mic <- c(df['MIC',])
pitch_std <- c(df['Pitch.STD',])
mis <- c(df['MIS',])
od <- c(df['Onset.Density',])
npvi <- c(df['nPVI',])
ric <- c(df['RIC',])
years <- c(seq(1950, 2021, 1))

df_viz <- data.frame(years, tonality, mic, pitch_std, mis, od, npvi, ric)

# FIGURES

# Tonality changepoints: 2000
ggplot(df_viz, aes(x=years)) + 
  geom_line(aes(y = tonality), color = "#F8766D") + 
  labs(x = "", y = "correlation coef.") +
  scale_x_continuous(breaks = c(seq(1950, 2021, by=5))) +
  scale_y_continuous(breaks = c(seq(.7, 1.0, by=.02))) +
  geom_vline(xintercept = 2000, size=.5) +
  scale_color_brewer(palette = "Set3") +
  theme(legend.position = "none") + 
  theme(aspect.ratio=.2)

ggsave(tonality_filename, path = "/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/time_series_viz")

# MIC changepoints: 1965, 1976, 2000
ggplot(df_viz, aes(x=years)) + 
  geom_line(aes(y = mic), color = "#C49A00") + 
  labs(x = "", y = "bits") +
  scale_x_continuous(breaks = c(seq(1950, 2021, by=5))) +
  geom_vline(xintercept = 1965, size=.5) +
  geom_vline(xintercept = 1976, size=.5) +
  geom_vline(xintercept = 2001, size=.5) +
  scale_color_brewer(palette = "Set3") +
  theme(legend.position = "none") +
  theme(aspect.ratio=.2)

ggsave(mic_filename, path = "/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/time_series_viz")

# Pitch STD changepoints: 1965, 1977, 2001
ggplot(df_viz, aes(x=years)) + 
  geom_line(aes(y = pitch_std), color = "#53B400") + 
  labs(x = "", y = "MIDI note #s") +
  scale_x_continuous(breaks = c(seq(1950, 2020, by=5))) +
  geom_vline(xintercept = 1965, size=.5) +
  geom_vline(xintercept = 1977, size=.5) +
  geom_vline(xintercept = 2001, size=.5) +
  scale_color_brewer(palette = "Set3") +
  theme(legend.position = "none") +
  theme(aspect.ratio=.2)

ggsave(pitch_std_filename, path = "/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/time_series_viz")

# MIS changepoints: 1965, 1995, 2009
ggplot(df_viz, aes(x=years)) + 
  geom_line(aes(y = mis), color = "#00C094") + 
  labs(x = "", y = "MIDI note #s") +
  scale_x_continuous(breaks = c(seq(1950, 2021, by=5))) +
  geom_vline(xintercept = 1965, size=.5) +
  geom_vline(xintercept = 1995, size=.5) +
  geom_vline(xintercept = 2009, size=.5) +
  scale_color_brewer(palette = "Set3") +
  theme(legend.position = "none") +
  theme(aspect.ratio=.2)

ggsave(mis_filename, path = "/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/time_series_viz")

# OD changepoints: 1965, 2000
ggplot(df_viz, aes(x=years)) + 
  geom_line(aes(y = od), color = "#00B6EB") + 
  labs(x = "", y = "notes/second") +
  scale_x_continuous(breaks = c(seq(1950, 2021, by=5))) +
  geom_vline(xintercept = 1965, size=.5) +
  geom_vline(xintercept = 2000, size=.5) +
  scale_y_continuous(breaks = c(seq(1.5, 3.0, by=.25)), limits = c(1.5, 3.0)) +
  scale_color_brewer(palette = "Set3") +
  theme(legend.position = "none") +
  theme(aspect.ratio=.2)

ggsave(od_filename, path = "/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/time_series_viz")

# nPVI changepoints: 1963, 2000
ggplot(df_viz, aes(x=years)) + 
  geom_line(aes(y = npvi), color = "#A58AFF") + 
  labs(x = "", y = "nPVI value") +
  scale_x_continuous(breaks = c(seq(1950, 2021, by=5))) +
  geom_vline(xintercept = 1963, size=.5) +
  geom_vline(xintercept = 2000, size=.5) +
  scale_y_continuous(breaks = c(seq(25, 60, by=5))) +
  scale_color_brewer(palette = "Set3") +
  theme(legend.position = "none") +
  theme(aspect.ratio=.2)

ggsave(npvi_filename, path = "/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/time_series_viz")

# RIC changepoints: 1974, 1997, 2007
ggplot(df_viz, aes(x=years)) + 
  geom_line(aes(y = ric), color = "#FB61D7") + 
  labs(x = "Year", y = "bits") +
  scale_x_continuous(breaks = c(seq(1950, 2021, by=5))) +
  geom_vline(xintercept = 1974, size=.5) +
  geom_vline(xintercept = 1997, size=.5) +
  geom_vline(xintercept = 2007, size=.5) +
  #scale_y_continuous(breaks = c(seq(8, 20, by=2))) +
  scale_color_brewer(palette = "Set3") +
  theme(legend.position = "none") + 
  theme(aspect.ratio=.2)

ggsave(ric_filename, path = "/Users/madelinehamilton/Documents/python_stuff/bimmuda_dom/time_series_viz")