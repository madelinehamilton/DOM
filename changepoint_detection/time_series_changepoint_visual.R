# Imports
library(ggplot2)

# time_series_changepoint_visual.R produces the graphs necessary for
# Fig 1 in the paper. The plots are saved separately because the 
# full figure is assembled elsewhere, but this script will give
# nice visualizations of the individual features and their
# changepoints. 

# Note: changepoints are specified manually by adding
# geom_vline(xintercept = changepoint, size=.5) lines to the plots
# So if you alter the changepoint detection settings and get 
# different changepoints, you will have to write them in manually
# when producing this figure

# SPECIFY BASE DIRECTORY
base_dir <- "/Users/madelinehamilton/Documents/python_stuff/death_of_melody/"

# Directory of the smoothed time series .csv
ts_name <- paste(base_dir, "output_data/dom_time_series_smoothed.csv", sep = "")

# Visualization directory
viz_dir <- paste(base_dir, "visualizations/timeseries_w_changepoints", sep = "")

# Import the times series
df_viz <- read.csv(file = ts_name)

# FIGURES

# You get a warning message; just leave it

# Tonality changepoints: 2000
ggplot(df_viz, aes(x=Year)) + 
  geom_line(aes(y = Tonality), color = "#F8766D") + 
  labs(x = "", y = "correlation coef.") +
  scale_x_continuous(breaks = c(seq(1950, 2021, by=5))) +
  scale_y_continuous(breaks = c(seq(.7, 1.0, by=.02))) +
  geom_vline(xintercept = 2000, size=.5) +
  scale_color_brewer(palette = "Set3") +
  theme(legend.position = "none") + 
  theme(aspect.ratio=.2)

ggsave("tonality.png", path = viz_dir)

# MIC changepoints: 1965, 1976, 2000
ggplot(df_viz, aes(x=Year)) + 
  geom_line(aes(y = MIC), color = "#C49A00") + 
  labs(x = "", y = "bits") +
  scale_x_continuous(breaks = c(seq(1950, 2021, by=5))) +
  geom_vline(xintercept = 1965, size=.5) +
  geom_vline(xintercept = 1976, size=.5) +
  geom_vline(xintercept = 2001, size=.5) +
  scale_color_brewer(palette = "Set3") +
  theme(legend.position = "none") +
  theme(aspect.ratio=.2)

ggsave("mic.png", path = viz_dir)

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

ggsave("pitch_std.png", path = viz_dir)

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

ggsave("mis.png", path = viz_dir)

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

ggsave("onset_density.png", path = viz_dir)

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

ggsave("npvi.png", path = viz_dir)

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

ggsave("ric.png", path = viz_dir)