"""
Configuration settings for stock preprocessing
"""

# Default column settings
DEFAULT_PRICE_COLUMN = 'Close'

# Moving Average settings
DEFAULT_MA_WINDOW_SIZE = 30

# Frequency Filter settings
DEFAULT_CUTOFF_FREQ = 0.05
DEFAULT_FILTER_TYPES = ['lowpass', 'highpass']

# Plotting settings
PLOT_FIGSIZE = (15, 15)
MA_PLOT_FIGSIZE = (15, 5)

# File paths
STOCK_DATA_PATH = "Noise\stock_data.csv"

# Noise calculation settings
RETURNS_HIST_BINS = 50

# SNR calculation settings
SNR_EPSILON = 1e-10  # To avoid division by zero in SNR calculation 