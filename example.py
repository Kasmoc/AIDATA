import pandas as pd
from Noise.stock_preprocessor import preprocess_stock_data

# Download sample stock data
ticker = "AAPL"
data = pd.read_csv("C:\\Users\\tonny\\OneDrive\\Desktop\\AIDproj\\stock_data.csv")

# Preprocess the data
ma_results, freq_filtered_low, freq_filtered_high = preprocess_stock_data(data)

# The results contain:
# ma_results: dictionary of different moving averages (SMA and EMA)
# freq_filtered_low: lowpass filtered data
# freq_filtered_high: highpass filtered data (noise component) 