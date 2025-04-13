import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

#Load dataset
df = pd.read_csv(r"stock_data.csv", parse_dates=['Date'])
df.sort_values('Date', inplace=True)

print(df.head())
print(df.info())

plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Close'], label='Raw Close Price')
plt.title("Raw Stock Prices (Close)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()


def custom_moving_average(signal, M):
    N = len(signal)
    output = []

    for i in range(N):
        if i < M - 1:
            avg = np.mean(signal[:i + 1])
        else:
            avg = np.mean(signal[i - M + 1:i + 1])
        output.append(avg)

    return np.array(output)

df['CustomMA'] = custom_moving_average(df['Close'], M=20)

plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Close'], label='Raw Close Price', alpha=0.5)
plt.plot(df['Date'], df['CustomMA'], label='20-Day Moving Average', color='red')
plt.title("Stock Prices Before and After Moving Average")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# Extract Close prices
close_prices = df['Close'].values

# Detrend by subtracting the mean (centering the data)
detrended_prices = close_prices - np.mean(close_prices)

# Perform FFT
fft_values = np.fft.fft(detrended_prices)
fft_frequencies = np.fft.fftfreq(len(detrended_prices))

# Only positive frequencies
positive_freq_indices = fft_frequencies > 0

# Plot FFT magnitude
plt.figure(figsize=(12, 6))
plt.plot(fft_frequencies[positive_freq_indices], np.abs(fft_values)[positive_freq_indices])
plt.title("FFT Frequency Spectrum of Detrended Close Prices")
plt.xlabel("Frequency (cycles per day)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()

# Define cutoff frequency (keep only slow trends)
cutoff = 0.02

# Create a filtered FFT version (zero out high frequencies)
fft_filtered = fft_values.copy()
fft_filtered[np.abs(fft_frequencies) > cutoff] = 0

# Inverse FFT to reconstruct cleaned signal in time domain
filtered_prices = np.fft.ifft(fft_filtered).real + np.mean(close_prices)

# Plot original vs. cleaned signal
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], close_prices, label='Original Close Prices', alpha=0.5)
plt.plot(df['Date'], filtered_prices, label='Cleaned (Low-Pass Filtered)', color='red')
plt.title("Original vs Cleaned Stock Prices using FFT Filtering (Cutoff = 0.02)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.show()
