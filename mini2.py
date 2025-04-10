import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

#Load dataset
data2 = pd.read_csv(r"C:\Users\anne\Desktop\Daki\s2\ai_and_data\miniprojekt\data2\AAPL_2006-01-01_to_2018-01-01.csv")

#print(data2.head())
#print(data2.info())

signal = data2[['Open', 'High', 'Low', 'Close', 'Volume']].values

features = ['Open', 'High', 'Low', 'Close', 'Volume']
fft_results = {}

for feature in features:
    sig = data2[feature].values
    fft = np.fft.fft(sig)
    fft_results[feature] = fft


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

fft = np.fft.fft(data2['Close'].values)
freqs = np.fft.fftfreq(len(fft), d=1)
magnitude = np.abs(fft)

# Only positive freqs
plt.plot(freqs[:len(freqs)//2], magnitude[:len(freqs)//2])
plt.title("Frequency Spectrum of Close Prices")
plt.xlabel("Frequency [1/days]")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()

'''
data2.plot(figsize=(10, 6))

plt.tight_layout()
plt.show()

window_size = 251
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    #data2[f'{col}_SMA_{window_size}'] = data2[col].rolling(window=window_size).mean()



plt.figure(figsize=(10, 5))
plt.plot(data2['Close'], label='Original')
plt.plot(data2['Close_SMA_251'], label='SMA (251)', linewidth=2)
plt.title("Close Price - Original vs Smoothed")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(data2['High'], label='Original')
plt.plot(data2['High_SMA_251'], label='SMA (251)', linewidth=2)
plt.title("High Price - Original vs Smoothed")
plt.xlabel("Time")
plt.ylabel("High")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()'
'''