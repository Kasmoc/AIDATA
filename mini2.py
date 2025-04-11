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

df['CustomMA'] = custom_moving_average(df['Close'], M=30)

plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Close'], label='Raw Close Price', alpha=0.5)
plt.plot(df['Date'], df['CustomMA'], label='30-Day Moving Average', color='red')
plt.title("Stock Prices Before and After Moving Average")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
