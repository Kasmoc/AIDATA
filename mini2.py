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

def iqr_outliers(df, column=['Close']):
    Q1 = df['Close'].quantile(0.25)
    Q3 = df['Close'].quantile(0.75)
    IQR = Q3 - Q1

    # Outlier-grænser
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df['Close'] < lower_bound) | (df['Close'] > upper_bound)]

    return outliers

# Kald funktionen
outliers = iqr_outliers(df)

df['Outlier'] = False
df.loc[outliers.index, 'Outlier'] = True

# Plot
plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Close'], label='Close Price', alpha=0.6)
plt.scatter(df[df['Outlier']]['Date'], df[df['Outlier']]['Close'], 
            color='red', label='Outliers', zorder=5)
plt.title("Outliers i aktiepriser over tid")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()

# 🖨 Valgfrit: Vis antal outliers
print(f"Antal outliers i 'Close': {len(outliers)}")


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
