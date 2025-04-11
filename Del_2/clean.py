import pandas as pd


#Load dataset
df = pd.read_csv('stock_data.csv', parse_dates=['Date'])

print(df.head())
print(df.info())
