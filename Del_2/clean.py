import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('stock_data.csv', parse_dates=['Date'])

def calculate_returns(closing_price, previous_price):
    return math.log(closing_price / previous_price)

def calculate_volatility(dataframe, window=20, column='Close'):
    # Make a copy to avoid modifying the original dataframe
    df_vol = dataframe.copy()
    
    # Calculate daily returns
    df_vol['Returns'] = np.log(df_vol[column] / df_vol[column].shift(1))
    
    # Calculate rolling volatility (standard deviation of returns)
    df_vol['Volatility'] = df_vol['Returns'].rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    return df_vol

def plot_volatility(dataframe):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot price
    ax1.plot(dataframe['Date'], dataframe['Close'], color='blue')
    ax1.set_title('Stock Price')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    
    # Plot volatility
    ax2.plot(dataframe['Date'], dataframe['Volatility'], color='red')
    ax2.set_title(f'Rolling Volatility (Window: {window} days)')
    ax2.set_ylabel('Annualized Volatility')
    ax2.set_xlabel('Date')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

# Example usage
if __name__ == "__main__":
    window = 20  # 20-day rolling window
    
    # Calculate volatility
    df_with_volatility = calculate_volatility(df, window=window)
    
    # Display first few rows to check
    print(df_with_volatility[['Date', 'Close', 'Returns', 'Volatility']].head(10))
    
    # Plot the results
    plot_volatility(df_with_volatility)
    plt.show()
    
    # Interpretation
    avg_vol = df_with_volatility['Volatility'].mean()
    max_vol = df_with_volatility['Volatility'].max()
    min_vol = df_with_volatility['Volatility'].min()
    
    print(f"\nVolatility Analysis:")
    print(f"Average Volatility: {avg_vol:.4f}")
    print(f"Maximum Volatility: {max_vol:.4f}")
    print(f"Minimum Volatility: {min_vol:.4f}")

