import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import probplot
import matplotlib.pyplot as plt

class StockNoiseAnalyzer:
    def __init__(self, data, price_column='Close'):
        """
        Initialize the analyzer with stock price data
        
        Parameters:
        data (pd.DataFrame): DataFrame containing stock price data
        price_column (str): Name of the column containing price data
        """
        self.data = data
        self.price_column = price_column
        self.original_prices = data[price_column]
        
    def calculate_noise_statistics(self, data):
        """Calculate basic noise statistics for given data"""
        return {
            'Standard Deviation': np.std(data),
            'Variance': np.var(data),
            'Mean Absolute Deviation': np.mean(np.abs(data - np.mean(data)))
        }
    
    def apply_filters(self):
        """Apply different filtering methods to the price data"""
        # Simple Moving Average
        self.sma = self.original_prices.rolling(window=20, center=True).mean()
        
        # Exponential Moving Average
        self.ema = self.original_prices.ewm(span=20, adjust=False).mean()
        
        # Savitzky-Golay filter
        self.savgol = pd.Series(
            signal.savgol_filter(self.original_prices, 21, 3),
            index=self.original_prices.index
        )
        
        return self.sma, self.ema, self.savgol
    
    def analyze_noise_reduction(self, filtered_data):
        """Analyze noise reduction for a filtered series"""
        residuals = self.original_prices - filtered_data
        noise_stats = self.calculate_noise_statistics(residuals)
        return residuals, noise_stats
    
    def plot_analysis(self):
        """Create comprehensive visualization of the analysis"""
        # Apply filters
        self.apply_filters()
        
        # Create figure
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3)
        
        # 1. Original Time Series with Moving Average
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.original_prices, 'r', label='Closing Price', alpha=0.5)
        ax1.plot(self.sma, 'b', label='20-day Moving Average')
        ax1.set_title('Stock Price Time Series')
        ax1.legend()
        
        # 2. Daily Returns
        returns = self.original_prices.pct_change()
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(returns, 'r', alpha=0.5)
        ax2.set_title('Daily Returns (Showing Volatility)')
        
        # 3. Distribution of Returns
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(returns.dropna(), bins=50, color='red', alpha=0.6)
        ax3.set_title('Distribution of Daily Returns')
        
        # 4. Comparison of Filtering Methods
        ax4 = fig.add_subplot(gs[1, :])
        ax4.plot(self.original_prices, 'r', label='Original', alpha=0.5)
        ax4.plot(self.sma, 'b', label='SMA')
        ax4.plot(self.ema, 'g', label='EMA')
        ax4.plot(self.savgol, 'purple', label='Savitzky-Golay')
        ax4.set_title('Comparison of Filtering Methods')
        ax4.legend()
        
        # 5. Residuals Analysis
        ax5 = fig.add_subplot(gs[2, 0])
        for method, filtered_data, color in [
            ('SMA', self.sma, 'blue'),
            ('EMA', self.ema, 'green'),
            ('Savitzky-Golay', self.savgol, 'purple')
        ]:
            residuals, stats = self.analyze_noise_reduction(filtered_data)
            ax5.plot(residuals, color, label=f'{method}', alpha=0.5)
        ax5.set_title('Residuals Over Time')
        ax5.legend()
        
        # 6. Q-Q Plot
        ax6 = fig.add_subplot(gs[2, 1])
        residuals = self.analyze_noise_reduction(self.sma)[0]
        probplot(residuals.dropna(), dist="norm", plot=ax6)
        ax6.set_title('Q-Q Plot of Residuals (SMA)')
        
        # 7. Residuals Distribution
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.hist(residuals.dropna(), bins=50, color='blue', alpha=0.6)
        ax7.set_title('Distribution of Residuals')
        
        plt.tight_layout()
        plt.show()
        
    def print_statistics(self):
        """Print noise statistics for original and filtered data"""
        print("\nOriginal Data Statistics:")
        original_stats = self.calculate_noise_statistics(self.original_prices)
        for stat_name, value in original_stats.items():
            print(f"{stat_name}: {value:.6f}")
            
        for method, filtered_data in [
            ('SMA', self.sma),
            ('EMA', self.ema),
            ('Savitzky-Golay', self.savgol)
        ]:
            _, stats = self.analyze_noise_reduction(filtered_data)
            print(f"\nNoise Statistics after {method}:")
            for stat_name, value in stats.items():
                print(f"{stat_name}: {value:.6f}")

def analyze_stock_noise(df, price_column='Close'):
    """Main function to run the complete analysis"""
    analyzer = StockNoiseAnalyzer(df, price_column)
    analyzer.plot_analysis()
    analyzer.print_statistics()

# Usage:
df = pd.read_csv("C:\\Users\\tonny\\OneDrive\\Desktop\\AIDproj\\stock_data.csv")
analyze_stock_noise(df)