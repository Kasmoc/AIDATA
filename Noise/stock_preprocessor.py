import numpy as np
import pandas as pd
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from config import *

class StockPreprocessor:
    def __init__(self, data, price_column=DEFAULT_PRICE_COLUMN):
        """
        Initialize preprocessor with stock price data.

        Parameters:
        - data (pd.DataFrame): DataFrame containing stock price data.
        - price_column (str): Name of the column containing price data.
        """
        # Error handling for input data
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input 'data' must be a pandas DataFrame")
        if price_column not in data.columns:
            raise ValueError(f"Column '{price_column}' not found in the DataFrame")
        if data[price_column].isnull().any():
            raise ValueError(f"Column '{price_column}' contains NaN values")

        self.data = data
        self.price_column = price_column
        self.prices = data[price_column].values

    def assess_noise(self):
        """
        Assess the degree of noise in the data through visualization.
        Plots daily returns and their distribution to quantify noise.
        """
        # Calculate daily returns
        returns = pd.Series(self.prices).pct_change().dropna()
        
        # Plot daily returns and distribution
        plt.figure(figsize=MA_PLOT_FIGSIZE)
        
        # Daily returns over time
        plt.subplot(1, 2, 1)
        plt.plot(returns.index, returns, 'b', label='Daily Returns')
        plt.title('Daily Returns (Showing Volatility)')
        plt.xlabel('Index')
        plt.ylabel('Daily Return')
        plt.legend()
        
        # Distribution of daily returns
        plt.subplot(1, 2, 2)
        plt.hist(returns, bins=RETURNS_HIST_BINS, color='r', alpha=0.7)
        plt.title('Distribution of Daily Returns')
        plt.xlabel('Daily Return')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        # Calculate and print noise metric (standard deviation of returns)
        noise_level = np.std(returns)
        print(f"Noise Level (Std of Daily Returns): {noise_level:.4f}")

    def apply_moving_average(self, window_size=DEFAULT_MA_WINDOW_SIZE):
        """
        Apply moving averages (SMA and EMA) to the data.

        Parameters:
        - window_size (int): Window size for moving averages.

        Returns:
        - dict: Dictionary of moving average results with method names as keys.
        """
        ma_results = {}
        
        # Simple Moving Average (SMA)
        ma_results[f'SMA_{window_size}'] = pd.Series(
            self.data[self.price_column]
            .rolling(window=window_size, center=True)
            .mean(),
            index=self.data.index
        )
        
        # Exponential Moving Average (EMA)
        ma_results[f'EMA_{window_size}'] = pd.Series(
            self.data[self.price_column]
            .ewm(span=window_size, adjust=False)
            .mean(),
            index=self.data.index
        )
            
        return ma_results
    
    def apply_frequency_filter(self, cutoff_freq=DEFAULT_CUTOFF_FREQ, filter_type='lowpass'):
        """
        Apply frequency domain filtering using FFT.

        Parameters:
        - cutoff_freq (float): Cutoff frequency for the filter.
        - filter_type (str): Type of filter ('lowpass' or 'highpass').

        Returns:
        - pd.Series: Filtered signal.
        """
        # Get FFT of the signal
        fft_vals = fft(self.prices)
        freqs = np.fft.fftfreq(len(self.prices))
        
        # Create filter mask
        if filter_type == 'lowpass':
            mask = np.abs(freqs) <= cutoff_freq
        else:  # highpass
            mask = np.abs(freqs) >= cutoff_freq
            
        # Apply filter
        fft_filtered = fft_vals * mask
        filtered_signal = np.real(ifft(fft_filtered))
        
        return pd.Series(filtered_signal, index=self.data.index)
    
    def plot_preprocessing_results(self):
        """
        Plot original data, preprocessing results, and residuals.
        - Moving averages (10-day SMA and EMA) in one plot.
        - Frequency domain filtering (original, lowpass, and highpass) in one plot.
        - Residuals for all methods in a separate plot.

        Returns:
        - ma_results (dict): Moving average results.
        - freq_filtered_low (pd.Series): Lowpass filtered data.
        - freq_filtered_high (pd.Series): Highpass filtered data.
        """
        # Apply preprocessing methods
        ma_results = self.apply_moving_average(window_size=DEFAULT_MA_WINDOW_SIZE)
        freq_filtered_low = self.apply_frequency_filter(cutoff_freq=DEFAULT_CUTOFF_FREQ, filter_type='lowpass')
        freq_filtered_high = self.apply_frequency_filter(cutoff_freq=DEFAULT_CUTOFF_FREQ, filter_type='highpass')
        
        # Create plots
        fig, axes = plt.subplots(3, 1, figsize=PLOT_FIGSIZE)
        
        # Plot 1: Moving Averages (10-day SMA and EMA)
        axes[0].plot(self.data.index, self.prices, 'r', label='Original', alpha=0.5)
        for name, series in ma_results.items():
            axes[0].plot(self.data.index, series, label=name)
        axes[0].set_title('10-Day Moving Averages')
        axes[0].legend()
        
        # Plot 2: Frequency Domain Filtering (Original, Lowpass, and Highpass in one plot)
        axes[1].plot(self.data.index, self.prices, 'r', label='Original', alpha=0.5)
        axes[1].plot(self.data.index, freq_filtered_low, 'g', label='Lowpass Filtered')
        axes[1].plot(self.data.index, freq_filtered_high, 'b', label='Highpass Filtered')
        axes[1].set_title('Frequency Domain Filtering (Lowpass and Highpass)')
        axes[1].legend()
        
        # Plot 3: Residuals
        for name, series in ma_results.items():
            residuals = self.prices - series
            axes[2].plot(self.data.index, residuals, label=f'Residuals ({name})', alpha=0.5)
        residuals_low = self.prices - freq_filtered_low
        residuals_high = self.prices - freq_filtered_high
        axes[2].plot(self.data.index, residuals_low, label='Residuals (Lowpass)', alpha=0.5)
        axes[2].plot(self.data.index, residuals_high, label='Residuals (Highpass)', alpha=0.5)
        axes[2].set_title('Residuals Over Time')
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
        
        return ma_results, freq_filtered_low, freq_filtered_high
    
    def get_preprocessing_stats(self, filtered_data):
        """
        Calculate statistics to evaluate preprocessing effectiveness.

        Parameters:
        - filtered_data (pd.Series or np.ndarray): Preprocessed data.

        Returns:
        - dict: Statistics including standard deviation, noise reduction, MAE, and SNR.
        """
        original_std = np.std(self.prices)
        filtered_std = np.std(filtered_data)
        noise_reduction = (original_std - filtered_std) / original_std * 100
        mae = np.mean(np.abs(self.prices - filtered_data))
        # Avoid division by zero in SNR calculation
        noise_power = np.mean((self.prices - filtered_data) ** 2)
        snr = 10 * np.log10(np.mean(self.prices ** 2) / (noise_power if noise_power > 0 else SNR_EPSILON))
        
        return {
            'Original Std': original_std,
            'Filtered Std': filtered_std,
            'Noise Reduction %': noise_reduction,
            'MAE': mae,
            'SNR (dB)': snr
        }

def preprocess_stock_data(df, price_column=DEFAULT_PRICE_COLUMN):
    """
    Main function to preprocess stock data.

    Parameters:
    - df (pd.DataFrame): DataFrame containing stock price data.
    - price_column (str): Name of the column containing price data.

    Returns:
    - ma_results (dict): Moving average results.
    - freq_filtered_low (pd.Series): Lowpass filtered data.
    - freq_filtered_high (pd.Series): Highpass filtered data.
    """
    preprocessor = StockPreprocessor(df, price_column)
    
    # Assess noise before preprocessing
    preprocessor.assess_noise()
    
    # Apply preprocessing and plot results
    ma_results, freq_filtered_low, freq_filtered_high = preprocessor.plot_preprocessing_results()
    
    # Print statistics for each method
    print("\nPreprocessing Statistics:")
    print("\nMoving Averages:")
    for name, series in ma_results.items():
        stats = preprocessor.get_preprocessing_stats(series)
        print(f"\n{name}:")
        for stat_name, value in stats.items():
            print(f"{stat_name}: {value:.4f}")
    
    print("\nLowpass Frequency Filtering:")
    freq_stats_low = preprocessor.get_preprocessing_stats(freq_filtered_low)
    for stat_name, value in freq_stats_low.items():
        print(f"{stat_name}: {value:.4f}")
        
    print("\nHighpass Frequency Filtering:")
    freq_stats_high = preprocessor.get_preprocessing_stats(freq_filtered_high)
    for stat_name, value in freq_stats_high.items():
        print(f"{stat_name}: {value:.4f}")
    
    return ma_results, freq_filtered_low, freq_filtered_high

if __name__ == "__main__":
    df = pd.read_csv(STOCK_DATA_PATH)
    preprocess_stock_data(df)
