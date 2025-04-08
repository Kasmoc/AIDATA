import pandas as pd
from stock_preprocessor import preprocess_stock_data
import config

def main():    
    # Load and preprocess the data
    df = pd.read_csv(config.STOCK_DATA_PATH)
    
    # Process with default configuration
    ma_results, freq_filtered_low, freq_filtered_high = preprocess_stock_data(df)

if __name__ == "__main__":
    main() 