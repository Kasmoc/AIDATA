import pandas as pd
import numpy as np 
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Load dataset
data = pd.read_csv("medical_students_dataset.csv")

categorical_columns = ['Gender', 'Blood Type', 'Diabetes', 'Smoking']

# One-hot-encode categorical features
data_encoded = pd.get_dummies(data, columns=categorical_columns)

# Plot histograms for all numerical features (before processing)
numeric_columns = data_encoded.select_dtypes(include=['number']).columns
data_encoded[numeric_columns].hist(figsize=(12, 10), bins=30, edgecolor="black")
plt.suptitle("Data Distribution Before Processing")
plt.tight_layout()
plt.show()

# DUPLICATES HANDLING - Keep duplicate rows with least NaN values
# Count missing values per row
data_encoded['missing_count'] = data_encoded.isna().sum(axis=1)

# Print duplicates before removal
duplicate_ids = data_encoded['Student ID'].duplicated(keep=False)
print(f"\nDuplicates found before processing: {duplicate_ids.sum()}")
if duplicate_ids.sum() > 0:
    print(data_encoded[duplicate_ids].sort_values('Student ID'))

# Keep the row with the fewest missing values for each Student ID
data_encoded = data_encoded.loc[data_encoded.groupby('Student ID')['missing_count'].idxmin()]

# Drop helper column
data_encoded = data_encoded.drop(columns='missing_count')

# Sort by Student ID and reset index
data_encoded = data_encoded.sort_values('Student ID').reset_index(drop=True)

# Use interpolation for Student ID (if needed)
# Only run this if we're expecting a continuous range of student IDs
data_encoded['Student ID'] = data_encoded['Student ID'].interpolate(limit_direction='both').astype(float)

# Check for duplicates after processing
duplicate_ids = data_encoded['Student ID'].duplicated(keep=False)
print(f"\nDuplicates found after processing: {duplicate_ids.sum()}")
if duplicate_ids.sum() > 0:
    print(data_encoded[duplicate_ids].sort_values('Student ID'))

# Define columns for different imputation methods
sampling_columns = ['Heart Rate', 'Age', 'Height', 'Weight', 'Blood Pressure', 'Cholesterol']
kde_columns = ['BMI', 'Temperature']

# Function for sampling imputation
def sampling_imputation(df, columns):
    """
    Perform sampling imputation on specified columns.
    For each column with missing values, randomly sample from the observed values.
    """
    df_imputed = df.copy()
    
    for column in columns:
        # Check if column has missing values
        if df_imputed[column].isnull().sum() > 0:
            # Get the observed (non-missing) values
            observed_values = df_imputed[column].dropna().values
            
            # Get indices of missing values
            missing_indices = df_imputed[column].isnull()
            
            # Number of missing values to impute
            n_missing = missing_indices.sum()
            
            # Randomly sample from observed values (with replacement)
            sampled_values = np.random.choice(observed_values, size=n_missing, replace=True)
            
            # Assign sampled values to missing positions
            df_imputed.loc[missing_indices, column] = sampled_values
    
    return df_imputed

# Function for KDE imputation
def kde_imputation(df, columns):
    """
    Perform KDE (Kernel Density Estimation) imputation on specified columns.
    """
    df_imputed = df.copy()
    
    for column in columns:
        # Check if column has missing values
        if df_imputed[column].isnull().sum() > 0:
            # Get the observed (non-missing) values
            observed_values = df_imputed[column].dropna().values
            
            # Get indices of missing values
            missing_indices = df_imputed[column].isnull()
            
            # Number of missing values to impute
            n_missing = missing_indices.sum()
            
            if n_missing > 0:
                # Create KDE from observed data
                kde = gaussian_kde(observed_values)
                
                # Sample from the KDE
                kde_samples = kde.resample(n_missing)[0]
                
                # Assign KDE-sampled values to missing positions
                df_imputed.loc[missing_indices, column] = kde_samples
    
    return df_imputed

# First, apply sampling imputation
data_encoded = sampling_imputation(data_encoded, sampling_columns)

# Then, apply KDE imputation
data_encoded = kde_imputation(data_encoded, kde_columns)

# Check remaining missing values
print("\nRemaining missing values after imputation:")
print(data_encoded.isna().sum())

# Plot histograms for all numerical features (after processing)
numeric_columns = data_encoded.select_dtypes(include=['number']).columns
data_encoded[numeric_columns].hist(figsize=(12, 10), bins=30, edgecolor="black")
plt.suptitle("Data Distribution After Processing")
plt.tight_layout()
plt.show()

# Save the processed dataset
output_path = r"C:\Users\anne\Desktop\Daki\s2\ai_and_data\miniprojekt\data\processed_medical_data.csv"
data_encoded.to_csv(output_path, index=False)
print(f"\nProcessed data saved to: {output_path}")
plt.show()