import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

file_path = "medical_students_dataset.csv"

raw_df = pd.read_csv(file_path)

# Save a copy of the raw data for comparison
raw_df_copy = raw_df.copy()

# Count missing values per row
raw_df['missing_count'] = raw_df.isna().sum(axis=1)

# Keep the row with the fewest missing values for each Student ID
df = raw_df.loc[raw_df.groupby('Student ID')['missing_count'].idxmin()]

# Drop helper column
df = df.drop(columns='missing_count')

# Sort by Student ID and remove empty rows
df = df.sort_values('Student ID').reset_index(drop=True)

# Use interpolation to add missing Student ID's
df['Student ID'] = df['Student ID'].interpolate(limit_direction='both').astype(float)

# Save a copy of data before imputation
df_before_imputation = df.copy()

# Perform single imputation pass with appropriate strategy per column
cols_for_mean_imp = ['BMI', 'Temperature']
cols_for_samling_imp = ['Heart Rate', 'Age', 'Height', 'Weight','Cholesterol', 'Gender', 'Blood Type', 'Diabetes', 'Smoking', 'Blood Pressure']

# Impute numeric columns with median
for col in cols_for_mean_imp:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# Impute categorical columns by sampling from observed values
for col in cols_for_samling_imp:
    if df[col].isnull().any():
        observed_values = df[col].dropna().values
        missing_indices = df[col].isnull()
        sampled_values = np.random.choice(observed_values, size=missing_indices.sum(), replace=True)
        df.loc[missing_indices, col] = sampled_values

# Save a copy of data after imputation
df_after_imputation = df.copy()