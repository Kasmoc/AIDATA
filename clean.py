import pandas as pd
import numpy as np 
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "medical_students_dataset.csv"

raw_df = pd.read_csv(file_path)

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

#Imputation for remaining collumns
df = pd.get_dummies(df, columns=['Gender', 'Blood Type', 'Diabetes', 'Smoking'])

imputer = SimpleImputer(strategy='median')
df[['BMI', 'Temperature']] = imputer.fit_transform(df[['BMI', 'Temperature']])

# Function to revert dummy variables back to a single categorical variable
def revert_dummies(df, feature_name):
    # Identify the dummy columns for this feature
    dummy_cols = [col for col in df.columns if col.startswith(feature_name + '_')]
    
    # Use idxmax to determine which dummy is active for each observation
    # This returns the column name (e.g., 'Gender_Male')
    df[feature_name] = df[dummy_cols].idxmax(axis=1).str.replace(feature_name + '_', '')
    
    # Optionally, drop the dummy columns
    df.drop(columns=dummy_cols, inplace=True)
    return df

# Revert each set of dummy variables
for feature in ['Gender', 'Blood Type', 'Diabetes', 'Smoking']:
    df = revert_dummies(df, feature)

def sampling_imputation(data, columns_to_impute):
    """
    Perform sampling imputation on a dataframe.
    For each column with missing values, randomly sample from the observed values
    in that column to fill in the missing values.
    """

    data_imputed = data[columns_to_impute].copy()
    
    for column in data_imputed.columns:
        # Check if column has missing values
        if data_imputed[column].isnull().sum() > 0:
            # Get the observed (non-missing) values
            observed_values = data_imputed[column].dropna().values
            
            # Get indices of missing values
            missing_indices = data_imputed[column].isnull()
            
            # Number of missing values to impute
            n_missing = missing_indices.sum()
            
            # Randomly sample from observed values (with replacement)
            sampled_values = np.random.choice(observed_values, size=n_missing, replace=True)
            
            # Assign sampled values to missing positions
            data_imputed.loc[missing_indices, column] = sampled_values
    
    return data_imputed

# Specify the columns to impute
columns_to_impute = ['Heart Rate', 'Age', 'Gender', 'Height', 'Weight', 
                     'Blood Type', 'Blood Pressure', 'Cholesterol', 'Diabetes', 'Smoking']

# Perform sampling imputation for these columns
imputed_data = sampling_imputation(df, columns_to_impute)

# Now, assign the imputed columns back using a list of column names
df[columns_to_impute] = imputed_data[columns_to_impute]


