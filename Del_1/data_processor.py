import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def Load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(how='all')
    df = df.sort_values(by='Student ID')
    df = df.drop_duplicates(subset='Student ID', keep='first')
    df['Student ID'] = df['Student ID'].interpolate(limit_direction='both').astype(float)
    return df

def impute_values(df, sampling_cols, kde_cols):
    # Sampling imputation
    for col in sampling_cols:

        # Get observed non-null values
        non_null_vals = df[col].dropna().values
        n_missing = df[col].isnull().sum()

        # Randomly sample observed values to impute missing entries
        sampled_values = np.random.choice(non_null_vals, size=n_missing, replace=True)
        df.loc[df[col].isnull(), col] = sampled_values
    
    # KDE imputation
    for col in kde_cols:

        # Get observed non-null values
        non_null_vals = df[col].dropna().values
        n_missing = df[col].isnull().sum()

        # Build Gaussian KDE based on the non-null values
        kde = gaussian_kde(non_null_vals)

        # Generate new samples from the estimated density; note resample returns a 2D array.
        kde_samples = kde.resample(n_missing)[0]
        df.loc[df[col].isnull(), col] = kde_samples
    
    return df

def visualize_data(df, title):
    # Plot histograms of numeric features
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df.hist(figsize=(10, 8), bins=30, edgecolor="black")
    plt.suptitle(f"{title} Distributions")
    plt.tight_layout()
    plt.show()

def knn(df):
    # Placeholder for KNN imputation function
    pass

def evaluate_model(model):
    # Placeholder for model evaluation function
    pass

def main(input_path, output_path, categorical_columns, sampling_columns, kde_columns, save_output=False):
    # Load and process data
    df = Load_and_clean_data(input_path)
    
    # Visualize raw data
    visualize_data(df, "Raw Data")

    # knn before imputation
    evaluate_model(knn(df))

    # Transform and impute
    df = pd.get_dummies(df, categorical_columns, dummy_na=False)
    df = impute_values(df, sampling_columns, kde_columns)
    
    # Visualize processed data
    visualize_data(df, "Processed Data")

    # knn after imputation
    evaluate_model(knn(df))
    
    if save_output: 
        # Save processed data
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")


# Execute function
main(
input_path= "medical_students_dataset.csv"
,
output_path= "processed_medical_data.csv"
,
categorical_columns= ['Gender', 'Blood Type', 'Diabetes', 'Smoking']
,
sampling_columns= ['Heart Rate', 'Age', 'Height', 'Weight', 'Blood Pressure', 'Cholesterol']
,
kde_columns= ['BMI', 'Temperature']
,
save_output=False
)