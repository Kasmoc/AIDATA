import pandas as pd
import numpy as np 
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt


file_path = "C:\\Users\\tonny\\OneDrive\\Desktop\\DAKI2 - Repo\\AI & Data\\Miniprojekt\\medical_students_dataset.csv"

raw_df = pd.read_csv(file_path)

# Select numerical columns only
numerical_columns = raw_df.select_dtypes(include=['int64', 'float64']).columns

# Create correlation matrix
correlation_matrix = raw_df[numerical_columns].corr()

# Create a figure with a larger size
plt.figure(figsize=(12, 10))

# Create heatmap
sns.heatmap(correlation_matrix, 
            annot=True,  # Show correlation values
            cmap='coolwarm',  # Color scheme
            center=0,  # Center the colormap at 0
            fmt='.2f',  # Round to 2 decimal places
            square=True)  # Make the plot square

# Customize the plot
plt.title('Correlation Matrix of Medical Students Dataset Features')
plt.tight_layout()  # Adjust layout to prevent label cutoff