import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv('medical_students_dataset.csv')

# Select numerical columns only
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Create correlation matrix
correlation_matrix = df[numerical_columns].corr()

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

# Save the plot
plt.savefig('correlation_matrix.png')
plt.close()

print("Correlation matrix has been created and saved as 'correlation_matrix.png'") 