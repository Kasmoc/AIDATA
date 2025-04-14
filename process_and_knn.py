import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

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

        # Generate new samples from the estimated density
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

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def knn_classifier(df, k=3):
    # Remove rows where Diabetes is NaN
    df = df.dropna(subset=['Diabetes'])
    # Separate features and target
    X = df.drop(columns=['Student ID', 'Age', 'Gender', 'Diabetes', 'Blood Type', 'Smoking'])
    y = df['Diabetes']
    
    # Split the data (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to numpy arrays for faster computation
    X_train_array = X_train.values
    X_test_array = X_test.values
    y_train_array = y_train.values

    predictions = []
    
    # Vectorized distance calculation using NumPy
    for i in range(len(X_test_array)):
        # Create a broadcasted subtraction of the current test point from all training points
        # This creates a matrix of differences
        differences = X_train_array - X_test_array[i]
        
        # Square the differences
        squared_differences = differences ** 2
        
        # Sum the squared differences along the feature axis (axis=1)
        sum_squared_differences = np.sum(squared_differences, axis=1)
        
        # Take the square root to get Euclidean distances
        distances = np.sqrt(sum_squared_differences)
        
        # Get indices of k smallest distances
        nearest_indices = np.argsort(distances)[:k]
        
        # Get corresponding labels
        k_nearest_labels = y_train_array[nearest_indices]
        
        # Majority vote using Counter
        vote = Counter(k_nearest_labels)
        predicted_label = vote.most_common(1)[0][0]
        predictions.append(predicted_label)

    # Evaluate performance using scikit-learn metrics
    cm = confusion_matrix(y_test.tolist(), predictions)
    acc = accuracy_score(y_test.tolist(), predictions)
    report = classification_report(y_test.tolist(), predictions)
    
    # Print evaluation results
    print("Confusion Matrix:")
    print(cm)
    print("\nAccuracy: {:.2f}%".format(acc * 100))
    print("\nClassification Report:")
    print(report)
    
    # Return predictions and accuracy for potential further use
    return predictions, acc

def main(input_path, output_path, sampling_columns, kde_columns, save_output=False):
    # Load and process data
    df = Load_and_clean_data(input_path)
    
    # Visualize raw data
    visualize_data(df, "Raw Data")

    # knn before imputation
    #knn_classifier(df)

    # Transform and impute missing values
    df = impute_values(df, sampling_columns, kde_columns)
    # Visualize processed data
    visualize_data(df, "Processed Data")

    # knn after imputation
    #knn_classifier(df)
    
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
sampling_columns= ['Heart Rate', 'Age', 'Height', 'Weight', 'Blood Pressure', 'Cholesterol']
,
kde_columns= ['BMI', 'Temperature']
,
save_output=True
)