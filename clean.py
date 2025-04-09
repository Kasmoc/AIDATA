import pandas as pd
import numpy as np 
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt


file_path = "C:\\Users\\tonny\\OneDrive\\Desktop\\DAKI2 - Repo\\AI & Data\\Miniprojekt\\medical_students_dataset.csv"

raw_df = pd.read_csv(file_path)

#Imputation for remaining collumns
data = pd.get_dummies(df, columns=['Gender', 'Blood Type', 'Diabetes', 'Smoking'])

imputer = SimpleImputer(strategy='median')
data[['BMI', 'Temperature']] = imputer.fit_transform(data[['BMI', 'Temperature']])

