import pandas as pd
import sqlite3

# Create connection to databases
connect_students = sqlite3.connect('students.db')
connect_health = sqlite3.connect('health_records.db')

# Create cursors
c_students = connect_students.cursor()
c_health = connect_health.cursor()

# Create students table
c_students.execute('''
CREATE TABLE IF NOT EXISTS students (
    [Student ID] INTEGER PRIMARY KEY,
    Age INTEGER,
    Gender TEXT,
    Height INTEGER,
    Weight INTEGER
)
''')

# Create health_records table
c_health.execute('''
CREATE TABLE IF NOT EXISTS health_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    [Student ID] INTEGER,
    [Blood Type] TEXT,
    BMI INTEGER,
    Temperature INTEGER,
    [Heart Rate] INTEGER,
    [Blood Pressure] INTEGER,
    Cholesterol INTEGER,
    Diabetes TEXT,
    Smoking TEXT,
    FOREIGN KEY ([Student ID]) REFERENCES students([Student ID])
)
''')

# Clean floats into integers
df = pd.read_csv('processed_medical_data.csv')
df['Height'] = df['Height'].astype(int)
df['Weight'] = df['Weight'].astype(int)
df['BMI'] = df['BMI'].astype(int)
df['Temperature'] = df['Temperature'].astype(int)

# Create separate dataframes for each table
students_df = df[['Student ID', 'Age', 'Gender', 'Height', 'Weight']].copy()
health_df = df[['Student ID', 'Blood Type', 'BMI', 'Temperature', 
                'Heart Rate', 'Blood Pressure', 'Cholesterol', 
                'Diabetes', 'Smoking']].copy()

# Import data into the respective tables
students_df.to_sql('students', connect_students, if_exists='replace')
health_df.to_sql('health_records', connect_health, if_exists='replace')

# Commit changes
connect_students.commit()
connect_health.commit()

