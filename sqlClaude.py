import sqlite3
import csv
import os

# Create connection to databases
conn_students = sqlite3.connect('students.db')
conn_health = sqlite3.connect('health_records.db')

# Create cursors
c_students = conn_students.cursor()
c_health = conn_health.cursor()

# Create tables
c_students.execute('''
CREATE TABLE IF NOT EXISTS students (
    student_id INTEGER PRIMARY KEY,
    age INTEGER,
    gender TEXT,
    height REAL,
    weight REAL
)
''')

c_health.execute('''
CREATE TABLE IF NOT EXISTS health_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    blood_type TEXT,
    bmi REAL,
    temperature REAL,
    heart_rate INTEGER,
    blood_pressure TEXT,
    cholesterol REAL,
    diabetes BOOLEAN,
    smoking BOOLEAN,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
)
''')

def safe_int(value):
    """Convert a value to integer safely, handling floats and strings."""
    try:
        # If it's a float string like '1.0', first convert to float then to int
        return int(float(value))
    except (ValueError, TypeError):
        print(f"Warning: Could not convert '{value}' to integer. Using 0 instead.")
        return 0

def safe_float(value):
    """Convert a value to float safely."""
    try:
        return float(value)
    except (ValueError, TypeError):
        print(f"Warning: Could not convert '{value}' to float. Using 0.0 instead.")
        return 0.0

def import_from_csv(csv_file_path):
    if not os.path.exists(csv_file_path):
        print(f"Error: File {csv_file_path} not found.")
        return False
    
    try:
        # Import data from CSV file
        with open(csv_file_path, 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            
            # Check if we have rows in the CSV
            row_count = 0
            
            for row in csvreader:
                row_count += 1
                
                # Get values with safe conversion
                student_id = safe_int(row['Student ID'])
                age = safe_int(row['Age'])
                gender = row['Gender']
                height = safe_float(row['Height'])
                weight = safe_float(row['Weight'])
                
                # Insert data into students table
                c_students.execute('''
                INSERT OR IGNORE INTO students 
                (student_id, age, gender, height, weight)
                VALUES (?, ?, ?, ?, ?)
                ''', (student_id, age, gender, height, weight))
                
                # Get health values with safe conversion
                blood_type = row['Blood Type']
                bmi = safe_float(row['BMI'])
                temperature = safe_float(row['Temperature'])
                heart_rate = safe_int(row['Heart Rate'])
                blood_pressure = row['Blood Pressure']
                cholesterol = safe_float(row['Cholesterol'])
                
                # Handle boolean values
                diabetes = 0
                if 'Diabetes' in row:
                    diabetes_val = row['Diabetes'].lower()
                    diabetes = 1 if diabetes_val in ('true', 'yes', '1', 't', 'y') else 0
                
                smoking = 0
                if 'Smoking' in row:
                    smoking_val = row['Smoking'].lower()
                    smoking = 1 if smoking_val in ('true', 'yes', '1', 't', 'y') else 0
                
                # Insert data into health_records table
                c_health.execute('''
                INSERT INTO health_records 
                (student_id, blood_type, bmi, temperature, heart_rate, 
                blood_pressure, cholesterol, diabetes, smoking)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    student_id, blood_type, bmi, temperature, heart_rate,
                    blood_pressure, cholesterol, diabetes, smoking
                ))
            
            if row_count == 0:
                print(f"Warning: No data found in {csv_file_path}")
                return False
                
            print(f"Processed {row_count} rows from CSV file.")
            return True
    except Exception as e:
        print(f"Error during import: {str(e)}")
        return False

# File path for the CSV file
csv_file = 'processed_medical_data.csv'

# Import data from CSV
success = import_from_csv(csv_file)

# Commit changes and close connections
if success:
    conn_students.commit()
    conn_health.commit()
    print(f"Data from {csv_file} has been imported into the databases successfully.")
    # Verify by counting rows
    c_students.execute("SELECT COUNT(*) FROM students")
    student_count = c_students.fetchone()[0]
    c_health.execute("SELECT COUNT(*) FROM health_records")
    health_count = c_health.fetchone()[0]
    print(f"Imported {student_count} student records and {health_count} health records.")
else:
    print("Import failed. No changes were committed to the databases.")

conn_students.close()
conn_health.close()