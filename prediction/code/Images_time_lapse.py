import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

# List of base directories to search for .pdb files
base_dirs = ['/Users/lzz/Downloads']
results = []

# Convert Excel-style serial date (days since 1899-12-30) to Python datetime object
def days_to_datetime(days):
    return datetime(1899, 12, 30) + timedelta(days=float(days))

# Iterate over each base directory
for base_dir in base_dirs:
    # Loop through all files in the directory
    for filename in os.listdir(base_dir):
        # Skip files that do not have a .pdb extension
        if not filename.endswith('.pdb'):
            continue

        pdb_path = os.path.join(base_dir, filename)
        conn = sqlite3.connect(pdb_path)

        try:
            # Read the GENERAL table to extract fertilization time
            general_df = pd.read_sql_query("SELECT * FROM GENERAL", conn)
            # Filter rows where 'Par' column is 'Fertilization'
            fert_row = general_df[general_df['Par'] == 'Fertilization']
            # If no fertilization time found, skip this file
            if fert_row.empty:
                continue
            # Extract fertilization time value and convert to datetime
            fert_time = fert_row.iloc[0]['Val']
            fert_datetime = days_to_datetime(fert_time)

            # Read the IMAGES table containing image capture times
            images_df = pd.read_sql_query("SELECT * FROM IMAGES", conn)

            # Iterate over each image record to calculate time since fertilization
            for _, row in images_df.iterrows():
                well = int(row['Well'])
                run = int(row['Run'])
                # Convert image time from Excel-style serial to datetime
                image_time = days_to_datetime(row['Time'])
                # Calculate time difference in hours between image and fertilization
                delta_hours = round((image_time - fert_datetime).total_seconds() / 3600, 1)
                # Skip if delta_hours is outside the range [96, 112]
                if not (96 <= delta_hours <= 112):
                    continue
                file_base = os.path.splitext(filename)[0]
                # Extract patient ID (everything before "_P_") to avoid redundant entries
                patient_id = file_base.split("_P_")[0]
                # Avoid adding duplicate patient-time entries
                results.append([patient_id, run, delta_hours])

        except Exception as e:
            # Print error message if processing this .pdb file fails
            print(f"Failed to process {pdb_path}: {e}")
        finally:
            # Ensure the database connection is closed
            conn.close()

# Convert results list to a DataFrame
results_df = pd.DataFrame(results, columns=["PatientID", "Run", "Time_PostFertilization_Hours"])
# Remove duplicate entries if any
results_df = results_df.drop_duplicates()
# Save the DataFrame to a CSV file
results_df.to_csv("Embryo_Times_PostFertilization_Hours.csv", index=False)