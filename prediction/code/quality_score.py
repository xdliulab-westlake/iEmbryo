import os
import sqlite3
import pandas as pd
from pathlib import Path

def process_fdb_file(fdb_file):
    try:
        # Connect to the .fdb database file in read-only mode
        conn = sqlite3.connect(f"file:{fdb_file}?mode=ro", uri=True)
        cur = conn.cursor()

        # Retrieve all table names in the database (optional debug info)
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cur.fetchall()
        print(f"📦 Tables in {fdb_file}: {[t[0] for t in tables]}")

        # Read data from the main table (change 'WELL' if needed)
        table_name = 'WELL'
        cur.execute(f"SELECT * FROM {table_name}")
        column_names = [desc[0] for desc in cur.description]
        records = cur.fetchall()
        df = pd.DataFrame(records, columns=column_names)

        print(f"📄 Columns in table '{table_name}': {column_names}")

        # Check if all required columns are present
        required_cols = ['Type', 'Par', 'Well', 'Time', 'Val']
        if all(col in df.columns for col in required_cols):
            # Filter rows for TE and ICM scores
            df = df[df['Par'].isin(['TE', 'ICM'])].copy()
            df['Event'] = df['Par']
            # Create a unique row identifier from ID and Well
            df['RowName'] = df['ID'].astype(str) + '_' + df['Well'].astype(str)
            # Pivot to form a matrix with TE and ICM scores
            matrix = df.pivot_table(index='RowName', columns='Event', values='Val', aggfunc='first')

            # Add a 'Quality' column based on TE and ICM combination
            def classify_quality(row):
                icm = str(row.get('ICM', '')).strip()
                te = str(row.get('TE', '')).strip()
                score = icm + te
                good_set = {"AA", "AB", "BA"}
                return "good" if score in good_set else "fair"

            matrix['Quality'] = matrix.apply(classify_quality, axis=1)

            # Add a column with source filename
            matrix['SourceFile'] = Path(fdb_file).name

            return matrix

        else:
            print(f"⚠️ Missing required columns in {table_name}, skipping matrix creation.")
            return None

    except Exception as e:
        print(f"❌ Error processing {fdb_file}: {e}")
        return None

    finally:
        cur.close()
        conn.close()

if __name__ == '__main__':
    # Explicitly list only the fdb files to process
    fdb_files = [
        Path("/Users/lzz/Downloads/MainDB_5068.fdb"),
        Path("/Users/lzz/Downloads/MainDB_5069.fdb"),
        Path("/Users/lzz/Downloads/MainDB_4424.fdb"),
        Path("/Users/lzz/Downloads/MainDB_4425.fdb"),
    ]

    all_matrices = []

    for fdb_file in fdb_files:
        matrix = process_fdb_file(fdb_file)
        if matrix is not None:
            all_matrices.append(matrix)

    # Merge and save combined results
    if all_matrices:
        merged = pd.concat(all_matrices)
        merged.to_csv("./MainDB.csv")
        print("✅ All matrices merged and saved as 'MainDB.csv'")
    else:
        print("⚠️ No valid matrices were created.")

    print("-------------- All .fdb Files Processed --------------")