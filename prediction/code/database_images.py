import sqlite3
import pandas as pd
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


def save_image(file_path, image_data):
    """Save the image data to a file in JPEG format."""
    with open(file_path, 'wb') as f:
        f.write(image_data)


def create_directories(df, db_filename, output_dir):
    """Pre-create all necessary directories for the given DataFrame."""
    unique_combinations = df[['Well', 'Focal']].drop_duplicates().values
    for well, focal in unique_combinations:
        top_level_dir_name = f"{db_filename}_{well}"
        top_level_dir = os.path.join(output_dir, top_level_dir_name, "images")
        if not os.path.exists(top_level_dir):
            os.makedirs(top_level_dir)

        focal_dir = os.path.join(top_level_dir, f"F{focal}")
        if not os.path.exists(focal_dir):
            os.makedirs(focal_dir)


def process_record(row, db_filename, output_dir):
    """Process a single record from the DataFrame."""
    well_value = str(row['Well'])
    top_level_dir_name = f"{db_filename}_{well_value}"
    top_level_dir = os.path.join(output_dir, top_level_dir_name, "images")
    focal_dir = os.path.join(top_level_dir, f"F{row['Focal']}")

    file_name = f"{db_filename}_{row['File']}.jpeg"
    file_path = os.path.join(focal_dir, file_name)

    try:
        save_image(file_path, row['Image'])
        return True
    except Exception as e:
        print(f"Error saving {file_path}: {e}")
        return False


def process_database(db_file, output_dir, max_workers=10):
    """Process a single database file and save images according to the specified structure."""
    try:
        # Connect to the SQLite database in read-only mode
        conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
        cur = conn.cursor()

        # Extract the base name of the database file (without extension)
        db_filename = Path(db_file).stem

        # Execute query and fetch column names
        cur.execute("SELECT * FROM IMAGES")
        column_names = [description[0] for description in cur.description]

        # Fetch all records and convert to DataFrame
        records = cur.fetchall()
        df = pd.DataFrame(records, columns=column_names)
        df['File'] = df.apply(lambda row: 'WELL%s_RUN%s' % (row['Well'], row['Run']), axis=1)

        # Create directories
        create_directories(df, db_filename, output_dir)

        # Define the number of workers based on CPU count or provided max_workers
        if max_workers is None:
            # For I/O-bound tasks, you can set max_workers higher than CPU count
            max_workers = min(32, multiprocessing.cpu_count() + 4)  # A common heuristic

        # Use ThreadPoolExecutor to process records concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_record, row, db_filename, output_dir): row for _, row in df.iterrows()}
            for future in as_completed(futures):
                future.result()

        print(f"Execution Completed for {db_file} ----- Total Count: {len(df)}")

    except Exception as e:
        print(f"Error processing {db_file}: {e}")
    finally:
        # Close the database connection
        cur.close()
        conn.close()

if __name__ == '__main__':
    # Define the directory containing the .pdb files
    db_root_dir = "./ScopeData/2024/"  # Replace with your actual root directory path

    # Define the output directory as the current working directory
    output_dir = os.getcwd()

    # Ensure the output directory is writable
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"The output directory {output_dir} is not writable.")

    # Search for all .pdb files in the db_root_dir and its subdirectories
    pdb_files = list(Path(db_root_dir).rglob("*.pdb"))

    # Optionally define the maximum number of workers (threads)
    max_workers = None  # Set this to an integer value if you want to specify the number of threads 

    # Process each .pdb file
    for db_file in pdb_files:
        process_database(db_file, output_dir, max_workers=max_workers)

    print("---------------- All Databases Processed Successfully ----------------")