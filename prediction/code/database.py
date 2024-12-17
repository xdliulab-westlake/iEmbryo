import sqlite3
import pandas as pd
import os
from pathlib import Path


def save_image(file_path, image_data):
    """Save the image data to a file in JPEG format."""
    with open(file_path, 'wb') as f:
        f.write(image_data)


if __name__ == '__main__':
    # Connect to the SQLite database
    db_file = "/Users/lzz/Documents/GitHub/IVF_AI/database/image.pdb"
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    
    # Extract the base name of the database file (without extension)
    db_filename = Path(db_file).stem
    
    # Execute query and fetch column names
    cur.execute("SELECT * FROM IMAGES")
    column_names = [tuple[0] for tuple in cur.description]
    
    # Fetch all records and convert to DataFrame
    records = cur.fetchall()
    df = pd.DataFrame(records, columns=column_names)
    
    # Add a new column 'File' for generating filenames based on specific format
    df['File'] = df.apply(lambda row: 'P_WELL%s_RUN%s' % (row['Well'], row['Run']), axis=1)

    # Define the root directory path
    root_dir = "/Users/lzz/Documents/GitHub/IVF_AI/prediction/code"  # Replace with your actual root directory path

    # Create the top-level directory using the database filename and Well value
    for index, row in df.iterrows():
        well_value = str(row['Well'])
        top_level_dir_name = f"{db_filename}_P_{well_value}"
        top_level_dir = os.path.join(root_dir, top_level_dir_name, "images")

        # Ensure the images directory exists within the top-level directory
        if not os.path.exists(top_level_dir):
            os.makedirs(top_level_dir)

        # Create Focal-specific subdirectories within the images directory
        focal_value = str(row['Focal'])
        focal_dir = os.path.join(top_level_dir, f"F{focal_value}")
        
        if not os.path.exists(focal_dir):
            os.makedirs(focal_dir)
        
        # Generate the file name and path
        file_name = f"{row['File']}.jpeg"  # Save images in JPEG format
        file_path = os.path.join(focal_dir, file_name)
        
        # Save the image
        image_data = row['Image']
        save_image(file_path, image_data)

    print("---------------- Execution Completed ----- Total Count: %s" % len(df))

    # Close the database connection
    cur.close()
    conn.close()