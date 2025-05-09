import os
import re
import cv2
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from openpyxl import load_workbook
import sys

# Set up logging
log_file = "process_log.txt"
sys.stdout = open(log_file, "w")
sys.stderr = sys.stdout

# Function to check if an image is empty
def is_empty(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return True
    std_dev = np.std(image)
    return std_dev < 1

# Function to check if an image file is corrupted
def is_corrupted(img_path):
    try:
        with Image.open(img_path) as img:
            img.verify()  # Check if the image is corrupted
        return False
    except Exception:
        return True

# Function to traverse a directory and check all images
def check_images(data_root):
    corrupted_images = []
    all_images = []

    # Walk through all files in the directory
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(root, file)
                all_images.append(img_path)
                if is_corrupted(img_path):
                    corrupted_images.append(img_path)

    # If corrupted images are found, issue a warning
    if corrupted_images:
        print("Warning: The following images are corrupted:")
        for img in corrupted_images:
            print(f" - {img}")

    return all_images, corrupted_images

# Function to find directories containing specific embryo IDs
def find_images_by_embryo_ids(data_root, embryo_ids):
    matching_files = []

    for root, _, files in os.walk(data_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                match = re.search(r'_S(\d+)_', file)
                if match:
                    embryo_id = f"S{match.group(1)}"
                    if embryo_id in embryo_ids:
                        matching_files.append(os.path.join(root, file))

    return matching_files

# Function to find images in F.. directories
def find_images_in_f0_directories(directories, corrupted_images_set):
    image_info = []
    skipped_images = 0

    # Dictionary to group images by (EmbryoID, Focal) pair
    grouped_images = {}

    for dir_path in directories:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    match = re.search(r'_RUN\d+_(\d+)\.\w+$', file)
                    if match:
                        embryo_id = f"{match.group(1)}_{match.group(2)}"
                        run_number = int(match.group(3))
                        focal = f"F{match.group(4)}"
                        img_path = os.path.join(root, file)
                        if img_path in corrupted_images_set:
                            print(f"Skipping corrupted image: {img_path}")
                            continue
                        key = (embryo_id, focal)
                        if key not in grouped_images:
                            grouped_images[key] = []
                        grouped_images[key].append({
                            'path': img_path,
                            'run_number': run_number,
                            'parent_dir': focal,  # still use focal as group
                            'embryo_id': embryo_id
                        })

    # For each embryo + focal, keep the image with largest run number and not empty
    for (embryo_id, focal), images in grouped_images.items():
        images.sort(key=lambda x: x['run_number'], reverse=True)
        for img_data in images:
            if not is_empty(img_data['path']):
                image_info.append(img_data)
                break
            else:
                skipped_images += 1

    return image_info, skipped_images

# Function to create necessary directories
def create_directories(data_dir):
    train_dir = os.path.join(data_dir, 'train')

    # Remove existing directories if they exist
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)

    os.makedirs(os.path.join(train_dir, 'class_0'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'class_1'), exist_ok=True)

    return train_dir

# Function to move images to the appropriate directories with age information
def move_images(image_info, target_dir, class_label, metadata):
    for img_data in image_info:
        src_path = img_data['path']
        parent_dir_name = img_data['parent_dir']
        filename = os.path.basename(src_path)

        # Extract embryo_id considering the directory structure
        embryo_id = img_data.get('embryo_id', None)
        if embryo_id is None:
            print(f"Warning: Embryo ID not found in image info for {src_path}.")
            continue

        # Check if the embryo_id exists in metadata
        if embryo_id not in metadata['EmbryoID'].values:
            print(f"Warning: Embryo ID {embryo_id} not found in metadata.")
            continue

        # Get the age information from metadata
        try:
            age = int(metadata[metadata['EmbryoID'] == embryo_id]['Age'].values[0])
        except (IndexError, ValueError):
            print(f"Warning: Age information not found or invalid for Embryo ID {embryo_id}.")
            continue

        # Append F.. information and then age information to the filename
        base_name, ext = os.path.splitext(filename)
        new_filename = f"{base_name}_{parent_dir_name}_{age}{ext}"
        dst_path = os.path.join(target_dir, class_label, new_filename)

        try:
            shutil.copy(src_path, dst_path)
            print(f"Copied {src_path} to {dst_path}")
        except Exception as e:
            print(f"Failed to copy {src_path} to {dst_path}: {e}")

# Main script starts here
if __name__ == "__main__":
    # Root directory containing all embryo folders
    os.chdir("/Users/lzz/Documents/GitHub/IVF_AI/data_cleaning/output")
    data_root = '/Users/lzz/Documents/GitHub/IVF_AI/prediction/output/cluster_0'
    metadata = pd.read_csv("./merging_data.csv")

    # Step 1: Check all images in the root directory for corruption
    print("Checking all images for corruption...")
    all_images, corrupted_images = check_images(data_root)

    print(f"Total images found: {len(all_images)}")
    print(f"Total corrupted images: {len(corrupted_images)}")

    # Convert corrupted images list to a set for faster lookup
    corrupted_images_set = set(corrupted_images)

    # Proceed with further processing
    # Extract embryo information
    columns_to_extract = [1, 8, 9, 36, 44, 48]
    column_names = ['PatientID', 'BlastNumber', 'EmbryoQuality', 'TransferNumber', 'Outcome', 'FetusNumber']

    embryo_info_2022 = pd.read_excel("../../embryo_info/embryo_info_2022.xls")
    embryo_info_2022 = embryo_info_2022.iloc[:, columns_to_extract]
    embryo_info_2022.columns = column_names

    embryo_info_2023 = pd.read_excel("../../embryo_info/embryo_info_2023.xls")
    embryo_info_2023 = embryo_info_2023.iloc[:, columns_to_extract]
    embryo_info_2023.columns = column_names

    embryo_info = pd.concat([embryo_info_2022, embryo_info_2023], ignore_index=True)

    # Remove embryos without outcome or cannot grow into blastocyst
    embryo_info = embryo_info.dropna(subset=['Outcome'])
    embryo_info = embryo_info[embryo_info['BlastNumber'] != 0]
    pregnance_PatientID = embryo_info['PatientID'].values.astype(str)
    print("Embryo Info after filtering:")
    print(embryo_info)

    # Add pregnancy information
    metadata['EmbryoID'] = metadata['EmbryoID'].str.replace('-', '_', regex=False)
    print(metadata)

    pregnance_info = metadata[metadata['PatientID'].isin(pregnance_PatientID)]
    pregnance_Good = metadata[
        (metadata['EmbryoFate'] == 'Transfer') |
        (metadata['EmbryoFate'] == 'FrozenEmbryoTransfer')
    ]
    pregnance_Good = pregnance_Good[pregnance_Good['PatientID'].str.isdigit()]
    pregnance_Good['PatientID'] = pregnance_Good['PatientID'].astype(int)
    pregnance_Good = pregnance_Good.sort_values(by=['PatientID', 'EmbryoID']).reset_index(drop=True)
    pregnance_Good_unique = pregnance_Good.drop_duplicates(subset=['EmbryoID', 'EmbryoFate'])

    # Count the number of unique EmbryoIDs for each PatientID in pregnance_Good_unique
    embryo_info['PatientID'] = embryo_info['PatientID'].astype(int)
    pregnance_Good_unique['PatientID'] = pregnance_Good_unique['PatientID'].astype(int)
    embryo_count = pregnance_Good_unique.groupby('PatientID')['EmbryoID'].nunique().reset_index()
    embryo_count.columns = ['PatientID', 'EmbryoCount']
    merged_df = pd.merge(embryo_info, embryo_count, on='PatientID', how='left')
    consistent_patients = merged_df[merged_df['TransferNumber'] == merged_df['EmbryoCount']]['PatientID']

    # If consistent, merge FetusNumber into pregnance_Good_unique
    if not consistent_patients.empty:
        consistent_embryo_info = merged_df[merged_df['PatientID'].isin(consistent_patients)]
        result = pd.merge(
            pregnance_Good_unique,
            consistent_embryo_info[['PatientID', 'FetusNumber', 'EmbryoQuality']],
            on='PatientID',
            how='left'
        )
    else:
        result = pregnance_Good_unique.copy()  # If no consistent rows, just copy the original data
    print("Merged DataFrame with FetusNumber:")
    print(result)

    # The IDs of the good embryos and the poor embryos
    embryo_Good = metadata[
        (metadata['EmbryoFate'] == 'Transfer') |
        (metadata['EmbryoFate'] == 'FrozenEmbryoTransfer') |
        (metadata['EmbryoFate'] == 'Freeze')
    ]['EmbryoID'].values
    print(embryo_Good)
    embryo_Poor = metadata[
        ~(
            (metadata['EmbryoFate'] == 'Transfer') |
            (metadata['EmbryoFate'] == 'FrozenEmbryoTransfer') |
            (metadata['EmbryoFate'] == 'Freeze')
        )
    ]['EmbryoID'].values

    # Find directories whose names contain good embryos and poor embryos
    good_embryo_dirs = [data_root]
    poor_embryo_dirs = [data_root]

    # Find images in F.. directories for good and poor embryos
    good_f0_image_paths, good_skipped_images = find_images_in_f0_directories(good_embryo_dirs, corrupted_images_set)
    poor_f0_image_paths, poor_skipped_images = find_images_in_f0_directories(poor_embryo_dirs, corrupted_images_set)

    # Create necessary directories
    train_dir = create_directories(data_root)

    # Move images to the appropriate directories with age and F.. information appended
    move_images(good_f0_image_paths, train_dir, 'class_1', metadata)
    move_images(poor_f0_image_paths, train_dir, 'class_0', metadata)

    # Output the counts
    print(f"Number of good embryo images: {len(good_f0_image_paths)}")
    print(f"Number of poor embryo images: {len(poor_f0_image_paths)}")
    print(f"Number of good embryo blank images skipped: {good_skipped_images}")
    print(f"Number of poor embryo blank images skipped: {poor_skipped_images}")
    print(f"Total corrupted images skipped: {len(corrupted_images)}")
    print("All images have been moved to the train directory with F.. and age information appended.")
