import os
import re
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to rename the files
def replace_dots_with_underscores(s):
    parts = re.split(r'[.-]', s)
    return '_'.join(parts)

# Function to find directories containing specific embryo IDs
def find_directories_with_embryos(data_root, embryo_ids):
    matching_dirs = []
    for root, dirs, files in os.walk(data_root):
        for dir_name in dirs:
            if dir_name in embryo_ids:
                matching_dirs.append(os.path.join(root, dir_name))
    return matching_dirs

# Function to find images in F0 directories
def find_images_in_f0_directories(directories):
    image_paths = []
    for dir_path in directories:
        max_run_number = -1
        max_image_path = None
        for root, dirs, files in os.walk(dir_path):
            if root.endswith('/F-30'):  # Only consider the last directory as 'F0'
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        match = re.search(r'RUN(\d+)', file)
                        if match:
                            run_number = int(match.group(1))
                            if run_number > max_run_number:
                                max_run_number = run_number
                                max_image_path = os.path.join(root, file)
        if max_image_path:
            image_paths.append(max_image_path)
    return image_paths

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
def move_images(image_paths, target_dir, class_label, metadata):
    for src_path in image_paths:
        filename = os.path.basename(src_path)
        # Extract embryo_id considering the additional 'images' directory
        embryo_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(src_path))))

        # Check if the embryo_id exists in metadata
        if embryo_id not in metadata['EmbryoID'].values:
            print(f"Warning: Embryo ID {embryo_id} not found in metadata.")
            continue
        
        # Get the age information from metadata
        age = int(metadata[metadata['EmbryoID'] == embryo_id]['Age'].values[0])
        
        # Append age information to the filename
        base_name, ext = os.path.splitext(filename)
        new_filename = f"{base_name}_{age}{ext}"
        dst_path = os.path.join(target_dir, class_label, new_filename)
        shutil.copy(src_path, dst_path)

# Set working directory
current_directory = os.getcwd()
new_directory = '/Users/lzz/Documents/GitHub/IVF_AI/prediction/code'
os.chdir(new_directory)

# Root directory containing all embryo folders
data_root = '../images'
metadata = pd.read_csv("../../data_cleaning/output/merging_data.csv")

# The ID of the good embryo and the bad embryo
metadata['EmbryoID'] = metadata['EmbryoID'].apply(replace_dots_with_underscores)
embryo_Good = metadata[(metadata['EmbryoFate'] == 'Transfer') | (metadata['EmbryoFate'] == 'FrozenEmbryoTransfer')]['EmbryoID'].values
embryo_Poor = metadata[~((metadata['EmbryoFate'] == 'Transfer') | (metadata['EmbryoFate'] == 'FrozenEmbryoTransfer'))]['EmbryoID'].values

# Find directories whose names contain good embryos and poor embryos
good_embryo_dirs = find_directories_with_embryos(data_root, embryo_Good)
poor_embryo_dirs = find_directories_with_embryos(data_root, embryo_Poor)

# Find images in F0 directories for good and poor embryos
good_f0_image_paths = find_images_in_f0_directories(good_embryo_dirs)
poor_f0_image_paths = find_images_in_f0_directories(poor_embryo_dirs)

# Create necessary directories
train_dir = create_directories(data_root)

# Move images to the appropriate directories with age information
move_images(good_f0_image_paths, train_dir, 'class_1', metadata)
move_images(poor_f0_image_paths, train_dir, 'class_0', metadata)

# Output the counts
print(f"Number of good embryos: {len(good_embryo_dirs)}")
print(f"Number of poor embryos: {len(poor_embryo_dirs)}")
print(f"Number of good embryo images: {len(good_f0_image_paths)}")
print(f"Number of poor embryo images: {len(poor_f0_image_paths)}")
print("All images have been moved to the train directory with age information appended.")