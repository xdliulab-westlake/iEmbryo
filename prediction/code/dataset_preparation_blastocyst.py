import os
import re
import cv2
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from openpyxl import load_workbook

# Function to rename the files
def replace_dots_with_underscores(s):
    parts = re.split(r'[.-]', s)
    return '_'.join(parts)

# Function to check if an image is empty
def is_empty(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return True
    std_dev = np.std(image)
    return std_dev < 1

# Function to find directories containing specific embryo IDs
def find_directories_with_embryos(data_root, embryo_ids):
    matching_dirs = []
    for root, dirs, files in os.walk(data_root):
        for dir_name in dirs:
            if dir_name in embryo_ids:
                matching_dirs.append(os.path.join(root, dir_name))
    return matching_dirs

# Function to find images in F.. directories
def find_images_in_f0_directories(directories):
    image_info = []
    skipped_images = 0
    
    for dir_path in directories:
        # Dictionary to hold images grouped by F.. directory
        f_dirs_images = {}
        
        # Walk through the directory and group images by their parent F.. directory
        for root, dirs, files in os.walk(dir_path):
            parent_dir_name = os.path.basename(root)
            if parent_dir_name.startswith('F'):
                if parent_dir_name not in f_dirs_images:
                    f_dirs_images[parent_dir_name] = []
                
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        match = re.search(r'RUN(\d+)', file)
                        if match:
                            run_number = int(match.group(1))
                            f_dirs_images[parent_dir_name].append({
                                'path': os.path.join(root, file),
                                'run_number': run_number,
                                'parent_dir': parent_dir_name
                            })
        
        # Process each F.. directory separately
        for f_dir, images in f_dirs_images.items():
            if images:
                # Sort images by RUN number in descending order
                images.sort(key=lambda x: x['run_number'], reverse=True)
                
                # Find the first non-empty image from this F.. directory
                for img_data in images:
                    if not is_empty(img_data['path']):
                        image_info.append(img_data)
                        break  # Only one image per F.. directory
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
        
        # Extract embryo_id considering the additional 'images' directory
        embryo_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(src_path))))

        # Check if the embryo_id exists in metadata
        if embryo_id not in metadata['EmbryoID'].values:
            print(f"Warning: Embryo ID {embryo_id} not found in metadata.")
            continue
        
        # Get the age information from metadata
        try:
            age = int(metadata[metadata['EmbryoID'] == embryo_id]['Age'].values[0])
        except IndexError:
            print(f"Warning: Age information not found for Embryo ID {embryo_id}.")
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

# Set working directory
current_directory = os.getcwd()
new_directory = '/Users/lzz/Documents/GitHub/IVF_AI/prediction/code'
os.chdir(new_directory)

# Root directory containing all embryo folders
data_root = '../images/selected_images'
metadata = pd.read_csv("../../data_cleaning/output/merging_data.csv")

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

# Remove embryos without outcome or can not grow into blastocyst 
embryo_info = embryo_info.dropna(subset=['Outcome'])
embryo_info = embryo_info[embryo_info['BlastNumber'] != 0]
pregnance_PatientID = embryo_info['PatientID'].values.astype(str)
print("Embryo Info after filtering:")
print(embryo_info)

# Add pregnance information
metadata['EmbryoID'] = metadata['EmbryoID'].apply(replace_dots_with_underscores)
pregnance_info = metadata[metadata['PatientID'].isin(pregnance_PatientID)]
pregnance_Good = metadata[(metadata['EmbryoFate'] == 'Transfer') | (metadata['EmbryoFate'] == 'FrozenEmbryoTransfer')]
pregnance_Good = pregnance_Good[pregnance_Good['PatientID'].str.isdigit()]
pregnance_Good['PatientID'] = pregnance_Good['PatientID'].astype(int)
pregnance_Good = pregnance_Good.sort_values(by=['PatientID','EmbryoID']).reset_index(drop=True)
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
    result = pd.merge(pregnance_Good_unique, consistent_embryo_info[['PatientID', 'FetusNumber', 'EmbryoQuality']], on='PatientID', how='left')
else:
    result = pregnance_Good_unique.copy()  # If no consistent rows, just copy the original data
print("Merged DataFrame with FetusNumber:")
print(result)

# The ID of the good embryo and the bad embryo
embryo_Good = metadata[(metadata['EmbryoFate'] == 'Transfer') | (metadata['EmbryoFate'] == 'FrozenEmbryoTransfer') | (metadata['EmbryoFate'] == 'Freeze')]['EmbryoID'].values
embryo_Poor = metadata[~((metadata['EmbryoFate'] == 'Transfer') | (metadata['EmbryoFate'] == 'FrozenEmbryoTransfer') | (metadata['EmbryoFate'] == 'Freeze'))]['EmbryoID'].values

# Find directories whose names contain good embryos and poor embryos
good_embryo_dirs = find_directories_with_embryos(data_root, embryo_Good)
poor_embryo_dirs = find_directories_with_embryos(data_root, embryo_Poor)

# Find images in F.. directories for good and poor embryos
good_f0_image_paths, good_skipped_images = find_images_in_f0_directories(good_embryo_dirs)
poor_f0_image_paths, poor_skipped_images = find_images_in_f0_directories(poor_embryo_dirs)

# Create necessary directories
train_dir = create_directories(data_root)

# Move images to the appropriate directories with age and F.. information
move_images(good_f0_image_paths, train_dir, 'class_1', metadata)
move_images(poor_f0_image_paths, train_dir, 'class_0', metadata)

# Output the counts 
print(f"Number of good embryo images: {len(good_f0_image_paths)}")
print(f"Number of poor embryo images: {len(poor_f0_image_paths)}")
print(f"Number of good embryo blank images skipped: {good_skipped_images}")
print(f"Number of poor embryo blank images skipped: {poor_skipped_images}")
print("All images have been moved to the train directory with F.. and age information appended.")