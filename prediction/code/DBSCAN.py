import os
import random
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from sklearn.cluster import DBSCAN
from time import time
import matplotlib.pyplot as plt

# Set work directory
current_directory = os.getcwd()
new_directory = '/Users/lzz/Documents/GitHub/IVF/prediction/code'
os.chdir(new_directory)

# Root directory containing all embryo folders
data_root = '../images'
output = '../output'

# Get a list of all image paths
all_image_paths = []

# Traverse through all embryo folders and collect image paths
for root, dirs, files in os.walk(data_root):
    if root.endswith('/F0'):  # Only consider the last directory as 'F0'
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                all_image_paths.append(os.path.join(root, file))

# Shuffle the list of image paths
random.shuffle(all_image_paths)

# Define the split ratio
train_ratio = 0.85
val_ratio = 0.15

# Calculate the split index
split_index = int(len(all_image_paths) * train_ratio)

# Split the image paths into training and validation sets
train_images = all_image_paths[:split_index]
val_images = all_image_paths[split_index:]

# Create directories for the training and validation sets
train_dir = os.path.join(data_root, 'train')
val_dir = os.path.join(data_root, 'val')

# If the training and validation directories exist, delete and recreate them
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
os.makedirs(train_dir)

if os.path.exists(val_dir):
    shutil.rmtree(val_dir)
os.makedirs(val_dir)

# Function to copy images to a destination directory
def copy_images(image_paths, dst_folder):
    for src_path in image_paths:
        dst_path = os.path.join(dst_folder, os.path.basename(src_path))
        try:
            if os.path.exists(dst_path):
                os.remove(dst_path)  # Remove the existing destination file
            shutil.copy(src_path, dst_path)
        except shutil.SameFileError:
            print(f"Skipping {src_path} because it is the same as {dst_path}")

# Copy the training set images to the training directory
copy_images(train_images, train_dir)

# Copy the validation set images to the validation directory
copy_images(val_images, val_dir)

print(f"Training set: {len(train_images)} images")
print(f"Validation set: {len(val_images)} images")

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Extract features
    features = base_model.predict(x)
    # Apply average pooling
    features_mean = np.mean(features, axis=(1, 2))
    return features_mean

# Start timing
start_time = time()

# Extract features from the training and validation sets
train_features_list = []
val_features_list = []

for path in train_images:
    features = extract_features(path)
    train_features_list.append(features)

for path in val_images:
    features = extract_features(path)
    val_features_list.append(features)

# Convert feature lists to NumPy arrays
train_features_array = np.vstack(train_features_list)
val_features_array = np.vstack(val_features_list)

# Function to perform DBSCAN clustering and evaluate the results
def perform_dbscan(features, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"eps={eps}, min_samples={min_samples} -> Clusters: {n_clusters}, Noise: {n_noise}")
    return labels, n_clusters, n_noise

# Try different values of eps and min_samples
eps_values = [0.1, 0.3, 0.5, 0.7, 1.0]
min_samples_values = [5, 10, 15, 20]

best_params = None
best_n_clusters = 0
best_n_noise = float('inf')

for eps in eps_values:
    for min_samples in min_samples_values:
        labels, n_clusters, n_noise = perform_dbscan(train_features_array, eps, min_samples)
        if n_clusters == 2 and n_noise < best_n_noise:
            best_params = (eps, min_samples)
            best_n_clusters = n_clusters
            best_n_noise = n_noise

# Use the best parameters for final clustering
best_eps, best_min_samples = best_params
print(f"Best parameters: eps={best_eps}, min_samples={best_min_samples}")

# Perform final DBSCAN clustering
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
train_labels = dbscan.fit_predict(train_features_array)
val_labels = dbscan.fit_predict(val_features_array)

# Create a DataFrame for the training set labels
train_df = pd.DataFrame({
    'Image Name': [os.path.basename(path) for path in train_images],
    'Cluster Label': train_labels
})

# Create a DataFrame for the validation set labels
val_df = pd.DataFrame({
    'Image Name': [os.path.basename(path) for path in val_images],
    'Cluster Label': val_labels
})

output_train_dir = os.path.join(output, 'prediction_DBSCAN_F0_train.csv')
output_val_dir = os.path.join(output, 'prediction_DBSCAN_F0_val.csv')

# Save the DataFrames to CSV files
train_df.to_csv(output_train_dir, index=False)
val_df.to_csv(output_val_dir, index=False)

# End timing
end_time = time()
elapsed_time = end_time - start_time

print(f"Cluster labels saved to CSV files.")
print(f"Total elapsed time: {elapsed_time:.2f} seconds")