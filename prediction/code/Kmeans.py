import os
import random
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from sklearn.cluster import KMeans
from time import time
from PIL import Image
import matplotlib.pyplot as plt

# current_directory = os.getcwd()
# new_directory = '/Users/lzz/Documents/GitHub/IVF/prediction/code'
# os.chdir(new_directory)

# # Root directory containing all embryo folders
# data_root = '../images'
# output = '../output'

# # Get a list of all image paths
# all_image_paths = []

# # Traverse through all embryo folders and collect image paths
# for root, dirs, files in os.walk(data_root):
#     if root.endswith('/F0'):  # Only consider the last directory as 'F-30'
#         for file in files:
#             if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#                 all_image_paths.append(os.path.join(root, file))

# # Shuffle the list of image paths
# random.shuffle(all_image_paths)

# # Define the split ratio
# train_ratio = 0.85
# val_ratio = 0.15

# # Calculate the split index
# split_index = int(len(all_image_paths) * train_ratio)

# # Split the image paths into training and validation sets
# train_images = all_image_paths[:split_index]
# val_images = all_image_paths[split_index:]

# # Create directories for the training and validation sets
# train_dir = os.path.join(data_root, 'train')
# val_dir = os.path.join(data_root, 'val')

# # If the training and validation directories exist, delete and recreate them
# if os.path.exists(train_dir):
#     shutil.rmtree(train_dir)
# os.makedirs(train_dir)

# if os.path.exists(val_dir):
#     shutil.rmtree(val_dir)
# os.makedirs(val_dir)

# # Function to copy images to a destination directory
# def copy_images(image_paths, dst_folder):
#     for src_path in image_paths:
#         dst_path = os.path.join(dst_folder, os.path.basename(src_path))
#         try:
#             if os.path.exists(dst_path):
#                 os.remove(dst_path)  # Remove the existing destination file
#             shutil.copy(src_path, dst_path)
#         except shutil.SameFileError:
#             print(f"Skipping {src_path} because it is the same as {dst_path}")

# # Copy the training set images to the training directory
# copy_images(train_images, train_dir)

# # Copy the validation set images to the validation directory
# copy_images(val_images, val_dir)

# print(f"Training set: {len(train_images)} images")
# print(f"Validation set: {len(val_images)} images")


# # Load the pre-trained VGG16 model
# base_model = VGG16(weights='imagenet', include_top=False)

# def extract_features(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
    
#     # Extract features
#     features = base_model.predict(x)
#     # Apply average pooling
#     features_mean = np.mean(features, axis=(1, 2))
#     return features_mean

# # Start timing
# start_time = time()

# # Extract features from the training and validation sets
# train_features_list = []
# val_features_list = []

# for path in train_images:
#     features = extract_features(path)
#     train_features_list.append(features)

# for path in val_images:
#     features = extract_features(path)
#     val_features_list.append(features)

# # Convert feature lists to NumPy arrays
# train_features_array = np.vstack(train_features_list)
# val_features_array = np.vstack(val_features_list)

# # Perform K-Means clustering
# n_clusters = 2  # Number of clusters you want
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# kmeans.fit(train_features_array)

# # Get the cluster labels for each sample
# train_labels = kmeans.labels_
# val_labels = kmeans.predict(val_features_array)

# # Create a DataFrame for the training set labels
# train_df = pd.DataFrame({
#     'Image Name': [os.path.basename(path) for path in train_images],
#     'Cluster Label': train_labels
# })

# # Create a DataFrame for the validation set labels
# val_df = pd.DataFrame({
#     'Image Name': [os.path.basename(path) for path in val_images],
#     'Cluster Label': val_labels
# })

# output_train_dir = os.path.join(output, 'prediction_Kmeans_F0_train.csv')
# output_val_dir = os.path.join(output, 'prediction_Kmeans_F0_val.csv')

# # Save the DataFrames to CSV files
# train_df.to_csv(output_train_dir, index=False)
# val_df.to_csv(output_val_dir, index=False)

# # End timing
# end_time = time()
# elapsed_time = end_time - start_time

# print(f"Cluster labels saved to CSV files.")
# print(f"Total elapsed time: {elapsed_time:.2f} seconds")


# Set working directory and define absolute paths for image root directory and output directory
current_directory = os.getcwd()
new_directory = '/Users/lzz/Documents/GitHub/IVF_AI/prediction/code'
os.chdir(new_directory)

# Define absolute paths for image root directory and output directory
abs_data_root = os.path.abspath('../images/')  # Absolute path to the folder containing images
abs_output = os.path.abspath('../output')

# Get a list of all image paths using absolute paths
all_image_paths = []

# Traverse through all embryo folders and collect image paths using absolute paths
for root, dirs, files in os.walk(abs_data_root):
    if root.endswith('/F0'):  # Only consider the last directory as 'F0'
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                abs_path = os.path.join(root, file)
                all_image_paths.append(abs_path)

# Shuffle the list of image paths
random.shuffle(all_image_paths)

print(f"Total images: {len(all_image_paths)} images")

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # Preprocess only for feature extraction
    
    # Extract features
    features = base_model.predict(x)
    # Apply average pooling
    features_mean = np.mean(features, axis=(1, 2))
    return features_mean

# Start timing
start_time = time()

# Extract features from all images
all_features_list = []

for path in all_image_paths:
    features = extract_features(path)
    all_features_list.append(features)

# Convert feature list to NumPy array
all_features_array = np.vstack(all_features_list)

# Perform K-Means clustering
n_clusters = 2  # Number of clusters you want
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(all_features_array)

# Get the cluster labels for each sample
all_labels = kmeans.labels_

# Create a DataFrame for the cluster labels with absolute paths
all_df = pd.DataFrame({
    'Image Name': [os.path.basename(path) for path in all_image_paths],
    'Cluster Label': all_labels,
    'Absolute Path': all_image_paths  # Store absolute paths in the DataFrame
})

# Save the DataFrame to a CSV file
output_all_dir = os.path.join(abs_output, 'prediction_Kmeans_F0_all.csv')
all_df.to_csv(output_all_dir, index=False)

# End timing
end_time = time()
elapsed_time = end_time - start_time

print(f"Cluster labels saved to {output_all_dir}")
print(f"Total elapsed time: {elapsed_time:.2f} seconds")

# Function to display images from a specific cluster using absolute paths without preprocessing
def display_images_from_cluster(cluster_label, df, num_images=10):
    """Display up to `num_images` from the specified cluster using absolute paths."""
    
    # Filter the DataFrame for the given cluster label
    cluster_df = df[df['Cluster Label'] == cluster_label]
    
    # Select the first `num_images` entries or all if fewer than `num_images`
    images_to_display = cluster_df.head(num_images)
    
    # Create a figure to display these images
    fig = plt.figure(figsize=(15, 4))
    for idx, (_, row) in enumerate(images_to_display.iterrows(), start=1):
        img_path = row['Absolute Path']
        try:
            with Image.open(img_path).convert('RGB') as img:  # Ensure image is in RGB mode
                # Resize the image for display purposes
                img = img.resize((224, 224))  # Resize to match the model input size
                ax = fig.add_subplot(1, num_images, idx)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f"Cluster {cluster_label}")
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")
    plt.show()

# Display images from each cluster using absolute paths from all_image_paths
unique_clusters = all_df['Cluster Label'].unique()
for cluster in unique_clusters:
    display_images_from_cluster(cluster, all_df)

print("Display completed.")