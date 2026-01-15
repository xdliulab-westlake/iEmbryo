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

# Function to extract features from an image using VGG16
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

# Function to copy images into the respective cluster directories and record paths
def copy_images_to_cluster_dirs(df, cluster_dirs):
    """Copy images to their respective cluster directories and record their paths."""
    cluster_paths = {i: [] for i in range(len(cluster_dirs))}
    
    for idx, row in df.iterrows():
        src_path = row['Absolute Path']
        cluster_label = row['Cluster Label']
        img_name = row['Image Name']
        
        # Define destination path within the cluster directory
        dst_dir = cluster_dirs[cluster_label]
        dst_path = os.path.join(dst_dir, img_name)
        
        try:
            shutil.copy(src_path, dst_path)
            cluster_paths[cluster_label].append(dst_path)
        except Exception as e:
            print(f"Failed to copy image {src_path} to {dst_path}: {e}")
    
    return cluster_paths

# Main script starts here
if __name__ == "__main__":
    # Set working directory and define absolute paths for image root directory and output directory
    current_directory = os.getcwd()
    new_directory = '/Users/lzz/Documents/GitHub/IVF_AI/prediction/code'
    os.chdir(new_directory)

    # Define absolute paths for image root directory and output directory
    abs_data_root = os.path.abspath('../images/selected_images_tM/train')  # Absolute path to the folder containing images
    abs_output = os.path.abspath('../output')

    # Ensure output directories exist
    os.makedirs(abs_output, exist_ok=True)

    # Get a list of all image paths using absolute paths
    all_image_paths = []

    # Traverse through all embryo folders and collect image paths using absolute paths
    for root, dirs, files in os.walk(abs_data_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                abs_path = os.path.join(root, file)
                all_image_paths.append(abs_path)

    # Shuffle the list of image paths
    random.shuffle(all_image_paths)

    print(f"Total images: {len(all_image_paths)} images")

    # Load the pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False)

    # Start timing
    start_time = time()

    # Extract features from all images
    all_features_list = [extract_features(path) for path in all_image_paths]

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

    # Display images from each cluster using absolute paths from all_image_paths
    unique_clusters = all_df['Cluster Label'].unique()
    for cluster in unique_clusters:
        display_images_from_cluster(cluster, all_df)

    print("Display completed.")

    # Create directories for storing images from each cluster if they don't exist
    cluster_dirs = [os.path.join(abs_output, f'cluster_{i}') for i in range(n_clusters)]
    for dir_path in cluster_dirs:
        os.makedirs(dir_path, exist_ok=True)

    # Copy images to cluster directories and get the paths
    cluster_image_paths = copy_images_to_cluster_dirs(all_df, cluster_dirs)