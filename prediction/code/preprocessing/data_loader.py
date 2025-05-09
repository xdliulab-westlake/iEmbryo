import os
import tensorflow as tf
import numpy as np
import re
from PIL import Image

# Data Loading and Preprocessing
def preprocess_image_custom(image, img_size=224, is_training=True):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # [0,1]

    if is_training:
        # Resize while preserving aspect ratio (long side = 256~512)
        scale_size = tf.random.uniform([], minval=256, maxval=512, dtype=tf.int32)
        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        ratio = tf.cast(scale_size, tf.float32) / tf.cast(tf.minimum(height, width), tf.float32)
        new_height = tf.cast(tf.round(tf.cast(height, tf.float32) * ratio), tf.int32) 
        new_width = tf.cast(tf.round(tf.cast(width, tf.float32) * ratio), tf.int32)
        image = tf.image.resize(image, [new_height, new_width])
        image = tf.image.random_crop(image, size=[img_size, img_size, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2) 
        image = tf.image.random_hue(image, max_delta=0.02)
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    else:
        # Center crop
        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        ratio = tf.cast(img_size + 32, tf.float32) / tf.cast(tf.minimum(height, width), tf.float32)
        new_height = tf.cast(tf.round(tf.cast(height, tf.float32) * ratio), tf.int32)
        new_width = tf.cast(tf.round(tf.cast(width, tf.float32) * ratio), tf.int32)
        image = tf.image.resize(image, [new_height, new_width])
        image = tf.image.resize_with_crop_or_pad(image, img_size, img_size)

    mean = tf.constant([123.68, 116.78, 103.94], dtype=tf.float32)
    image = image * 255.0 - mean
    return image

def parse_image(img_path, label, img_size=224, is_training=True):
    """
    Reads and preprocesses an image using GPU operations.
    """
    # Ensure the path is a string tensor
    img_path = tf.convert_to_tensor(img_path, dtype=tf.string)

    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, expand_animations=False)

    # Automatically detect the number of channels
    num_channels = tf.shape(img)[-1]

    # Convert grayscale to RGB if needed
    img = tf.cond(
        tf.equal(num_channels, 1),
        lambda: tf.image.grayscale_to_rgb(img),
        lambda: img
    )

    # If image has more than 3 channels, keep only the first 3
    img = tf.cond(
        tf.greater(num_channels, 3),
        lambda: img[:, :, :3],
        lambda: img
    )

    img.set_shape([None, None, 3])
    img = preprocess_image_custom(img, img_size=img_size, is_training=is_training)
    return img, label

def load_data(data_dir, img_size=224, batch_size=32, is_training=True):
    image_paths = []
    labels = []
    ages = []
    qualities = []

    for label, class_name in enumerate(['class_0', 'class_1']):
        class_dir = os.path.join(data_dir, 'train', class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist.")
            continue

        for img_file in sorted(os.listdir(class_dir)):  
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue

            img_path = os.path.join(class_dir, img_file)
            match = re.search(r'_(\d+)\.\w+$', img_file)
            if match:
                age = int(match.group(1))
            else:
                print(f"Warning: Could not parse age information from filename {img_file}.")
                continue

            image_paths.append(img_path)
            labels.append(label)
            ages.append(age)
            qualities.append(1 if class_name == 'class_1' else 0)

    image_paths_tensor = tf.convert_to_tensor(image_paths, dtype=tf.string)
    labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int64)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths_tensor, labels_tensor))
    dataset = dataset.map(lambda x, y: parse_image(x, y, img_size, is_training), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    images_list = []
    labels_list = []
    for img_batch, label_batch in dataset:
        images_list.append(img_batch.numpy())
        labels_list.append(label_batch.numpy())

    return np.concatenate(images_list), np.concatenate(labels_list), np.array(ages), np.array(qualities), image_paths