# data_utils.py

import numpy as np
import tensorflow as tf

def balance_classes(images, labels, ages, qualities):
    """
    Ensures an equal number of samples from each class by downsampling the larger class.
    """
    def ensure_numpy_array(array):
        if isinstance(array, tf.Tensor):
            return array.numpy()
        elif isinstance(array, list):
            if isinstance(array[0], np.ndarray):
                return np.concatenate(array, axis=0)
            return np.array(array)
        return array

    images = ensure_numpy_array(images)
    labels = ensure_numpy_array(labels)
    ages = ensure_numpy_array(ages)
    qualities = ensure_numpy_array(qualities)

    labels = np.reshape(labels, (-1,))
    ages = np.reshape(ages, (-1,))
    qualities = np.reshape(qualities, (-1,))

    unique_classes, class_counts = np.unique(labels, return_counts=True)
    min_count = min(class_counts)

    balanced_images, balanced_labels, balanced_ages, balanced_qualities = [], [], [], []

    for class_label in unique_classes:
        class_indices = np.where(labels == class_label)[0]
        if len(class_indices) > min_count:
            selected_indices = np.random.choice(class_indices, size=min_count, replace=False)
        else:
            selected_indices = class_indices

        balanced_images.extend(images[selected_indices])
        balanced_labels.extend(labels[selected_indices])
        balanced_ages.extend(ages[selected_indices])
        balanced_qualities.extend(qualities[selected_indices])

    return (
        np.array(balanced_images),
        np.array(balanced_labels),
        np.array(balanced_ages),
        np.array(balanced_qualities)
    )

def check_class_distribution(y_train, y_val, y_test):
    class_names = ['Class 0', 'Class 1']    
    train_counts = np.bincount(y_train)
    val_counts = np.bincount(y_val)
    test_counts = np.bincount(y_test)

    print("Train set class distribution:")
    for i, count in enumerate(train_counts):
        print(f"{class_names[i]}: {count}")

    print("Validation set class distribution:")
    for i, count in enumerate(val_counts):
        print(f"{class_names[i]}: {count}")

    print("Test set class distribution:")
    for i, count in enumerate(test_counts):
        print(f"{class_names[i]}: {count}")
