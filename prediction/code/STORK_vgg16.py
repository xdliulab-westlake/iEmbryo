import os
import re
import glob
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# Define the strategy for multi-GPU training
strategy = tf.distribute.MirroredStrategy()
num_gpus = len(tf.config.list_physical_devices('GPU'))
batch_size_per_replica = 128  # Original batch size per GPU
global_batch_size = batch_size_per_replica * num_gpus

# 1. Data Loading and Preprocessing
def load_data(data_dir, img_size=224):
    images = []
    labels = []
    ages = []
    qualities = []  # Embryo quality labels
    
    for label, class_name in enumerate(['class_0', 'class_1']):
        class_dir = os.path.join(data_dir, 'train', class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist.")
            continue
        
        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue
            
            img_path = os.path.join(class_dir, img_file)
            
            try:
                # Load and resize image without normalizing
                img = Image.open(img_path).resize((img_size, img_size))
                img_array = np.array(img)  # Remove normalization (/ 255.0)
                
                if img_array.ndim == 2:  # If grayscale, convert to RGB
                    img_array = np.stack((img_array,) * 3, axis=-1)
                images.append(img_array)

                # Extract age information from filename
                match = re.search(r'_RUN\d+_(F-?\d+)_(\d+)\.\w+$', img_file)
                if match:
                    age = int(match.group(2))
                    ages.append(age)
                    
                    # Add quality information based on class name
                    quality = 1 if class_name == 'class_1' else 0
                    qualities.append(quality)
                    
                    labels.append(label)
                else:
                    print(f"Warning: Could not parse age information from filename {img_file}.")
            except Exception as e:
                print(f"Error processing file {img_file}: {e}")
    
    images = np.array(images)
    labels = np.array(labels)
    ages = np.array(ages)
    qualities = np.array(qualities)  # Embryo quality labels
    
    return images, labels, ages, qualities


# 2. Build a pre-trained VGG16 model with custom top layers
def build_vgg16_model(input_shape=(224, 224, 3), num_classes=2):
    # Load the VGG16 model without the top layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the convolutional base
    base_model.trainable = False

    # Create the new model on top
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model with a learning rate schedule
    initial_learning_rate = 0.0001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.9,
        staircase=True
    )
    optimizer = Adam(learning_rate=lr_schedule,
                     weight_decay=1e-5)
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Train Deep Learning Model with Data Augmentation using VGG16
def balance_classes(images, labels, ages, qualities):
    class_counts = np.bincount(labels)
    min_count = min(class_counts)
    balanced_images = []
    balanced_labels = []
    balanced_ages = []
    balanced_qualities = []
    
    for class_label in range(len(class_counts)):
        class_indices = np.where(labels == class_label)[0]
        if len(class_indices) > min_count:
            selected_indices = np.random.choice(class_indices, size=min_count, replace=False)
        else:
            selected_indices = class_indices
        
        balanced_images.extend(images[selected_indices])
        balanced_labels.extend(labels[selected_indices])
        balanced_ages.extend(ages[selected_indices])
        balanced_qualities.extend(qualities[selected_indices])
    
    return np.array(balanced_images), np.array(balanced_labels), np.array(balanced_ages), np.array(balanced_qualities)

# Check class distribution
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

# check gpu availability
def check_gpu_availability():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # Print detailed information about each device
    for device in tf.config.list_physical_devices():
        print(device)

# 3. Training the model with augmentation
def train_model_with_augmentation(data_dir, num_epochs=3, global_batch_size=None, img_size=224, resume_training=True, workers=30, max_queue_size = 60):
    # Check GPU availability before training starts
    check_gpu_availability()

    # Assuming load_data is defined elsewhere and loads images, labels, ages, qualities from data_dir
    images, labels, ages, qualities = load_data(data_dir, img_size)
    
    # Balance classes
    balanced_images, balanced_labels, balanced_ages, balanced_qualities = balance_classes(images, labels, ages, qualities)

    # Split the dataset with stratification
    X_train, X_temp, y_train, y_temp, age_train, age_temp, quality_train, quality_temp = train_test_split(
        balanced_images, balanced_labels, balanced_ages, balanced_qualities, test_size=0.3, random_state=42, stratify=balanced_labels
    )
    X_val, X_test, y_val, y_test, age_val, age_test, quality_val, quality_test = train_test_split(
        X_temp, y_temp, age_temp, quality_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    check_class_distribution(y_train, y_val, y_test)
    
    # Define the model
    model_path = 'vgg16_model.h5'
    initial_epoch = 0
    
    if resume_training and os.path.exists(model_path):
        print("Loading existing model from", model_path)
        model = tf.keras.models.load_model(model_path)
        
        # Try to extract the epoch number from the training log
        if os.path.exists('training.log'):
            with open('training.log', 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1]
                    initial_epoch = int(last_line.split(',')[0]) + 1
        else:
            print("Training log not found. Setting initial epoch to 0.")
            initial_epoch = 0
    else:
        print("Building new model from scratch.")
        model = build_vgg16_model((img_size, img_size, 3), num_classes=2)
    
    # Define CSV logger to log training history
    csv_logger = CSVLogger('training.log', append=True)
    
    # Data augmentation using tf.image operations
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.resize_with_crop_or_pad(image, img_size + 20, img_size + 20)  # Add padding
        image = tf.image.random_crop(image, [img_size, img_size, 3])  # Random crop back to original size
        return image, label

    # Create datasets with efficient data pipeline
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = (
        train_dataset
        .shuffle(buffer_size=1024)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)  # Apply data augmentation
        .batch(global_batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(global_batch_size).prefetch(tf.data.AUTOTUNE)

    # Train the model with data augmentation
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        initial_epoch=initial_epoch,
        callbacks=[csv_logger]
    )
    # Check GPU usage after training completes
    check_gpu_availability()
    
    # Save the model
    model.save(model_path)
    model.summary()

    print("VGG16 Model saved as", model_path)
    
    return X_train, y_train, age_train, quality_train, X_test, y_test, age_test, quality_test

# 4. Evaluate Deep Learning Model
def evaluate_deep_learning_model(X_test, y_test, model_path):
    # Load model
    dl_model = tf.keras.models.load_model(model_path)
    
    # Ensure evaluation is done using the same global batch size as training
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(global_batch_size)

    # Evaluate model
    test_loss, test_accuracy = dl_model.evaluate(X_test, y_test)
    print(f"Deep Learning Model Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Compute AUC for model
    probabilities_dl = dl_model.predict(X_test)
    if len(np.unique(y_test)) < 2:
        print("Only one class present in y_test. ROC AUC score is not defined in that case for the Deep Learning Model.")
    else:
        auc_dl = roc_auc_score(y_test, probabilities_dl[:, 1])
        print(f"Deep Learning Model Test AUC: {auc_dl:.4f}")
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, probabilities_dl[:, 1])
        # plt.figure(figsize=(8, 6))
        # plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc_dl:.2f})')
        # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic (ROC) Curve')
        # plt.legend(loc="lower right")
        # plt.show()
    
    # Compute confusion matrix for deep learning model
    y_pred_dl = np.argmax(probabilities_dl, axis=1)
    cm_dl = confusion_matrix(y_test, y_pred_dl)
    
    # Create confusion matrix 
    class_names = ['Class 1', 'Class 0']
    cm_df = pd.DataFrame(cm_dl, index=class_names, columns=class_names)
    print("Deep Learning Model Confusion Matrix:")
    print(cm_df)
    
# 5. Build Decision Tree Model
def build_decision_tree(X_train, y_train, age_train, quality_train):
    # Combine image features, age, and quality into a single feature matrix
    X_train_combined = np.hstack((X_train, age_train.reshape(-1, 1), quality_train.reshape(-1, 1)))
    
    # Define and train the decision tree model
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    dt = DecisionTreeClassifier()
    
    # Use StratifiedKFold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(dt, param_grid, cv=skf, scoring='accuracy')
    grid_search.fit(X_train_combined, y_train)
    
    best_dt = grid_search.best_estimator_
    
    return best_dt

# 6. Train Decision Tree Model
def train_decision_tree(X_train, y_train, age_train, quality_train, X_test, y_test, age_test, quality_test, model_path):
    # Load the deep learning model
    dl_model = tf.keras.models.load_model(model_path)
    
    # Get the image features from the deep learning model
    intermediate_layer_model = Model(inputs=dl_model.input, outputs=dl_model.layers[-2].output)
    X_train_features = intermediate_layer_model.predict(X_train)
    X_test_features = intermediate_layer_model.predict(X_test)
    
    # Train the decision tree model
    dt_model = build_decision_tree(X_train_features, y_train, age_train, quality_train)
    
    # Save the decision tree model
    joblib.dump(dt_model, 'decision_tree_model.pkl')
    print("Decision Tree Model saved as decision_tree_model.pkl")
    
    return dt_model, X_test_features

# 7. Evaluate Decision Tree Model
def evaluate_decision_tree_model(X_test, y_test, age_test, quality_test, X_test_features, dt_model):
    # Combine image features, age, and quality into a single feature matrix
    X_test_combined = np.hstack((X_test_features, age_test.reshape(-1, 1), quality_test.reshape(-1, 1)))
    
    # Evaluate the decision tree model
    dt_predictions = dt_model.predict(X_test_combined)
    dt_accuracy = accuracy_score(y_test, dt_predictions)
    print(f"Decision Tree Model Test Accuracy: {dt_accuracy:.4f}")
    
    # Compute AUC for decision tree model
    probabilities_dt = dt_model.predict_proba(X_test_combined)[:, 1]
    if len(np.unique(y_test)) < 2:
        print("Only one class present in y_test. ROC AUC score is not defined in that case for the Decision Tree Model.")
    else:
        auc_dt = roc_auc_score(y_test, probabilities_dt)
        print(f"Decision Tree Model Test AUC: {auc_dt:.4f}")
        
        # Plot ROC curve
        # fpr, tpr, _ = roc_curve(y_test, probabilities_dt)
        # plt.figure(figsize=(8, 6))
        # plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc_dt:.2f})')
        # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic (ROC) Curve')
        # plt.legend(loc="lower right")
        # plt.show()

    # Compute confusion matrix for decision tree model
    cm_dt = confusion_matrix(y_test, dt_predictions)
    print("Decision Tree Model Confusion Matrix:")
    print(cm_dt)
    
    # Display the decision tree graphically with accuracy and AUC
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model, filled=True, feature_names=['feature_' + str(i) for i in range(X_test_features.shape[1])] + ['age', 'quality'], class_names=['class_0', 'class_1'])
    
    # Add accuracy and AUC to the plot
    plt.text(0.5, 0.5, f"Accuracy: {dt_accuracy:.4f}\nAUC: {auc_dt:.4f}", fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.title("Decision Tree Structure")
    plt.show()
    
    # Calculate and display the probability of successful pregnancy for each leaf node
    def calculate_leaf_probabilities(dt_model, X_test_combined, y_test):
        # Predict the leaf nodes for the test data
        leaf_nodes = dt_model.apply(X_test_combined)
        
        # Group by leaf nodes and calculate the success probability
        leaf_probabilities = {}
        for leaf_node in np.unique(leaf_nodes):
            indices = np.where(leaf_nodes == leaf_node)[0]
            if len(indices) > 0:
                positive_samples = np.sum(y_test[indices])
                total_samples = len(indices)
                probability = positive_samples / total_samples
                leaf_probabilities[leaf_node] = {
                    'total_samples': total_samples,
                    'positive_samples': positive_samples,
                    'probability': probability
                }
        
        return leaf_probabilities

    # Call the function to calculate probabilities
    leaf_probabilities = calculate_leaf_probabilities(dt_model, X_test_combined, y_test)
    
    # Print the results
    for leaf_node, stats in leaf_probabilities.items():
        print(f"Leaf Node: {leaf_node}")
        print(f"Total Samples: {stats['total_samples']}")
        print(f"Positive Samples: {stats['positive_samples']}")
        print(f"Probability of Successful Pregnancy: {stats['probability']:.4f}")
        print("-" * 50)

def show_samples(images, labels, num_samples=25):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        axes[i].imshow(images[i])
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis('off')
    plt.show()

# Main function
if __name__ == "__main__":
    data_dir = '/Users/lzz/Documents/GitHub/IVF_AI/prediction/images/selected_images'

    # Run dataset_prepration.py
    # with open('/Users/lzz/Documents/GitHub/IVF_AI/prediction/code/dataset_preparation_blastocyst.py') as f:
    #     code = f.read()
    # exec(code)

    # Kmeans.py

    # remove black images    
    csv_path = "/Users/lzz/Documents/GitHub/IVF_AI/prediction/output/prediction_Kmeans_F0_all.csv"
    df = pd.read_csv(csv_path)

    black_images = df[df['Cluster Label'] == 1]
    filtered_df = df[df['Cluster Label'] != 1]

    output_filtered_csv_path = "/Users/lzz/Documents/GitHub/IVF_AI/prediction/output/prediction_Kmeans_F0_filtered.csv"
    filtered_df.to_csv(output_filtered_csv_path, index=False)
    print(f"Filtered CSV saved to {output_filtered_csv_path}")

    for index, row in black_images.iterrows():
        img_path = row['Absolute Path']
        if os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception as e:
                print(f"Failed to remove image {img_path}: {e}")
        else:
            print(f"Image not found: {img_path}")

    print("Black images removal completed.")

    # Train the deep learning model with data augmentation
    X_train, y_train, age_train, quality_train, X_test, y_test, age_test, quality_test = train_model_with_augmentation(
        data_dir, num_epochs=1000, resume_training=True, 
        global_batch_size=global_batch_size, workers = 30, max_queue_size = 60
    )

    # Check sample labels
    # show_samples(X_train, y_train)
    # show_samples(X_test, y_test)

    # Evaluate the deep learning model
    latest_model = max(glob.glob('vgg16_model.h5'), key=os.path.getctime)
    evaluate_deep_learning_model(X_test, y_test, latest_model)
    
    # Train the decision tree model
    dt_model, X_test_features = train_decision_tree(X_train, y_train, age_train, quality_train, X_test, y_test, age_test, quality_test, 'vgg16_model.h5')

    # Evaluate the decision tree model
    evaluate_decision_tree_model(X_test, y_test, age_test, quality_test, X_test_features, dt_model)