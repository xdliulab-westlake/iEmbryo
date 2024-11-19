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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# 1. Data Loading and Preprocessing
def load_data(data_dir, img_size=299):
    images = []
    labels = []
    ages = []
    qualities = []  # Embryo quality labels
    
    for label, class_name in enumerate(['class_0', 'class_1']):
        class_dir = os.path.join(data_dir, 'train', class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = Image.open(img_path).resize((img_size, img_size))
            img_array = np.array(img) / 255.0  # Normalize
            if img_array.ndim == 2:  # If grayscale, convert to RGB
                img_array = np.stack((img_array,) * 3, axis=-1)
            images.append(img_array)
            labels.append(label)
            
            # Add age information
            age = int(re.search(r'_RUN\d+_(\d+)\.jpeg$', img_file).group(1))
            ages.append(age)
            
            # Add quality information
            quality = 1 if class_name == 'class_1' else 0  # Set quality based on class name
            qualities.append(quality)
    
    images = np.array(images)
    labels = np.array(labels)
    ages = np.array(ages)
    qualities = np.array(qualities)  # Embryo quality labels
    
    return images, labels, ages, qualities

# 2. Build Deep Learning Model
def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)

    conv_3x3_reduce = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(conv_3x3_reduce)

    conv_5x5_reduce = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(conv_5x5_reduce)

    maxpool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu')(maxpool)

    output = Concatenate()([conv_1x1, conv_3x3, conv_5x5, pool_proj])
    return output

def build_inception_v1(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Initial convolution
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Inception modules
    x = inception_module(x, 64, 96, 128, 16, 32, 32)
    x = inception_module(x, 128, 128, 192, 32, 96, 64)

    # Reduction
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # More Inception modules
    x = inception_module(x, 192, 96, 208, 16, 48, 64)
    x = inception_module(x, 160, 112, 224, 24, 64, 64)
    x = inception_module(x, 128, 128, 256, 24, 64, 64)
    x = inception_module(x, 112, 144, 288, 32, 64, 64)
    x = inception_module(x, 256, 160, 320, 32, 128, 128)

    # Reduction
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Final Inception module
    x = inception_module(x, 256, 160, 320, 32, 128, 128)
    x = inception_module(x, 384, 192, 384, 48, 128, 128)

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Fully connected layers
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# 3. Train Deep Learning Model with Data Augmentation
def train_model_with_augmentation(data_dir, num_epochs=20, batch_size=32, img_size=299):
     # Load data
    images, labels, ages, qualities = load_data(data_dir, img_size)
    
    # Split the dataset with stratification
    X_train, X_temp, y_train, y_temp, age_train, age_temp, quality_train, quality_temp = train_test_split(
        images, labels, ages, qualities, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test, age_val, age_test, quality_val, quality_test = train_test_split(
        X_temp, y_temp, age_temp, quality_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Check class distribution
    def check_class_distribution(y_train, y_val, y_test):
        print("Train set class distribution:", np.bincount(y_train))
        print("Validation set class distribution:", np.bincount(y_val))
        print("Test set class distribution:", np.bincount(y_test))
    
    check_class_distribution(y_train, y_val, y_test)
    
    # Define the model
    model_path = 'inception_v1_model.h5'
    initial_epoch = 0
    
    # Check if there is an existing model
    if os.path.exists(model_path):
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
        model = build_inception_v1((img_size, img_size, 3), num_classes=2)
    
    # Define CSV logger to log training history
    csv_logger = CSVLogger('training.log', append=True)
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,  # Random rotation
        width_shift_range=0.1,  # Width shift
        height_shift_range=0.1,  # Height shift
        shear_range=0.1,  # Shear intensity
        zoom_range=0.1,  # Zoom range
        horizontal_flip=True,  # Horizontal flip
        fill_mode='nearest'  # Fill mode
    )
    
    # Train the model with data augmentation
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=num_epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=len(X_train) // batch_size,
        callbacks=[csv_logger]
    )
    
    # Save the model
    model.save(model_path)
    print("Inception-V1 Model saved as", model_path)
    
    return X_train, y_train, age_train, quality_train, X_test, y_test, age_test, quality_test

# 4. Evaluate Deep Learning Model
def evaluate_deep_learning_model(X_test, y_test, model_path):
    # Load the deep learning model
    dl_model = tf.keras.models.load_model(model_path)
    
    # Evaluate the deep learning model
    test_loss, test_accuracy = dl_model.evaluate(X_test, y_test)
    print(f"Deep Learning Model Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Compute AUC for deep learning model
    probabilities_dl = dl_model.predict(X_test)
    if len(np.unique(y_test)) < 2:
        print("Only one class present in y_test. ROC AUC score is not defined in that case for the Deep Learning Model.")
    else:
        auc_dl = roc_auc_score(y_test, probabilities_dl[:, 1])
        print(f"Deep Learning Model Test AUC: {auc_dl:.4f}")
        
        # Plot ROC curve
        # fpr, tpr, _ = roc_curve(y_test, probabilities_dl[:, 1])
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
    print("Deep Learning Model Confusion Matrix:")
    print(cm_dl)

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

# Main function
if __name__ == "__main__":
    data_dir = '/Users/lzz/Documents/GitHub/IVF_AI/prediction/images'

    # Run dataset_prepration.py
    with open('/Users/lzz/Documents/GitHub/IVF_AI/prediction/code/dataset_preparation.py') as f:
        code = f.read()
    exec(code)

    # Train the deep learning model with data augmentation
    X_train, y_train, age_train, quality_train, X_test, y_test, age_test, quality_test = train_model_with_augmentation(data_dir)
    
    # Evaluate the deep learning model
    latest_model = max(glob.glob('inception_v1_model.h5'), key=os.path.getctime)
    evaluate_deep_learning_model(X_test, y_test, latest_model)
    
    # Train the decision tree model
    dt_model, X_test_features = train_decision_tree(X_train, y_train, age_train, quality_train, X_test, y_test, age_test, quality_test, 'deep_learning_model.h5')

    # Evaluate the decision tree model
    evaluate_decision_tree_model(X_test, y_test, age_test, quality_test, X_test_features, dt_model)