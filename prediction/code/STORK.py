import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from PIL import Image
import joblib

# 1. Data Loading and Preprocessing
def load_data(data_dir, img_size=299):
    """
    Load the dataset.
    
    Parameters:
    - data_dir: Path to the dataset directory
    - img_size: Image size, default is 299x299
    
    Returns:
    - images: Array of image data (N, 299, 299, 3)
    - labels: Array of labels (N,)
    - ages: Array of ages (N,)
    """
    images = []
    labels = []
    ages = []
    
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
    
    images = np.array(images)
    labels = np.array(labels)
    ages = np.array(ages)
    
    return images, labels, ages

# 2. Build Deep Learning Model
def build_model(input_shape, num_classes):
    """
    Build the deep learning model.
    
    Parameters:
    - input_shape: Input shape of the images (height, width, channels)
    - num_classes: Number of classes
    
    Returns:
    - model: Compiled Keras model
    """
    # Image input
    image_input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    
    # Fully connected layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=image_input, outputs=x)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# 3. Train Deep Learning Model
def train_model(data_dir, num_epochs=10, batch_size=32, img_size=299):
    """
    Train the deep learning model.
    
    Parameters:
    - data_dir: Path to the dataset directory
    - num_epochs: Number of training epochs
    - batch_size: Batch size
    - img_size: Image size
    """
    # Load data
    images, labels, ages = load_data(data_dir, img_size)
    
    # Split the dataset with stratification
    X_train, X_temp, y_train, y_temp, age_train, age_temp = train_test_split(
        images, labels, ages, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test, age_val, age_test = train_test_split(
        X_temp, y_temp, age_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Check class distribution
    def check_class_distribution(y_train, y_val, y_test):
        print("Train set class distribution:", np.bincount(y_train))
        print("Validation set class distribution:", np.bincount(y_val))
        print("Test set class distribution:", np.bincount(y_test))
    
    check_class_distribution(y_train, y_val, y_test)
    
    # Define the model
    model = build_model((img_size, img_size, 3), num_classes=2)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=num_epochs,
        batch_size=batch_size
    )
    
    # Save the model
    model.save('model.h5')
    print("Deep Learning Model saved as model.h5")
    
    return X_train, y_train, age_train, X_test, y_test, age_test

# 4. Build Decision Tree Model
def build_decision_tree(X_train, y_train, age_train):
    """
    Build the decision tree model.
    
    Parameters:
    - X_train: Training image features
    - y_train: Training labels
    - age_train: Training ages
    
    Returns:
    - model: Trained decision tree model
    """
    # Combine image features and age into a single feature matrix
    X_train_combined = np.hstack((X_train, age_train.reshape(-1, 1)))
    
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

# 5. Train Decision Tree Model
def train_decision_tree(X_train, y_train, age_train, X_test, y_test, age_test):
    """
    Train the decision tree model.
    
    Parameters:
    - X_train: Training image features
    - y_train: Training labels
    - age_train: Training ages
    - X_test: Test image features
    - y_test: Test labels
    - age_test: Test ages
    """
    # Load the deep learning model
    dl_model = tf.keras.models.load_model('model.h5')
    
    # Get the image features from the deep learning model
    intermediate_layer_model = Model(inputs=dl_model.input, outputs=dl_model.layers[-2].output)
    X_train_features = intermediate_layer_model.predict(X_train)
    X_test_features = intermediate_layer_model.predict(X_test)
    
    # Train the decision tree model
    dt_model = build_decision_tree(X_train_features, y_train, age_train)
    
    # Save the decision tree model
    joblib.dump(dt_model, 'decision_tree_model.pkl')
    print("Decision Tree Model saved as decision_tree_model.pkl")
    
    return dt_model, X_test_features

# 6. Evaluate Models
def evaluate_models(X_test, y_test, age_test, X_test_features, dt_model):
    """
    Evaluate the deep learning and decision tree models.
    
    Parameters:
    - X_test: Test image data
    - y_test: Test labels
    - age_test: Test ages
    - X_test_features: Test image features from the deep learning model
    - dt_model: Trained decision tree model
    """
    # Load the deep learning model
    dl_model = tf.keras.models.load_model('model.h5')
    
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
    
    # Combine image features and age into a single feature matrix
    X_test_combined = np.hstack((X_test_features, age_test.reshape(-1, 1)))
    
    # Evaluate the decision tree model
    dt_predictions = dt_model.predict(X_test_combined)
    dt_accuracy = np.mean(dt_predictions == y_test)
    print(f"Decision Tree Model Test Accuracy: {dt_accuracy:.4f}")
    
    # Compute AUC for decision tree model
    probabilities_dt = dt_model.predict_proba(X_test_combined)[:, 1]
    if len(np.unique(y_test)) < 2:
        print("Only one class present in y_test. ROC AUC score is not defined in that case for the Decision Tree Model.")
    else:
        auc_dt = roc_auc_score(y_test, probabilities_dt)
        print(f"Decision Tree Model Test AUC: {auc_dt:.4f}")

# Main function
if __name__ == "__main__":
    data_dir = '/Users/lzz/Documents/GitHub/IVF_AI/prediction/images'
    
    # Train the deep learning model
    X_train, y_train, age_train, X_test, y_test, age_test = train_model(data_dir)
    
    # Train the decision tree model
    dt_model, X_test_features = train_decision_tree(X_train, y_train, age_train, X_test, y_test, age_test)
    
    # Evaluate the models
    evaluate_models(X_test, y_test, age_test, X_test_features, dt_model)