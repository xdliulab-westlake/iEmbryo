import os
import re
import math
import time
import glob
import copy
import gc
import joblib
import shutil
import concurrent.futures
from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Input, GlobalAveragePooling2D, Dense, Dropout, Flatten
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications import InceptionV3, EfficientNetB0, ResNet50
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import absl.logging
from preprocessing.data_loader import *
from preprocessing.image_utils import copy_images_by_split
from utils.data_utils import balance_classes, check_class_distribution

# Build a pre-trained VGG16 model with custom top layers
def build_vgg16_model(input_shape=(224, 224, 3), num_classes=2, fine_tune_at=18, fine_tune=False):
    # Load the VGG16 model without the top layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    if fine_tune:
        for layer in base_model.layers[:]:
            layer.trainable = True
        # Freeze the base model layers up to the specified layer
        # for layer in base_model.layers[fine_tune_at:]:
        #     layer.trainable = True
    else:
        base_model.trainable = False

    # Create the new model on top
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
  
    # Global average pooling
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.0002))(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.0002))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model with a learning rate schedule
    optimizer = Adam(learning_rate=1e-4, weight_decay=1e-5)
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# check gpu availability
def check_gpu_availability():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # Print detailed information about each device
    for device in tf.config.list_physical_devices():
        print(device)

# Training the model with augmentation using data generators
def train_model_with_data_generator(data_dir, num_epochs=3, global_batch_size=None, img_size=224, resume_training=True, fine_tune=False):
    # Set strategy for distributed training
    strategy = tf.distribute.MirroredStrategy()
    
    start_time = time.time()
    
    # Check GPU availability
    check_gpu_availability()
    
    # Create data generators for memory-efficient training
    # 70% training, 15% validation, 15% test split
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.3  # 30% for validation+test (will split this further)
    )
    
    val_test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.3
    )
    
    # Create training generator (70% of data)
    train_generator = train_datagen.flow_from_directory(
        data_dir,  # Use main data directory
        target_size=(img_size, img_size),
        batch_size=global_batch_size,
        class_mode='sparse',
        subset='training',  # This gives us 70%
        shuffle=True,
        seed=42
    )
    
    # Create validation+test generator (30% of data)
    val_test_generator = val_test_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=global_batch_size,
        class_mode='sparse',
        subset='validation',  # This gives us 30%
        shuffle=False,
        seed=42
    )
    
    # Split the 30% into 15% validation and 15% test
    # We'll use the first half for validation, second half for test
    val_samples = val_test_generator.samples // 2
    test_samples = val_test_generator.samples - val_samples
    
    # Create separate validation and test generators
    val_generator = val_test_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=global_batch_size,
        class_mode='sparse',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    # We'll create a test generator later for evaluation
    
    # Calculate actual split percentages
    total_samples = train_generator.samples + val_test_generator.samples
    train_pct = train_generator.samples / total_samples * 100
    val_pct = val_samples / total_samples * 100
    test_pct = test_samples / total_samples * 100
    
    print(f"Data split:")
    print(f"  Train: {train_generator.samples} samples ({train_pct:.1f}%)")
    print(f"  Validation: {val_samples} samples ({val_pct:.1f}%)")
    print(f"  Test: {test_samples} samples ({test_pct:.1f}%)")
    print(f"  Total: {total_samples} samples")
    print(f"Classes found: {train_generator.class_indices}")

    # Define the model path and callbacks
    model_path = 'vgg16_best_model'  # Save only the best model based on validation loss
    initial_epoch = 0
    csv_logger = CSVLogger('training.log', append=True)
    
    # Define checkpoint callback to save only the best model (based on validation loss)
    checkpoint_callback = ModelCheckpoint(
        filepath=model_path,
        save_weights_only=False,  # Save the full model (architecture + weights)
        monitor='val_loss',  # Monitor validation loss
        mode='min',  # Save when validation loss is minimized
        save_best_only=True  # Save only the best model
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // global_batch_size
    validation_steps = val_generator.samples // global_batch_size

    with strategy.scope():
        model = build_vgg16_model((img_size, img_size, 3), num_classes=2)

        if resume_training and os.path.exists(model_path):
            print("Loading existing model from", model_path)
            model = tf.keras.models.load_model(model_path)

            # if fine_tune:
            #     print("Fine-tuning enabled: Unfreezing layers starting from", fine_tune_at)
            #     for layer in model.layers[:fine_tune_at]:
            #         layer.trainable = False
            #     for layer in model.layers[fine_tune_at:]:
            #         layer.trainable = True
            # else:
            #     print("Fine-tuning disabled: Keeping all previously frozen layers.")

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
        
        # Train the model with data generators
        try:
            print(f"Final batch size: {global_batch_size} ({global_batch_size // strategy.num_replicas_in_sync} per GPU)")
            print(f"Total global batch size: {global_batch_size}")
            print(f"Train samples: {train_generator.samples} | Batch size: {global_batch_size}")
            print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")
            
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=num_epochs,
                initial_epoch=initial_epoch,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=[csv_logger, checkpoint_callback, early_stopping, reduce_lr]
            )

            # Check if best model was saved; if not, save the current model
            # (inserted as per instructions)
            if os.path.exists(model_path):
                print(f"Best model saved at: {model_path}")
            else:
                print(f"No best model found at {model_path}, saving final model manually.")
                model.save(model_path)
                print(f"Final model saved at: {model_path}")

            # Calculate the time consumed
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Total training time: {elapsed_time:.2f} seconds")

        except Exception as e:
            print(f"Training was interrupted due to an error: {e}")
            # Optionally save the model in case of interruption (though not needed if only saving the best model)
            model.save('vgg16_model_at_interruption')
            print("Model saved at interruption point.")

    # Create a test generator for final evaluation
    def create_test_generator(data_dir, img_size, batch_size):
        test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)
        test_gen = test_datagen.flow_from_directory(
            data_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='sparse',
            subset='validation',
            shuffle=False,
            seed=42
        )
        # Skip first half (validation), return second half (test)
        test_gen.batch_index = test_gen.samples // 2 // batch_size
        return test_gen
    
    return train_generator, val_generator, create_test_generator

# 4. Evaluate Deep Learning Model
def evaluate_deep_learning_model(X_eval, y_eval, model_path, dataset_name="Test", save_path="auc_curve.pdf"):
    """
    Evaluate a deep learning model on a given dataset and save the ROC AUC curve as a PDF.
    
    Parameters:
    - X_eval: Input features for evaluation.
    - y_eval: True labels for evaluation.
    - model_path: Path to the trained model.
    - dataset_name: Name of the dataset (e.g., "Test" or "Validation").
    - save_path: Path to save the ROC AUC plot as a PDF.
    """
    
    # Load model
    dl_model = tf.keras.models.load_model(model_path)
    
    # Ensure evaluation is done using the same global batch size as training with proper sharding for multi-GPU
    eval_batch_size = min(global_batch_size, 128)  # Ensure batch size is not too large for evaluation
    eval_dataset = tf.data.Dataset.from_tensor_slices((X_eval, y_eval)).batch(eval_batch_size).prefetch(tf.data.AUTOTUNE)

    # Evaluate model
    eval_loss, eval_accuracy = dl_model.evaluate(eval_dataset)
    print(f"Deep Learning Model {dataset_name} Loss: {eval_loss:.4f}, {dataset_name} Accuracy: {eval_accuracy:.4f}")
    
    # Convert one-hot labels to integer labels if necessary
    if y_eval.ndim > 1 and y_eval.shape[1] == 2:
        y_eval = np.argmax(y_eval, axis=1)
    # Compute AUC for model
    probabilities_dl = dl_model.predict(eval_dataset)
    
    if len(np.unique(y_eval)) < 2:
        print(f"Only one class present in {dataset_name} set. ROC AUC score is not defined.")
    else:
        auc_dl = roc_auc_score(y_eval, probabilities_dl[:, 1])
        print(f"Deep Learning Model {dataset_name} AUC: {auc_dl:.4f}")
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_eval, probabilities_dl[:, 1])
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_dl:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal reference line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{dataset_name} Set ROC Curve')
        plt.legend(loc="lower right")
        
        # Save the figure as a PDF
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"ROC AUC curve saved as {save_path}")
        plt.close()

    # Compute confusion matrix
    y_pred_dl = (probabilities_dl > 0.5).astype(int).flatten()
    cm_dl = confusion_matrix(y_eval, y_pred_dl)
    
    # Create confusion matrix DataFrame
    class_names = ['Class 1', 'Class 0']
    cm_df = pd.DataFrame(cm_dl, index=class_names, columns=class_names)
    print(f"Deep Learning Model {dataset_name} Confusion Matrix:")
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
    # Convert one-hot labels to integer labels if necessary
    if y_train.ndim > 1 and y_train.shape[1] == 2:
        y_train = np.argmax(y_train, axis=1)
    if y_test.ndim > 1 and y_test.shape[1] == 2:
        y_test = np.argmax(y_test, axis=1)
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
    # Convert one-hot labels to integer labels if necessary
    if y_test.ndim > 1 and y_test.shape[1] == 2:
        y_test = np.argmax(y_test, axis=1)
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
    os.chdir("/liaozizhuo/scripts")
    data_dir = '/liaozizhuo/selected_images/'

    # Get all available GPUs
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Set memory growth to avoid TensorFlow allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU {gpu} is set to memory growth.")
        except RuntimeError as e:
            print(f"Error in setting memory growth: {e}")
    
    # Additional TensorFlow memory optimization
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # Define the strategy for multi-GPU training
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    batch_size_per_replica = 64
    global_batch_size = batch_size_per_replica * max(1, num_gpus)

    # Run dataset_prepration.py
    # with open('/Users/lzz/Documents/GitHub/IVF_AI/prediction/code/dataset_preparation_blastocyst.py') as f:
    #     code = f.read()
    # exec(code)

    # Kmeans.py

    # remove black images    
    # csv_path = "/liaozizhuo/selected_images/images_train/prediction_Kmeans_F0_all.csv"
    # df = pd.read_csv(csv_path)

    # black_images = df[df['Cluster Label'] == 1]
    # filtered_df = df[df['Cluster Label'] != 1]

    # output_filtered_csv_path = "/liaozizhuo/selected_images/images_train/prediction_Kmeans_F0_filtered.csv"
    # filtered_df.to_csv(output_filtered_csv_path, index=False)
    # print(f"Filtered CSV saved to {output_filtered_csv_path}")

    # for index, row in black_images.iterrows():
    #     img_path = row['Absolute Path']
    #     if os.path.exists(img_path):
    #         try:
    #             os.remove(img_path)
    #         except Exception as e:
    #             print(f"Failed to remove image {img_path}: {e}")
    #     else:
    #         print(f"Image not found: {img_path}")

    # print("Black images removal completed.")

    # Train the model with memory-efficient data generators
    train_generator, val_generator, create_test_generator = train_model_with_data_generator(
        data_dir,
        num_epochs=2000, 
        global_batch_size=global_batch_size, 
        img_size=224, 
        resume_training=False,
        fine_tune=True
    )


    # show_samples(X_train, y_train)
    # show_samples(X_test, y_test)

    # Create test generator and evaluate the model
    test_generator = create_test_generator(data_dir, 224, global_batch_size)
    latest_model_path = 'vgg16_best_model'
    if os.path.exists(latest_model_path):
        print(f"Evaluating model: {latest_model_path}")
        
        # Load and evaluate model with test generator
        model = tf.keras.models.load_model(latest_model_path)
        test_loss, test_accuracy = model.evaluate(test_generator)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        # Generate predictions for ROC curve
        predictions = model.predict(test_generator)
        # Note: You'll need to collect true labels from the generator for full evaluation
    else:
        print("No model found at the specified path.")
                                   
    # Note: Decision tree training commented out due to data generator format
    # You'll need to adapt the decision tree code to work with generators
    # dt_model, X_test_features = train_decision_tree(...)
    # evaluate_decision_tree_model(...)
