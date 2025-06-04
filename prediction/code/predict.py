import os
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image
import streamlit as st

# ============ Image Preprocessing ============ #
def preprocess_image_custom(image, img_size=224, is_training=False):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if is_training:
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

def parse_image(img_path, img_size=224):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = preprocess_image_custom(img, img_size=img_size, is_training=False)
    return img

# ============ Load Model ============ #
@st.cache_resource
def load_model(model_dir):
    model = tf.keras.models.load_model(model_dir)
    return model

# ============ App UI ============ #
st.title("Embryo Prediction Interface")
model_path = st.text_input("Enter model path:", "/Users/lzz/Documents/GitHub/IVF_AI/prediction/code/vgg16_best_model")
image_dir = st.text_input("Enter image directory:", "/Users/lzz/Documents/GitHub/IVF_AI/prediction/selected_images")

if st.button("Run Predictions"):
    if not os.path.isdir(image_dir):
        st.error("Image directory not found.")
    elif not os.path.exists(model_path):
        st.error("Model path not found.")
    else:
        model = load_model(model_path)
        results = []

        image_files = []
        for root, _, files in os.walk(image_dir):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, filename))

        st.success(f"Found {len(image_files)} image(s).")

        for i, image_path in enumerate(image_files):
            try:
                img_tensor = parse_image(image_path)
                img_tensor = tf.expand_dims(img_tensor, axis=0)
                pred = model.predict(img_tensor, verbose=0)[0]
                pred_class = int(np.argmax(pred))
                pred_prob = float(pred[pred_class])
                focal_depth = os.path.basename(os.path.dirname(image_path))

                results.append({
                    "filename": os.path.basename(image_path),
                    "relative_path": os.path.relpath(image_path, image_dir),
                    "predicted_class": pred_class,
                    "probability": pred_prob,
                    "focal_depth": focal_depth
                })

                # Display image and prediction
                st.image(image_path, width=256, caption=f"{os.path.basename(image_path)}")
                st.write(f"**Predicted Class**: {pred_class}")
                st.write(f"**Probability**: {pred_prob:.4f}")
                st.write(f"**Focal Depth**: {focal_depth}")
                st.markdown("---")

            except Exception as e:
                st.error(f"Error reading image {image_path}: {e}")

        if results:
            df = pd.DataFrame(results)
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "prediction_results.csv", "text/csv")