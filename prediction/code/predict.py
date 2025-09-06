import os
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image
import streamlit as st
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import io
import base64

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

# ============ Grad-CAM Generation ============ #
def generate_gradcam(model, image, class_index, layer_name='block5_conv3'):
    """
    Generate Grad-CAM heatmap for attention visualization
    """
    try:
        # Create model that outputs target layer and predictions
        grad_model = Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # Calculate gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            if predictions.shape[1] == 2:
                loss = predictions[:, class_index]
            else:
                loss = predictions[:, 0] if class_index == 0 else (1 - predictions[:, 0])
        
        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Generate heatmap
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to image size
        heatmap = cv2.resize(heatmap, (224, 224))
        
        # Apply colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return heatmap
    except Exception as e:
        st.warning(f"Could not generate Grad-CAM: {e}")
        return None

# ============ App UI ============ #
st.set_page_config(page_title="Embryo Prediction Interface", layout="wide")
st.title("🔬 Embryo Prediction Interface")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    model_path = st.text_input("Model Path:", 
                              value="/Users/lzz/Documents/GitHub/IVF_AI/prediction/code/vgg16_best_model")
    image_dir = st.text_input("Image Directory:", 
                             value="/Users/lzz/Documents/GitHub/IVF_AI/prediction/selected_images")
    
    # Options
    st.header("Options")
    generate_heatmaps = st.checkbox("Generate Attention Heatmaps", value=True)
    max_images = st.slider("Max images to process", min_value=1, max_value=50, value=10)

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["📊 Batch Prediction", "🎯 Single Prediction", "📈 Results Analysis"])

with tab1:
    st.header("Batch Prediction with Attention Heatmaps")
    
    if st.button("🚀 Run Batch Predictions", type="primary"):
        if not os.path.isdir(image_dir):
            st.error("Image directory not found.")
        elif not os.path.exists(model_path):
            st.error("Model path not found.")
        else:
            # Load model
            with st.spinner("Loading model..."):
                model = load_model(model_path)
            st.success("Model loaded successfully!")
            
            # Collect image files
            image_files = []
            for root, _, files in os.walk(image_dir):
                for filename in files:
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_files.append(os.path.join(root, filename))
            
            image_files = image_files[:max_images]  # Limit number of images
            st.info(f"Processing {len(image_files)} image(s)...")
            
            results = []
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process images in batches of 3 for display
            for batch_idx in range(0, len(image_files), 3):
                batch_files = image_files[batch_idx:batch_idx+3]
                
                # Create two columns for better layout
                col1, col2 = st.columns(2)
                
                for idx, image_path in enumerate(batch_files):
                    try:
                        # Update progress
                        progress = (batch_idx + idx + 1) / len(image_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing: {os.path.basename(image_path)}")
                        
                        # Process image
                        img_tensor = parse_image(image_path)
                        img_tensor = tf.expand_dims(img_tensor, axis=0)
                        pred = model.predict(img_tensor, verbose=0)[0]
                        pred_class = int(np.argmax(pred))
                        pred_prob = float(pred[pred_class])
                        focal_depth = os.path.basename(os.path.dirname(image_path))
                        
                        # Load original image for display
                        original_img = Image.open(image_path)
                        original_img = original_img.resize((224, 224))
                        
                        # Determine which column to use (alternate for better distribution)
                        current_col = col1 if idx % 2 == 0 else col2
                        
                        with current_col:
                            # Create expandable container for each image
                            with st.expander(f"📷 {os.path.basename(image_path)} - {['Poor', 'Good'][pred_class]} ({pred_prob:.3f})", expanded=True):
                                
                                if generate_heatmaps:
                                    # Create sub-columns for original and heatmap
                                    img_col1, img_col2 = st.columns(2)
                                    
                                    with img_col1:
                                        st.image(original_img, caption="Original Image", use_column_width=True)
                                    
                                    with img_col2:
                                        # Generate heatmap
                                        heatmap = generate_gradcam(model, img_tensor, pred_class)
                                        if heatmap is not None:
                                            # Convert heatmap from BGR to RGB
                                            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                                            
                                            # Overlay heatmap on original image
                                            original_array = np.array(original_img)
                                            if original_array.max() <= 1.0:
                                                original_array = (original_array * 255).astype(np.uint8)
                                            
                                            # Blend heatmap with original
                                            overlay = cv2.addWeighted(original_array, 0.6, heatmap_rgb, 0.4, 0)
                                            
                                            st.image(overlay, caption="Attention Heatmap", use_column_width=True)
                                        else:
                                            st.info("Heatmap generation not available")
                                else:
                                    st.image(original_img, caption="Original Image", use_column_width=True)
                                
                                # Display prediction details
                                st.markdown(f"""
                                **Prediction Details:**
                                - **Class**: {['Poor Quality', 'Good Quality'][pred_class]}
                                - **Confidence**: {pred_prob:.4f}
                                - **Focal Depth**: {focal_depth}
                                - **File**: {os.path.basename(image_path)}
                                """)
                        
                        # Store results
                        results.append({
                            "filename": os.path.basename(image_path),
                            "relative_path": os.path.relpath(image_path, image_dir),
                            "predicted_class": pred_class,
                            "class_name": ['Poor', 'Good'][pred_class],
                            "probability": pred_prob,
                            "focal_depth": focal_depth
                        })
                        
                    except Exception as e:
                        st.error(f"Error processing {image_path}: {e}")
                
                # Add separator between batches
                if batch_idx + 3 < len(image_files):
                    st.markdown("---")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display summary
            if results:
                st.success(f"✅ Processed {len(results)} images successfully!")
                
                # Quick stats
                st.subheader("📊 Quick Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                df_results = pd.DataFrame(results)
                good_count = len(df_results[df_results['predicted_class'] == 1])
                poor_count = len(df_results[df_results['predicted_class'] == 0])
                avg_confidence = df_results['probability'].mean()
                
                with col1:
                    st.metric("Total Images", len(results))
                with col2:
                    st.metric("Good Quality", good_count)
                with col3:
                    st.metric("Poor Quality", poor_count)
                with col4:
                    st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                
                # Download results
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    "batch_prediction_results.csv",
                    "text/csv",
                    key="download_batch_results"
                )

with tab2:
    st.header("Single Image Prediction")
    
    uploaded_file = st.file_uploader("Choose an embryo image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None and os.path.exists(model_path):
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing image..."):
                # Save uploaded file temporarily
                temp_path = f"/tmp/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Load model and make prediction
                model = load_model(model_path)
                img_tensor = parse_image(temp_path)
                img_tensor = tf.expand_dims(img_tensor, axis=0)
                pred = model.predict(img_tensor, verbose=0)[0]
                pred_class = int(np.argmax(pred))
                pred_prob = float(pred[pred_class])
                
                with col2:
                    # Generate heatmap
                    heatmap = generate_gradcam(model, img_tensor, pred_class)
                    if heatmap is not None:
                        # Convert and overlay heatmap
                        original_array = np.array(image.resize((224, 224)))
                        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                        overlay = cv2.addWeighted(original_array, 0.6, heatmap_rgb, 0.4, 0)
                        st.image(overlay, caption="Attention Heatmap", use_column_width=True)
                    else:
                        st.info("Heatmap generation not available")
                
                # Display results
                st.subheader("📋 Prediction Results")
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.metric(
                        "Predicted Quality", 
                        ['Poor Quality', 'Good Quality'][pred_class],
                        delta=None
                    )
                
                with result_col2:
                    st.metric(
                        "Confidence Score", 
                        f"{pred_prob:.4f}",
                        delta=f"{(pred_prob - 0.5):.3f}" if pred_prob > 0.5 else f"{(pred_prob - 0.5):.3f}"
                    )
                
                # Clean up temp file
                os.remove(temp_path)

with tab3:
    st.header("📈 Results Analysis")
    st.info("This section will show detailed analysis of batch prediction results.")
    
    # Placeholder for future analysis features
    st.markdown("""
    **Available Analysis Features:**
    - Distribution of quality predictions
    - Confidence score analysis
    - Focal depth correlation
    - Batch processing statistics
    """)

# Footer
st.markdown("---")
st.markdown("*Embryo Quality Prediction System - Powered by Deep Learning*")