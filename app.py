import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# -------------------------------
# Load Autoencoder Model
# -------------------------------
autoencoder = load_model("models/autoencoder_model.h5")  # <<-- MAKE SURE FILE NAME MATCHES

adaptive_threshold = 0.085  # your chosen threshold

# -------------------------------
# Preprocessing for Autoencoder
# -------------------------------
def preprocess_for_autoencoder(image):
    img = cv2.resize(np.array(image), (128, 128)).astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

# -------------------------------
# Autoencoder Classification
# -------------------------------
def classify_image(image):
    processed_image = preprocess_for_autoencoder(image)
    reconstructed_image = autoencoder.predict(processed_image)

    error = np.mean(np.abs(processed_image - reconstructed_image))
    classification = "Dusty" if error > adaptive_threshold else "Clean"

    return processed_image[0], reconstructed_image[0], classification, error

# -------------------------------
# Show Contours (Optional Visualization)
# -------------------------------
def display_image_with_contours(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoured_img = image.copy()
    cv2.drawContours(contoured_img, contours, -1, (0, 255, 0), 2)
    return contoured_img

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Solar Panel Cleanliness Classifier (Autoencoder Only)")
st.write("Upload an image to determine whether the solar panel is **Clean** or **Dusty**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image_np, caption="Uploaded Image", use_container_width=True)

    # Run autoencoder classification
    original, reconstructed, classification, error = classify_image(image_np)

    st.write(f"### **Prediction:** {classification}")
    st.write(f"Reconstruction Error: `{error:.4f}`")

    # Display reconstructed image
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Original (128×128)", use_container_width=True)
    with col2:
        st.image(reconstructed, caption="Reconstructed (128×128)", use_container_width=True)

    # Display contours visualization
    contoured_img = display_image_with_contours(image_np)
    st.image(contoured_img, caption="Detected Surface Edges / Contours", use_container_width=True)
