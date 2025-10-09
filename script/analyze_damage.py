import tensorflow as tf
import cv2
import numpy as np
import glob
from scipy import stats
import os



# Load trained model
autoencoder = tf.keras.models.load_model(
    "models/autoencoder_model.h5",
    custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
)

# Path to damaged images
damaged_folder = "data/raw/Faulty_solar_panel/Physical-Damage"

def calculate_damage_score(contours):
    if not contours:
        return 0
    squiggliness_score = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area > 10:
            squiggliness_score += perimeter / area
    return squiggliness_score

def process_image_with_autoencoder(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img_resized = cv2.resize(img, (128, 128)) / 255.0
    img_input = np.expand_dims(img_resized, axis=(0, -1))
    reconstructed = autoencoder.predict(img_input, verbose=0)[0].squeeze()

    # Compute reconstruction difference (this highlights damage)
    diff = np.abs(img_resized - reconstructed)

    # Scale back to 0–255
    diff_uint8 = (diff * 255).astype(np.uint8)

    # Use adaptive or Otsu thresholding for more sensitivity
    _, thresh = cv2.threshold(diff_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return calculate_damage_score(contours)

# Analyze damaged images
scores = []
for img_path in glob.glob(os.path.join(damaged_folder, "*.jpg")):
    score = process_image_with_autoencoder(img_path)
    if score is not None:
        scores.append(score)

print(f"✅ Processed {len(scores)} images.")
print("Average squiggliness score:", np.mean(scores))
