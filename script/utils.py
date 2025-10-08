import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

def load_autoencoder(model_path="models/autoencoder_model.h5"):
    custom_objects = {"mse": MeanSquaredError()}
    model = load_model(model_path, custom_objects=custom_objects)
    print("✅ Autoencoder model loaded successfully!")
    return model

def preprocess_image(img_path, target_size=(128, 128)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img_resized = cv2.resize(img, target_size) / 255.0
    img_input = np.expand_dims(img_resized, axis=(0, -1))
    return img_input
