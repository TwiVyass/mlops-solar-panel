import os
import cv2
from scripts.utils import load_autoencoder, preprocess_image
from scripts.calculate_squiggliness import calculate_damage_score

def analyze_damaged_images(damaged_folder, threshold=10000):
    autoencoder = load_autoencoder()
    major_scores, minor_scores = [], []

    for root, _, files in os.walk(damaged_folder):
        for filename in files:
            img_path = os.path.join(root, filename)
            img_input = preprocess_image(img_path)
            if img_input is None:
                continue

            reconstructed = autoencoder.predict(img_input, verbose=0)[0]
            reconstructed = (reconstructed * 255).astype('uint8')
            _, thresh = cv2.threshold(reconstructed, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            score = calculate_damage_score(contours)
            (major_scores if score > threshold else minor_scores).append(score)

    return major_scores, minor_scores

if __name__ == "__main__":
    folder = "data/major_damage"
    major, minor = analyze_damaged_images(folder)
    print(f"Processed {len(major)+len(minor)} images.")
