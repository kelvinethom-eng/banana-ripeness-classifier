import sys
import numpy as np
import tensorflow as tf
from PIL import Image

# Load trained model
MODEL_PATH = "banana_ripeness_detector.keras"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["overripe", "ripe", "rotten", "unripe"]


def predict_ripeness(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)

    predictions = model.predict(img_array)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100

    print(f"\nImage: {image_path}")
    print(f"Result: {CLASS_NAMES[predicted_idx]}")
    print(f"Confidence: {confidence:.2f}%\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_ripeness(sys.argv[1])
    else:
        print("Usage: python predict.py <path_to_image.jpg>")