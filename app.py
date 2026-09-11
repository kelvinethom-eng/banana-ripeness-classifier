import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.title("🍌 Banana Ripeness Detector")

# Load model
model = tf.keras.models.load_model("banana_ripeness_detector.keras")
class_names = ["overripe", "ripe", "rotten", "unripe"]

uploaded_file = st.file_uploader(
    "Upload a banana photo...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess & predict
    img_resized = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized), axis=0)
    preds = model.predict(img_array)
    label = class_names[np.argmax(preds[0])]
    confidence = np.max(preds[0]) * 100

    st.success(f"**Prediction:** {label} ({confidence:.1f}%)")
