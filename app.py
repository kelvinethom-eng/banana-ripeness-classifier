import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Banana Ripeness Detector", page_icon="🍌")

st.title("🍌 Banana Ripeness Classifier")
st.write(
    "Upload a banana photo to detect its ripeness stage: **Unripe**, **Ripe**, **Overripe**, or **Rotten**."
)


# Load model with caching to prevent reloading on every run
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("banana_ripeness_detector.keras")


model = load_model()
CLASS_NAMES = ["overripe", "ripe", "rotten", "unripe"]

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        # Preprocess image to match training pipeline
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]) * 100)

    st.success(
        f"**Prediction:** {predicted_class.upper()} ({confidence:.2f}% confidence)"
    )

    # Display breakdown
    st.subheader("Confidence Scores")
    for cls, prob in zip(CLASS_NAMES, predictions[0]):
        st.write(f"- **{cls.capitalize()}**: {prob*100:.2f}%")