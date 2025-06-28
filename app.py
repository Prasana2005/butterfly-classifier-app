import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pickle

# Load the trained model
model = load_model("Butterfly_classification.keras")

# Load class indices and create reverse mapping
with open("class_indices.pkl", "rb") as f:
    class_indices = pickle.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

# Streamlit UI
st.set_page_config(page_title="Butterfly Classifier", layout="centered")
st.title("🦋 Butterfly Species Classifier")
st.markdown("Upload a butterfly image to identify its species.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = index_to_class[predicted_index]
    confidence = np.max(prediction) * 100

    # Set a confidence threshold (you can adjust this)
    threshold = 60.0

    if confidence >= threshold:
        st.success(f"🎯 Predicted Species: **{predicted_class}**")
        st.info(f"Confidence: {confidence:.2f}%")
    else:
        st.error("🚫 This image is likely **not a butterfly**, or the model is unsure.")
        st.info(f"Prediction Confidence: {confidence:.2f}% (Below threshold)")
