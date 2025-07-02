# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load model and class names
model = tf.keras.models.load_model("cifar10_model.h5")
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

st.title("🚀 CIFAR-10 Image Recognition App")
st.write("Upload an image and get prediction using a custom-trained CNN.")

uploaded_file = st.file_uploader("Upload Image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((32, 32))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]
    # Threshold for fallback (e.g., 60%)
    threshold = 0.60
    if confidence < threshold:
        st.markdown("### ❌ Prediction: **Unknown Object**")
        st.markdown(f"🧐 Model isn't confident enough (only `{confidence*100:.2f}%`) that this image belongs to any of the 10 CIFAR-10 categories.")
        st.info("Try uploading an image of a common object like a cat, ship, or airplane.")
    else:
        st.markdown(f"### ✅ Prediction: **{class_names[class_index]}**")
        st.markdown(f"Confidence: `{confidence * 100:.2f}%`")
