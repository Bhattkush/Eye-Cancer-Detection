import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("model/eye_cancer_model.h5")

st.title("👁️ AI-Based Eye Cancer Detection")

uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.reshape(img, (1,224,224,3))

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        st.error("⚠️ Eye Cancer Detected")
    else:
        st.success("✅ Normal Eye")
