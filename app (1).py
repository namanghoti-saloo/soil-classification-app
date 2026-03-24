
import streamlit as st
import numpy as np
from PIL import Image
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# 🔥 Google Drive file ID
file_id = "12H9nBcbESdRYwy08Io-vAAEKdZxpbV1K"
url = f"https://drive.google.com/uc?id={file_id}"

# Download model if not exists
if not os.path.exists("soil_model.keras"):
    gdown.download(url, "soil_model.keras", quiet=False)

# Load model
model = load_model("soil_model.keras")

class_labels = ['Alluvial Soil', 'Black Soil', 'Clay Soil', 'Red Soil']

def predict_image(img):
    img = img.resize((224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    return predicted_class

st.title("🌱 Soil Classification App")

uploaded_file = st.file_uploader("Upload Soil Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img)

    result = predict_image(img)
    st.success(f"Prediction: {result}")
