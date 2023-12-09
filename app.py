import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)



# Load the model
model = load_model("keras_model.h5", compile=False)


# Load the labels
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

def predict(image_path):
    data = process_image(image_path)
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name[2:], confidence_score

# Streamlit UI
st.title("Authentify AI")

uploaded_file = st.file_uploader("Upload Image...", type=["jpg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Result...")

    class_name, confidence_score = predict(uploaded_file)

    
    st.write(f"Class: {class_name}")
    st.write(f"Confidence: {confidence_score*100:.2f}")
