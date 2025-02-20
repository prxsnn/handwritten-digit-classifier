import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load the trained model
model = tf.keras.models.load_model("digit_recognition_model.keras")

def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert colors (black background, white digit)
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image = np.array(image) / 255.0  # Normalize to [0,1]
    image = image.reshape(1, 28, 28, 1)  # Reshape for model
    return image

st.title("Handwritten Digit Recognition")
st.write("Draw a digit (0-9) below and get the model's prediction!")

# Upload or draw digit
canvas_result = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if canvas_result:
    image = Image.open(canvas_result)
    st.image(image, caption='Uploaded Image',  use_container_width=True)
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    
    st.write(f"### Predicted Digit: {predicted_digit}")
