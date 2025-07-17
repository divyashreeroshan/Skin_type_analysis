import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load the model
model = load_model('model/skin_type_model.h5')
labels = ['dry', 'normal', 'oily']

# Streamlit app
st.set_page_config(page_title="Skin Type Analyzer", layout="centered")
st.title("🌿 Skin Type Analyzer")

st.write("Upload an image of your skin to predict your skin type:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.resize((150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict button
    if st.button("Analyze"):
        prediction = model.predict(img_array)
        predicted_class = labels[np.argmax(prediction)]
        st.success(f"**Prediction:** {predicted_class.capitalize()}")
