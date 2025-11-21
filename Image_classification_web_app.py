import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st

# App Header
st.markdown("""
<h1 style='text-align: center;'>🍑 Fruits Classification Model 🍎</h1>
""", unsafe_allow_html=True)

# Load Model
model = load_model("Image_classification_model.keras")

# Page Background
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #0866a3, #d9fdd3);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Fruit Categories
data_category = [
    'Apple', 'Avocado', 'Banana', 'Grapes', 'Mango',
    'Orange', 'Pineapple', 'Pomegranate', 'Strawberry', 'Watermelon'
]
    
st.markdown("""
<div style="text-align:center;">
    <h3><b>Upload image of any following fruits:</b></h3>
    🍎 Apple &nbsp;&nbsp; 🥑 Avocado &nbsp;&nbsp; 🍎 Pomegranate &nbsp;&nbsp; 🍌 Banana &nbsp;&nbsp; 🍇 Grapes <br>
    🍊 Orange &nbsp;&nbsp; 🍍 Pineapple &nbsp;&nbsp; 🍓 Strawberry &nbsp;&nbsp; 🥭 Mango &nbsp;&nbsp; 🍉 Watermelon
</div>
""", unsafe_allow_html=True)


# File Upload

img_height = 180
img_width = 180

img = st.file_uploader("", type=["jpg", "jpeg", "png"])

# Prediction Logic
if img is not None:

    # Load & preprocess
    loaded_img = tf.keras.utils.load_img(img, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(loaded_img)
    img_batch = tf.expand_dims(img_array, 0)

    # Prediction
    prediction = model.predict(img_batch)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    # Output
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(img, width=180)

    with col2:
        st.success(f'***Fruit detected by the Model :*** **{data_category[predicted_class]}**')
        st.info(f'***With confidence score of :*** {confidence:.2f}%')

else:
    st.warning("📌 **Please upload an image file.**")
