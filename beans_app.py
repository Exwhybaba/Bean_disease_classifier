import streamlit as st
import io
import pickle
import numpy as np
import cv2
import base64
from tensorflow.keras.models import load_model

# Load the trained model
loaded_model = load_model('Imagemodel.h5', compile=False)

# Load the encoder
with open('encoder.sav', 'rb') as f:
    encoder = pickle.load(f)

# Descriptions for different predictions
descriptions = {
    'angular_leaf_spot': "Angular lesions with water-soaked margins, often with a yellow halo.",
    'bean_rust': "Orange to rusty-brown powdery pustules on the undersides of leaves.",
    'healthy': "Vibrant green leaves without lesions or rust-like growth."
}

def classifier(image):
    # Preprocess the image for prediction
    resize = cv2.resize(image, (150, 150))
    rescaling = resize / 255.0
    rescaling = rescaling.reshape((1, 150, 150, 3))
    
    # Make predictions using the loaded model
    predictor = loaded_model.predict(rescaling, verbose=0)
    predicted_class = np.argmax(predictor, axis=1)
    predict = encoder.inverse_transform(predicted_class)
    confidence = np.max(predictor) * 100  # Confidence as a percentage
    return predict[0], confidence

def encode_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    return encoded_image

def main():
    # Set page configuration
    st.set_page_config(page_title="Bean Disease Detector", page_icon=":seedling:", layout="wide")

    # Background image using custom CSS
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encode_image_as_base64('./images/crop_health.jpg')}");  
                background-size: cover;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # App title and description with blue color and adjusted margin for title
    st.markdown("<h1 class='custom-title'>Bean Disease Detector</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: white; margin-top: -40px;'>Upload an image.</h2>", unsafe_allow_html=True)

    # Apply custom styles using CSS
    st.markdown(
        """
        <style>
            .custom-title {
                color: white;
                margin-top: -45px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Apply white color to the file uploader label text using CSS
    st.markdown(
        """
        <style>
            .css-qrbaxs {
                color: white !important;
                display: block;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # File uploader label text with white color
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")

    # Display image and classification
    if uploaded_file is not None:
        # Decode the uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display image and classification
        col1, col2 = st.columns([3, 1]) 
        with col1:
            st.image(image_rgb, caption="Uploaded Image.", use_column_width=True, width = 100)
           
        with col2:
            st.markdown(
                f"""
                <style>
                    .stApp {{
                        background-image: url("data:image/jpeg;base64,{encode_image_as_base64('./images/mossgreen.jpg')}");  
                        background-size: cover;
                    }}
                </style>
                """,
                unsafe_allow_html=True,
            )
            
            with st.spinner("Detecting..."):
                prediction, confidence = classifier(image)
            st.success(f"Detection: {prediction} (Accuracy: {confidence:.2f}%)")
            
            # Display dynamic description based on prediction
            if prediction in descriptions:
                st.markdown(f"### Description:\n{descriptions[prediction]}")

    # Copyright notice
    st.markdown(
        """
        ---
        © 2023 Oyelayo Seye. All rights reserved.
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
