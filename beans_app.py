import streamlit as st
import io
import pickle
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import base64

# Load the model
model_path = r"C:\Users\Administrator\Documents\Sandbox\pytouch\Bean\Imagemodel.h5"
encoder_path = r"C:\Users\Administrator\Documents\Sandbox\pytouch\Bean\encoder.sav"
loaded_model = load_model(model_path)

# Load the encoder
with open(encoder_path, 'rb') as file:
    encoder = pickle.load(file)
    
    
# Descriptions for different predictions
descriptions = {
    'angular_leaf_spot': "Angular lesions with water-soaked margins, often with a yellow halo.",
    'bean_rust': "Orange to rusty-brown powdery pustules on the undersides of leaves.",
    'healthy': "Vibrant green leaves without lesions or rust-like growth."
}

def classifier(image):
    resize = cv2.resize(image, (150, 150))
    rescaling = resize / 255.0
    rescaling = rescaling.reshape((1, 150, 150, 3))
    predictor = loaded_model.predict(rescaling, verbose=0)
    predicted_class = np.argmax(predictor, axis=1)
    predict = encoder.inverse_transform(predicted_class)
    return predict[0]

def encode_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    return encoded_image

def main():
    # Set page configuration
    st.set_page_config(page_title="Bean Disease Detector", page_icon=":seedling:", layout="wide")

    # Custom CSS styles
    st.markdown("""
        <style>
            body {
                background-color: #f4f4f4;
                margin-top: -60px; 
                margin-bottom: 100px; /* Adjust the margin-bottom value as needed */
            }
            .st-bc {
                color: #333333;
            }
        </style>
    """, unsafe_allow_html=True)

    # Background image using custom CSS
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encode_image_as_base64('C:/Users/Administrator/Documents/Sandbox/pytouch/Bean/cowpea2.jpg')}");  
                background-size: cover;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # App title and description
    st.title("Bean Disease Detector")
    st.markdown("## Upload an image.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Display image and classification
    if uploaded_file is not None:
        # Decode the uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display image and classification
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_rgb, caption="Uploaded Image.", use_column_width=True)

        with col2:
            with st.spinner("Detecting..."):
                prediction = classifier(image)
            st.success(f"Detection: {prediction}")
            
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
