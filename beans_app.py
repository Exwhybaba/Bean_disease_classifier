import streamlit as st
import io
import pickle
import numpy as np
import cv2
import requests
import tensorflow as tf
import base64

# GitHub raw file URL for the model
model_url = 'https://github.com/Exwhybaba/Bean_disease_classifier/raw/main/Imagemodel.hdf5'
encoder_url = 'https://raw.githubusercontent.com/Exwhybaba/Beans_disease_classifier/main/encoder.sav'

# Function to load the model from URL and save it locally
def load_model_from_url(url, local_path):
    response = requests.get(url)
    if response.status_code == 200:
        # Save the model file locally
        with open(local_path, 'wb') as f:
            f.write(response.content)

        # Load the model from the local file
        loaded_model = tf.keras.models.load_model(local_path)
        return loaded_model
    else:
        st.error(f"Failed to download the model file. Status code: {response.status_code}")
        return None

# Function to load the encoder from URL
def load_encoder_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        # Load the encoder from the binary stream
        encoder_binary = io.BytesIO(response.content)
        encoder = pickle.load(encoder_binary)
        return encoder
    else:
        st.error(f"Failed to download the encoder file. Status code: {response.status_code}")
        return None

# Function to classify an image
def classifier(image, loaded_model, encoder):
    # Resize and preprocess the image
    resized_image = cv2.resize(image, (150, 150))
    preprocessed_image = resized_image / 255.0
    preprocessed_image = preprocessed_image.reshape((1, 150, 150, 3))

    # Make prediction using the loaded model
    predictions = loaded_model.predict(preprocessed_image, verbose=0)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = encoder.inverse_transform(predicted_class)[0]
    confidence = np.max(predictions) * 100

    return predicted_label, confidence

# Local path to save the model file
local_model_path = 'Imagemodel.hdf5'

# Load the model from the URL and save it locally
loaded_model = load_model_from_url(model_url, local_model_path)
encoder = load_encoder_from_url(encoder_url)

# Check if loading was successful
if loaded_model is not None and encoder is not None:
    # Your Streamlit app code...
    st.title("Bean Disease Detector")
    st.markdown("## Upload an image.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Example usage in main function
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        prediction, confidence = classifier(image, loaded_model, encoder)
        st.success(f"Detection: {prediction} (Accuracy: {confidence:.2f}%)")

        # Example: Display image
        st.image(image, caption="Uploaded Image.", use_column_width=True)
