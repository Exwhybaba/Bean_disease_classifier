import streamlit as st
import io
import pickle
import numpy as np
import cv2
import requests
import tensorflow as tf
import base64

# Load the model and encoder URLs
model_url = 'https://raw.githubusercontent.com/Exwhybaba/Beans_disease_classifier/main/Imagemodel.h5'
encoder_url = 'https://raw.githubusercontent.com/Exwhybaba/Beans_disease_classifier/main/encoder.sav'

# Download the model file
response = requests.get(model_url)
model_content = response.content

#loaded model
loaded_model = tf.keras.models.load_model(model_content )




# Download the encoder file
response = requests.get(encoder_url)
encoder_content = response.content

# Load the model and transformers
encoder = pickle.loads(encoder_content)

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

# Function to encode image as base64
def encode_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    return encoded_image

# Main function
def main():
    # Set page configuration
    st.set_page_config(page_title="Bean Disease Detector", page_icon=":seedling:", layout="wide")

    # Load the model and encoder
    loaded_model = load_model_from_url(model_url)
    encoder = load_encoder_from_url(encoder_url)

    # Check if loading was successful
    if loaded_model is None or encoder is None:
        return

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
        

if __name__ == "__main__":
    main()
