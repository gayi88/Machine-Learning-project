import streamlit as st
import numpy as np
import joblib
import matplotlib
import matplotlib.pyplot as plt
import cv2
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import gdown
import os

file_svm = "1skhiOh9N0OW4wNj_KM2jliDClXu6_y6Q"  # svm model
file_rf = "1uh3XLl_bCBm_4Yi6YuCds71NWBjcKVAY"  # rf model

# Local file paths
output_svm = "svm_model.pkl"
output_rf = "rf_model.pkl"

# URLs for downloading
url1 = f"https://drive.google.com/uc?id={file_svm}"
url2 = f"https://drive.google.com/uc?id={file_rf}"

# Download models only if they don't exist
if not os.path.exists(output_svm):
    gdown.download(url1, output_svm, quiet=False)
if not os.path.exists(output_rf):
    gdown.download(url2, output_rf, quiet=False)

# Model file paths for selection
model_files = {
    "svm (98.42%)": output_svm,
    "rf (97.34%)": output_rf,
}

# Streamlit app title and description
st.markdown("<h1 style='text-align: center; color: #ff5733;'>📊 MNIST DATA</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;font-style: italic; color: #1f77b4;'>✍️ Handwritten Digit Prediction</h2>", unsafe_allow_html=True)
st.markdown("""
The MNIST dataset consists of 70,000 images of handwritten digits, ranging from 0 to 9. 
We trained two models on this dataset: 
- A Support Vector Machine (SVM) model with an accuracy of 98.42%.
- A Random Forest model with an accuracy of 97.34%.
""")

# Create a radio box for model selection
selected_model = st.radio("Select a model (svm (98.42%) or rf (97.34%))", list(model_files.keys()))
st.markdown(f"Selected model: {selected_model}")


# Load the selected model
model = None
try:
    model_path = model_files[selected_model]
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None
    
# Create a canvas for drawing digits
canvas_result = st_canvas(
    stroke_width=30,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
)
# Process and predict if there's a drawn image
if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data[:, :, 0:3]

        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Resize to match MNIST input (28x28)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

        # Invert colors (MNIST has black digits on a white background)
        img = cv2.bitwise_not(img)

        # Normalize pixel values (scale to 0-1 like MNIST dataset)
        img = img.astype("float32") / 255.0

        # Flatten to match model input shape (1, 784)
        processed_img = img.flatten().reshape(1, -1)

        # Display the processed image
        st.subheader("Processed Image")
        plt.imshow(processed_img.reshape(28, 28), cmap="gray")
        st.pyplot(plt)

        # Make a prediction
        if model:
            prediction = model.predict(processed_img)[0]
            st.subheader(f"Predicted Digit: {prediction}")
        else:
            st.error("No model selected or loaded.")
