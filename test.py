from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

# Load the Keras model
model = load_model("keras_model.h5", compile=False)

# Class names for classification
class_names = ["1", "0"]

# Streamlit UI code
def main():
    st.title("Image Classification")
    st.header("Pneumonia X-Ray Classification")
    st.text("Upload a Pneumonia X-Ray for classification")

    # File upload widget
    file = st.file_uploader('Upload an image file')

    # Placeholder for confidence scores
    confidence_scores = None

    # Main prediction logic
    if file is not None:
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        img = Image.open(file)
        img = img.convert('RGB')
        size = (224, 224)
        image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

        # Convert image to numpy array and normalize
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        # Predict using the model
        prediction = model.predict(data)
        confidence_scores = prediction[0]

        # Display prediction result
        st.subheader("Prediction Result:")
        st.image(image, use_column_width=True, caption="Uploaded Image")

        for i, class_name in enumerate(class_names):
            st.write(f"Class: {class_name}")
            st.write(f"Probability: {confidence_scores[i]:.2f}")
    else:
        st.write("Please upload an image.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
