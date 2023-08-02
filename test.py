from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

# Load the Keras model
model = load_model("keras_model.h5", compile=False)

# Streamlit UI code
def main():
    st.title("Image Classification")
    st.header("Pneumonia X-Ray Classification")
    st.text("Upload a Pneumonia X-Ray for classification")

    # File upload widget
    file = st.file_uploader('Upload an image file')

    # Main prediction logic
    if file is not None:
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        img = Image.open(file)
        # img = img.convert('RGB')
        size = (224, 224)
        image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

        # Convert image to numpy array and normalize
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        # Predict using the model
        prediction = model.predict(data)
        probability = model.predict_proba(data)
        probabilities = prediction[0]  # Probabilities for each class

        # Display prediction result
        st.subheader("Prediction Result:")
        st.image(image, use_column_width=True, caption="Uploaded Image")
        if (probabilities==1):
            st.write("Pneumonia Found!")
            st.write("The probability is:" , probability)
        else :
            st.write("Pneumonia Not Found.")
            st.write("The probability is:" , probability)
        # for class_index, class_probability in enumerate(probabilities):
        #     # Normalize the probability between 0 and 1
        #     class_probability = class_probability / np.sum(probabilities)
        #     st.write(f"Class {class_index}: Probability: {class_probability:.2f}")
    # else:
    #     st.write("Please upload an image.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
