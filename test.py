from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

model = load_model("keras_model.h5", compile=False)

file = st.file_uploader('Upload an image file')

class_names = ["1", "0"]

confidence_scores = None

if file is not None:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(file)
    img = img.convert('RGB')  # Convert to RGB mode to ensure 3 channels
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    confidence_scores = prediction[0]


def main():
    st.title("Image Classification")
    st.header("Pneumonia X-Ray Classification")
    st.text("Upload a Pneumonia X-Ray for classification")
    
    if confidence_scores is not None:
        st.subheader("Prediction Result:")
        st.image(image, use_column_width=True, caption="Uploaded Image")
        
        if confidence_score==1:
            st.write("Pneumonia Found")
        else :
            st.write("No Pneumonia,take a chill pill")



if __name__ == "__main__":
    main()
