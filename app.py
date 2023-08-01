from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st

def teachable_machine_classificattion(img):
    np.set_printoptions(suppress=True)
    model = load_model("keras_Model.h5", compile=False)

# Load the labels
    class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
    image = Image.open("C:/Users/Lenovo/OneDrive/Desktop/archive (1)/chest_xray/test/PNEUMONIA/BACTERIA-40699-0001.jpeg").convert("RGB")


# resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
    image_array = np.asarray(image)

# Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
    data[0] = normalized_image_array

# Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name,confidence_score

def main():
    st.title("Image Classification")
    st.header("Pneumonia X-Ray Classification")
    st.text("Upload a Pneumonia X-Ray for classification")
    uploaded_file=st.file_uploader("Choose a X-Ray File....",type="jpeg")
    if uploaded_file is not None:
        image_file=Image.open(uploaded_file)
        st.image(image_file,caption="Uploaded X-Ray Image")
        st.write("")
        label, confidence_score = teachable_machine_classificattion(image_file)

# Assuming the label contains both the class index and the class name separated by a space and ending with a newline character ("\n")

# Split the label into index and name
        label, confidence_score = teachable_machine_classificattion(image_file)

# Split the label into index and name
        label_index, label_name = label.strip().split(" ", 1)

# Check if label_index is "1" (Pneumonia) and display appropriate message
        if label_index == "1" and label_name.lower() == "pneumonia":
            st.success(f"Pneumonia found with confidence: {confidence_score:.2f}")
                          
        else:
           st.error("No Pneumonia Found")

             
    

if __name__ == "__main__":
    main()