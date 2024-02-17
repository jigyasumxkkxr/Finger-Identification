import streamlit as st
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


def main():
    st.title("Hand Orientation Prediction")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_path = save_uploaded_image(uploaded_file)

        predict_hand_orientation(image_path, image)


def save_uploaded_image(uploaded_file):
    # Save the uploaded image to a temporary location
    temp_file = "temp_image.png"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getvalue())
    return temp_file


def predict_hand_orientation(image_path, image):
    def load_model(model_path):
        """
        Loads a saved model from a specified path.
        """
        print(f"Loading saved model from: {model_path}")
        custom_objects = {'KerasLayer': hub.KerasLayer}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        return model

    Img_size = 224

    def process_image(image_path):
        """
        Turns Image into Tensors
        """
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, size=[Img_size, Img_size])
        return image

    model = load_model(
        r"C:\Users\jigya\Desktop\Finger-Orientation-Prediction\20240217-12141708172075-all-images-Fingers.h5")
    X = [image_path]
    batched_image = process_image(image_path)
    batched_image = tf.expand_dims(batched_image, 0)
    predict = model.predict(batched_image)
    unique_finger = ['0L', '0R', '1L', '1R', '2L', '2R', '3L', '3R', '4L', '4R', '5L',
                     '5R']
    var = unique_finger[np.argmax(predict)]

    st.image(image, caption='Uploaded Image', width=200)
    st.text(var)


if __name__ == "__main__":
    main()
