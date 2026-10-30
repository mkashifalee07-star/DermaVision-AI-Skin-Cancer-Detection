
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(
    page_title="Skin Cancer Classification",
    page_icon="🔬",
    layout="centered"
)

st.title("Skin Cancer Classification")
st.write("Educational Deep Learning Project using MobileNetV2")

st.warning(
    "Educational purpose only. This application is not a medical "
    "diagnostic tool. Always consult a qualified dermatologist."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_cancer_model.keras")

model = load_model()

with open("class_names.json", "r") as file:
    class_names = json.load(file)

uploaded_file = st.file_uploader(
    "Upload a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    if st.button("Classify Image"):

        predictions = model.predict(image_array, verbose=0)[0]

        predicted_index = np.argmax(predictions)

        st.subheader("Prediction Result")

        st.success(
            f"Predicted Class: {class_names[predicted_index]}"
        )

        st.write("Class Confidence Scores:")

        for i, class_name in enumerate(class_names):

            confidence = float(predictions[i]) * 100

            st.write(f"{class_name}: {confidence:.2f}%")

            st.progress(float(predictions[i]))
