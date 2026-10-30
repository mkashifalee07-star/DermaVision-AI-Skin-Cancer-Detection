import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
import pandas as pd
from PIL import Image
from datetime import datetime

st.set_page_config(
    page_title="DermaVision AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_path, "r") as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)

load_css()

# App branding
st.sidebar.markdown(
    """
    <div style="background:#102A43;padding:20px;border-radius:15px;text-align:center;">
        <h1 style="color:#5EEAD4;margin:0;">🔬 DermaVision AI</h1>
        <p style="color:#D5E7F2;font-size:12px;margin-bottom:0;">
            Skin Image Classification
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.write("")

# Navigation
page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Home",
        "Image Analysis",
        "Analytics",
        "About Model",
        "Medical Disclaimer"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption("Powered by TensorFlow & MobileNetV2")
st.sidebar.caption("Educational Project | 2026")

# Main header
st.markdown(
    """
    <div style="background:#102A43;padding:25px;border-radius:16px;margin-bottom:25px;">
        <h1 style="color:#5EEAD4;margin:0;">DermaVision AI</h1>
        <p style="color:#FFFFFF;margin-top:8px;">
            AI-Powered Experimental Skin Lesion Classification
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("Selected Page:", page)

# ==========================================
# AI MODEL LOADING
# ==========================================

@st.cache_resource
def load_skin_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "skin_cancer_model.keras"
    )
    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_class_labels():
    labels_path = os.path.join(
        os.path.dirname(__file__),
        "class_names.json"
    )

    with open(labels_path, "r") as file:
        return json.load(file)


try:
    model = load_skin_model()
    class_names = load_class_labels()

    st.sidebar.success("AI Model Loaded Successfully")

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()



# ==========================================
# HOME PAGE
# ==========================================

if page == "Home":

    st.markdown(
        """
        <div style="
            background:linear-gradient(120deg,#102A43,#167C78);
            padding:40px;
            border-radius:20px;
            color:white;
            margin-bottom:25px;
        ">
            <h1 style="color:#5EEAD4;font-size:38px;">
                Welcome to DermaVision AI
            </h1>
            <p style="color:white;font-size:17px;">
                Explore experimental AI-powered skin lesion image
                classification using deep learning.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Explore Your AI Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Supported Classes", "9")

    with col2:
        st.metric("AI Architecture", "MobileNetV2")

    with col3:
        st.metric("Image Input", "224 × 224")

    st.markdown("---")

    st.subheader("About DermaVision AI")

    st.write(
        """
        DermaVision AI is an educational deep learning project
        designed to explore image classification of skin lesions.

        It uses transfer learning with the MobileNetV2 architecture
        and TensorFlow to generate experimental classification outputs.
        """
    )

    st.subheader("Supported Image Categories")

    for start in range(0, len(class_names), 3):

        cols = st.columns(3)

        for col, class_name in zip(
            cols,
            class_names[start:start + 3]
        ):
            with col:
                st.markdown(
                    f"""
                    <div style="
                        background:white;
                        border:1px solid #D7E8E8;
                        border-radius:12px;
                        padding:16px;
                        margin-bottom:12px;
                        min-height:65px;
                    ">
                        <span style="color:#167C78;font-weight:600;">
                            {class_name.title()}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("---")

    st.info(
        "This project is strictly for educational purposes. "
        "Its predictions are not medical diagnoses. "
        "Consult a qualified dermatologist for medical concerns."
    )



# ==========================================
# IMAGE ANALYSIS - IMAGE UPLOAD
# ==========================================

if page == "Image Analysis":

    st.title("🔬 Skin Image Analysis")

    st.write(
        "Upload a skin lesion image to explore "
        "the experimental AI classification output."
    )

    st.markdown("---")

    st.subheader("Upload Your Image")

    uploaded_file = st.file_uploader(
        "Choose a skin image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG"
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.success("Image uploaded successfully!")

            col1, col2 = st.columns([1.5, 1])

            with col1:
                st.image(
                    image,
                    caption="Uploaded Skin Image",
                    use_container_width=True
                )

            with col2:
                st.markdown("### Image Information")

                st.write("**Format:**", uploaded_file.type)

                st.write(
                    "**Original Dimensions:**",
                    f"{image.width} × {image.height}"
                )

                st.write(
                    "**Colour Mode:**",
                    image.mode
                )

                st.write(
                    "**File Size:**",
                    f"{uploaded_file.size / 1024:.2f} KB"
                )

            st.markdown("---")

            st.info(
                "Your image is ready for experimental "
                "classification. The next step will add "
                "the AI prediction functionality."
            )

        except Exception:
            st.error(
                "Unable to read this image. "
                "Please upload a valid JPG or PNG file."
            )

    else:

        st.markdown(
            """
            <div style="
                background:#FFFFFF;
                border:2px dashed #16A6A1;
                border-radius:16px;
                padding:35px;
                text-align:center;
                margin-top:15px;
            ">
                <h2 style="color:#167C78;">📷 Upload Your Image</h2>
                <p style="color:#64748B;">
                    Drag and drop your image or browse files
                </p>
                <p style="color:#94A3B8;font-size:13px;">
                    JPG • JPEG • PNG
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )



# ==========================================
# AI PREDICTION AND CONFIDENCE SCORES
# ==========================================

if page == "Image Analysis" and uploaded_file is not None:

    st.markdown("---")
    st.subheader("AI Classification")

    if st.button("Analyze Image", type="primary", use_container_width=True):

        with st.spinner("AI model is analyzing the image..."):

            # Prepare image for MobileNetV2
            processed_image = image.resize((224, 224))

            image_array = np.array(
                processed_image,
                dtype=np.float32
            )

            image_array = np.expand_dims(image_array, axis=0)

            # Generate predictions
            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            predicted_index = int(np.argmax(predictions))

            predicted_class = class_names[predicted_index]

            confidence = float(
                predictions[predicted_index] * 100
            )

            # Save prediction history
            if "prediction_history" not in st.session_state:
                st.session_state.prediction_history = []

            prediction_record = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "predicted_class": predicted_class,
                "confidence": confidence
            }

            st.session_state.prediction_history.append(prediction_record)

            st.session_state.last_prediction = {
                "time": prediction_record["time"],
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_scores": {
                    class_names[i]: float(predictions[i] * 100)
                    for i in range(len(class_names))
                }
            }

        st.success("Experimental classification completed!")

        st.markdown("### Predicted Class")

        st.markdown(
            f"""
            <div style="
                background:#102A43;
                padding:25px;
                border-radius:15px;
                text-align:center;
            ">
                <h2 style="color:#5EEAD4;margin:0;">
                    {predicted_class.title()}
                </h2>
                <h3 style="color:white;">
                    Model confidence score: {confidence:.2f}%
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            "Confidence is the model's softmax score, "
            "not the probability of having a disease."
        )

        st.markdown("---")
        st.subheader("All 9 Class Confidence Scores")

        sorted_indices = np.argsort(predictions)[::-1]

        for index in sorted_indices:

            class_name = class_names[index]
            score = float(predictions[index])

            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(class_name.title())

            with col2:
                st.write(f"{score * 100:.2f}%")

            st.progress(score)

        st.markdown("---")
        st.subheader("Top 3 Model Predictions")

        top_three = sorted_indices[:3]

        for rank, index in enumerate(top_three, start=1):

            st.write(
                f"{rank}. **{class_names[index].title()}** "
                f"— {predictions[index] * 100:.2f}%"
            )

        st.warning(
            "Educational demonstration only. This model is not "
            "clinically validated and must not be used for diagnosis "
            "or treatment decisions. Consult a qualified dermatologist."
        )



# ==========================================
# DARK MODE / LIGHT MODE
# ==========================================

st.sidebar.markdown("---")
st.sidebar.subheader("Appearance")

dark_mode = st.sidebar.toggle(
    "Enable Dark Mode",
    value=False,
    key="dark_mode"
)

if dark_mode:

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0B1724 !important;
            color: #E2E8F0 !important;
        }

        h1, h2, h3, h4, p, label,
        [data-testid="stMarkdownContainer"] {
            color: #E2E8F0 !important;
        }

        section[data-testid="stSidebar"] {
            background-color: #102A43 !important;
        }

        div[data-testid="stMetric"] {
            background-color: #172B3D !important;
            border: 1px solid #28506B !important;
            border-radius: 12px;
        }

        [data-testid="stFileUploader"] {
            background-color: #172B3D !important;
            border-color: #16A6A1 !important;
        }

        .stTextInput input {
            background-color: #172B3D !important;
            color: white !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #F0F7F7 !important;
            color: #102A43 !important;
        }

        section[data-testid="stSidebar"] {
            background-color: #E1F0EF !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )



# ==========================================
# ANALYTICS PAGE
# ==========================================

if page == "Analytics":

    st.title("📊 Prediction History")
    st.write("View and track your image analysis results.")

    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    history = st.session_state.prediction_history

    if len(history) > 0:

        st.subheader("Session Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Images Analyzed", len(history))

        with col2:
            average_confidence = sum(
                item["confidence"] for item in history
            ) / len(history)

            st.metric(
                "Average Confidence",
                f"{average_confidence:.2f}%"
            )

        st.markdown("---")
        st.subheader("Previous Predictions")

        history_df = pd.DataFrame(history)

        history_df = history_df.rename(columns={
            "time": "Date & Time",
            "predicted_class": "Predicted Class",
            "confidence": "Confidence (%)"
        })

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download History as CSV",
            data=csv_data,
            file_name="dermavision_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("🗑️ Clear Prediction History"):
            st.session_state.prediction_history = []
            st.rerun()

    else:

        st.info(
            "No prediction history available yet. "
            "Go to Image Analysis and analyze an image first."
        )



# ==========================================
# ABOUT MODEL PAGE
# ==========================================

if page == "About Model":

    st.title("🧠 About DermaVision AI")
    st.write(
        "Learn about the deep learning model behind this application."
    )

    st.markdown("---")

    st.subheader("Model Architecture")

    st.write(
        "DermaVision AI uses MobileNetV2, a pre-trained deep learning "
        "architecture originally trained on ImageNet. Transfer learning "
        "was used to adapt it for skin lesion image classification."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Architecture", "MobileNetV2")

    with col2:
        st.metric("Input Size", "224 × 224")

    with col3:
        st.metric("Output Classes", "9")

    st.markdown("---")

    st.subheader("Training Information")

    training_details = {
        "Training images": "1,791",
        "Validation images": "448",
        "Original test images": "118",
        "Training epochs": "15",
        "Best validation accuracy": "56.92%",
        "Test accuracy": "43.22%"
    }

    for detail, value in training_details.items():
        st.write(f"**{detail}:** {value}")

    st.info(
        "The test accuracy of 43.22% indicates limited performance "
        "on the held-out test set. Results should not be interpreted "
        "as reliable medical predictions."
    )

    st.markdown("---")

    st.subheader("Supported Skin Lesion Categories")

    for index, name in enumerate(class_names, start=1):
        st.write(f"{index}. {name.replace('_', ' ').title()}")

    st.markdown("---")

    st.subheader("Technology Stack")

    st.write(
        """
        - Python
        - TensorFlow and Keras
        - MobileNetV2
        - Streamlit
        - NumPy and Pandas
        - Pillow
        """
    )


# ==========================================
# MEDICAL DISCLAIMER PAGE
# ==========================================

if page == "Medical Disclaimer":

    st.title("⚠️ Medical Disclaimer")

    st.warning(
        "DermaVision AI is an educational and research project. "
        "It is NOT a medical device and must NOT be used to diagnose "
        "skin cancer or any other medical condition."
    )

    st.subheader("Important Information")

    st.write(
        """
        1. The application provides experimental predictions based on a deep learning model.

        2. Model predictions may be incorrect or misleading.

        3. A confidence score is not the probability that a person has a particular disease.

        4. The model has limited test performance and has not been clinically validated.

        5. The application cannot replace a qualified dermatologist or medical professional.
        """
    )

    st.markdown("---")

    st.subheader("When to Seek Medical Advice")

    st.write(
        "If you notice a new, changing, bleeding, painful, or unusual "
        "skin spot, consult a qualified healthcare professional. "
        "Do not delay medical care because of this application's result."
    )

    st.markdown("---")

    st.subheader("Privacy Notice")

    st.write(
        "This project does not intentionally save uploaded images "
        "to a permanent database. Prediction history is stored in "
        "the current Streamlit session and may disappear when the "
        "session ends."
    )

    st.error(
        "Never use DermaVision AI results to make treatment decisions."
    )



# ==========================================
# DOWNLOADABLE PREDICTION REPORT
# ==========================================

if page == "Image Analysis" and "last_prediction" in st.session_state:

    last_result = st.session_state.last_prediction

    st.markdown("---")
    st.subheader("📄 Download Prediction Report")

    report_text = (
        "DERMAVISION AI - IMAGE ANALYSIS REPORT\n"
        "========================================\n\n"
        f"Analysis Date & Time: {last_result['time']}\n"
        f"Predicted Class: {last_result['predicted_class']}\n"
        f"Confidence Score: {last_result['confidence']:.2f}%\n\n"
        "CLASS-WISE SCORES\n"
        "-----------------\n"
    )

    for class_name, score in last_result["class_scores"].items():
        report_text += f"{class_name}: {score:.2f}%\n"

    report_text += (
        "\nIMPORTANT MEDICAL DISCLAIMER\n"
        "----------------------------\n"
        "This report is educational only and is not a medical diagnosis. "
        "The confidence score is not the probability of having a disease. "
        "Consult a qualified healthcare professional for medical advice.\n"
    )

    st.download_button(
        label="📥 Download Prediction Report",
        data=report_text,
        file_name="dermavision_prediction_report.txt",
        mime="text/plain",
        use_container_width=True
    )

