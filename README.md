# Skin Cancer Classification Using Deep Learning

## Project Overview
This project uses Transfer Learning with MobileNetV2
to classify skin lesion images into 9 different classes.

## Technologies Used
- Python
- TensorFlow / Keras
- MobileNetV2
- Streamlit
- NumPy
- Pillow
- Matplotlib
- Scikit-learn

## Dataset Classes
1. Actinic keratosis
2. Basal cell carcinoma
3. Dermatofibroma
4. Melanoma
5. Nevus
6. Pigmented benign keratosis
7. Seborrheic keratosis
8. Squamous cell carcinoma
9. Vascular lesion

## Model
- Architecture: MobileNetV2
- Input image size: 224 x 224
- Output classes: 9
- Transfer learning with frozen ImageNet base model

## Test Results
- Test Accuracy: 43.22%
- Weighted Precision: 0.4678
- Weighted Recall: 0.4322
- Weighted F1-score: 0.3878

These are results from a small test dataset and should
not be interpreted as clinical performance.

## Run Locally

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## Medical Disclaimer
This project is developed for educational and research
demonstration purposes only.

It is NOT a medical diagnostic tool.
It must NOT be used to diagnose skin cancer or make
medical decisions. Always consult a qualified dermatologist.