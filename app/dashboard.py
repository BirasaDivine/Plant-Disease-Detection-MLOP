import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

API_URL = "http://localhost:8000"

def check_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code == 200:
            return True
    except requests.exceptions.ConnectionError:
        return False
st.title("Plant Disease Detection Dashboard")
st.header("Model Status")
if check_health():
    st.success(" Model is Online and Running")
else:
    st.error(" Model is Offline or Not Reachable")

st.header("Data Visualisations")

# Chart 1 - Class Distribution
st.subheader("Class Distribution")
fig1, ax1 = plt.subplots()
classes = ["Early Blight", "Late Blight", "Healthy"]
counts = [1000, 1000, 152]
colours = ["#E74C3C", "#3498DB", "#2ECC71"]
ax1.bar(classes, counts, color=colours, edgecolor="black")
ax1.set_ylabel("Number of Images")
ax1.set_title("Dataset Class Distribution")
st.pyplot(fig1)
st.write("The dataset is imbalanced , Healthy has only 152 images compared to 1000 each for disease classes.")

# Chart 2 - Sample Images
st.subheader("Sample Images per Class")
st.info("Dataset contains potato leaf images across 3 classes from the PlantVillage dataset.")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Early Blight", "1,000 images", "46.5%")
with col2:
    st.metric("Late Blight", "1,000 images", "46.5%")
with col3:
    st.metric("Healthy", "152 images", "7.1%")

# Chart 3 - Model Accuracy 
st.subheader("Model Accuracy Comparison")
fig3, ax3 = plt.subplots()
experiments = ["RF", "SVM", "Ensemble", "BaseCNN", "RegCNN", "DeepCNN", "MobileNet"]
accuracies = [91.3, 94.1, 95.1, 97.8, 96.7, 95.0, 98.8]
bar_colours = ["#3498DB", "#3498DB", "#3498DB", "#3498DB", "#3498DB", "#3498DB", "#F1C40F"]
ax3.bar(experiments, accuracies, color=bar_colours, edgecolor="black")
ax3.set_ylabel("Accuracy (%)")
ax3.set_ylim((85, 100))
ax3.set_title("Accuracy Across All 7 Experiments")
plt.xticks(rotation=45)
st.pyplot(fig3)
st.write("MobileNetV2 achieves the highest accuracy at 98.8%, shown in gold.")

# Section 3
st.header("Predict Potato Disease")

uploaded_file = st.file_uploader(
    "Upload a potato leaf image",
    type=["jpg", "jpeg", "png"],
    key="predict_uploader"
)

if uploaded_file is not None:
    st.image(uploaded_file, width=300)
    
    if st.button("Predict Disease", key="predict_btn"):
        with st.spinner("Analyzing..."):
            try:
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(f"{API_URL}/predict", files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Prediction: {result.get('class')}")
                    st.write(f"Confidence: {result.get('confidence')}%")
                else:
                    st.error("Prediction failed")
            except Exception as e:
                st.error(f"Error: {e}")

st.header("Upload Data and Retrain Model")

class_name = st.selectbox(
    "Select Class for Uploaded Images",
    ["Early Blight", "Late Blight", "Healthy"]
)

uploaded_files = st.file_uploader(
    "Upload new training images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="retrain_uploader"
)

if st.button("Upload Images"):
    if len(uploaded_files) == 0:
        st.warning("Please upload at least one image first")
    else:
        with st.spinner("Uploading..."):
            try:
                files = [("files", f.getvalue()) for f in uploaded_files]
                response = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                    data={"class_name": class_name}
                )
                if response.status_code == 200:
                    st.success(response.json()["message"])
                else:
                    st.error("Upload failed")
            except Exception as e:
                st.error("Error connecting to API")

if st.button("Retrain Model"):
    with st.spinner("Retraining... this may take a few minutes"):
        try:
            response = requests.post(f"{API_URL}/retrain")
            if response.status_code == 200:
                st.success("Model retrained successfully")
            else:
                st.error("Retraining failed")
        except Exception as e:
            st.error("Error connecting to API")