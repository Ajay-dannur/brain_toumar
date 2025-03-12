import streamlit as st
import torch
import numpy as np
import cv2
from yolov10.models import YOLO  # Ensure import works after installation

# Load YOLOv10 model
model = YOLO("best.onnx")  # Ensure the model is in the same directory

def detect_objects(image):
    results = model(image)  # Run inference
    result_img = results[0].plot()  # Draw bounding boxes
    return result_img

# Streamlit UI
st.title("Brain Tumor Detection with YOLOv10")
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1))
    result_img = detect_objects(image)
    st.image(result_img, caption="Detection Result", use_column_width=True)
