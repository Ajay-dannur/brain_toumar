import os
import streamlit as st
import torch
import numpy as np
import cv2

# ✅ Install YOLOv10 manually before importing
os.system("pip install -q git+https://github.com/THU-MIG/yolov10.git")

# ✅ Import YOLOv10 after installation
try:
    from yolov10.models import YOLO
    print("✅ YOLOv10 imported successfully!")
except ModuleNotFoundError:
    st.error("❌ YOLOv10 not found. Please restart the app.")
    exit()

# ✅ Load the trained YOLOv10 model
model = YOLO("best.onnx")  # Ensure best.onnx is in the same directory

def detect_objects(image):
    results = model(image)  # Run inference
    result_img = results[0].plot()  # Draw bounding boxes
    return result_img

# ✅ Streamlit UI
st.title("🧠 Brain Tumor Detection with YOLOv10")
st.write("Upload an MRI scan to detect brain tumors.")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1))
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    # Process the image
    result_img = detect_objects(image)
    st.image(result_img, caption="Detection Result", use_column_width=True)
