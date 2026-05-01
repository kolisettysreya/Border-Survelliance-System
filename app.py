import streamlit as st
import cv2
import tempfile
import os
from pathlib import Path

st.set_page_config(page_title="Border Surveillance System", layout="wide")

st.title("🛡️ Border Surveillance System")
st.markdown("AI-powered border monitoring using YOLOv8 trained on VisDrone dataset")

st.sidebar.header("About")
st.sidebar.markdown("""
**Model:** YOLOv8m  
**Dataset:** VisDrone  
**Classes:** Person, Vehicle, Motorcycle, Drone, Animal  
**Task:** Perimeter crossing detection
""")

st.markdown("### How it works")
col1, col2, col3 = st.columns(3)
with col1:
    st.info("📹 Upload a video or image")
with col2:
    st.info("🤖 YOLOv8 detects objects")
with col3:
    st.info("🚨 Alerts on perimeter crossing")

uploaded = st.file_uploader("Upload an image for detection",
                            type=["jpg", "jpeg", "png"])

if uploaded:
    from ultralytics import YOLO
    import numpy as np

    model_path = "models/best.pt"
    if not os.path.exists(model_path):
        st.error("Model file not found. Please ensure best.pt is in the models/ folder.")
    else:
        model = YOLO(model_path)

        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Running detection..."):
            results = model(img, conf=0.3)
            annotated = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        st.image(annotated_rgb, caption="Detection Result", use_column_width=True)

        boxes = results[0].boxes
        if len(boxes) > 0:
            st.success(f"✅ Detected {len(boxes)} object(s)")
            names = model.names
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                st.write(f"- **{names[cls]}** — confidence: {conf:.2%}")
        else:
            st.warning("No objects detected")