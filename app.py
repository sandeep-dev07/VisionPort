import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch

# ----------------------------------------
# 1. Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="VisionPort",
    page_icon="🚦",
    layout="wide"
)

# ----------------------------------------
# 2. Title
# ----------------------------------------
st.title("VisionPort v0.1")

st.write("GPU-Accelerated Traffic Object Detection")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Running on **{device.upper()}**")

# ----------------------------------------
# 3. Load YOLO Model
# ----------------------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    return model

model = load_model()

# ----------------------------------------
# 4. Upload Image
# ----------------------------------------
uploaded_file = st.file_uploader(
    "Upload a traffic image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(
        image,
        caption="Uploaded Traffic Image",
        use_container_width=True
    )

    # Run YOLO detection
    results = model.predict(image, device=device)

    # Get detected image
    detected_image = results[0].plot()

    st.subheader("Detection Results")
    st.image(
        detected_image,
        caption="Detection Results",
        use_container_width=True
    )