import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64

st.set_page_config(page_title="Fuzzy Image Processor", layout="centered")
st.title("🧠 Fuzzy Logic + OpenCV Image Processing")

# Sidebar Controls
st.sidebar.header("⚙️ Tools & Parameters")
tool = st.sidebar.selectbox("Select Tool", [
    "Grayscale", "Gradient", "Fuzzy Edge Detection",
    "Thresholding", "Histogram Equalization"
])

threshold = st.sidebar.slider("Threshold", 0, 255, 127)
low_w = st.sidebar.slider("Fuzzy Low Weight", 0.0, 1.0, 0.3)
med_w = st.sidebar.slider("Fuzzy Medium Weight", 0.0, 1.0, 0.7)
high_w = st.sidebar.slider("Fuzzy High Weight", 0.0, 1.0, 0.4)

# File Upload
uploaded_file = st.file_uploader("📁 Upload Image", type=["png", "jpg", "jpeg"])

# Image Processing Functions
def apply_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_gradient(img):
    gray = apply_grayscale(img)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad = np.hypot(sobelx, sobely)
    return np.uint8(grad / grad.max() * 255)

def apply_threshold(img, t):
    gray = apply_grayscale(img)
    _, thresh = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    return thresh

def apply_hist_eq(img):
    gray = apply_grayscale(img)
    return cv2.equalizeHist(gray)

def fuzzy_edge_detection(img, low_w, med_w, high_w):
    gray = apply_grayscale(img)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    norm_img = blurred.astype(np.float32) / 255.0

    low = np.clip((0.5 - norm_img) / 0.5, 0, 1)
    med = 1 - np.abs(norm_img - 0.5) * 2
    high = np.clip((norm_img - 0.5) / 0.5, 0, 1)

    edge_strength = np.maximum(low * low_w, np.maximum(med * med_w, high * high_w))
    return np.uint8(edge_strength * 255)

# Run Processor
if uploaded_file:
    image = Image.open(uploaded_file)
    img_np = np.array(image.convert("RGB"))
    st.image(img_np, caption="Original Image", use_column_width=True)

    if st.button("✨ Apply Tool"):
        if tool == "Grayscale":
            processed = apply_grayscale(img_np)
        elif tool == "Gradient":
            processed = apply_gradient(img_np)
        elif tool == "Thresholding":
            processed = apply_threshold(img_np, threshold)
        elif tool == "Histogram Equalization":
            processed = apply_hist_eq(img_np)
        elif tool == "Fuzzy Edge Detection":
            processed = fuzzy_edge_detection(img_np, low_w, med_w, high_w)
        else:
            processed = img_np

        st.image(processed, caption="Processed Image", use_column_width=True)

        # Download button
        is_gray = len(processed.shape) == 2
        processed_bgr = processed if is_gray else cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".png", processed_bgr)
        b64 = base64.b64encode(buffer).decode()
        href = f'<a href="data:file/png;base64,{b64}" download="processed.png">📥 Download Result</a>'
        st.markdown(href, unsafe_allow_html=True)
