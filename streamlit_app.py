import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64
import skfuzzy as fuzz

st.set_page_config(page_title="Fuzzy Image Processor", layout="centered")
st.title("🧠 Fuzzy Logic-Based Image Processing using OpenCV + scikit-fuzzy")
st.markdown("""
This mini-project demonstrates soft computing principles using fuzzy logic and OpenCV. 
It includes fuzzy edge detection, fuzzy thresholding, fuzzy contrast enhancement, fuzzy brightness boost, and more.
""")

# Sidebar Controls
st.sidebar.header("⚙️ Tools & Parameters")
tool = st.sidebar.selectbox("Select Tool", [
    "Grayscale", "Gradient", "Fuzzy Edge Detection",
    "Fuzzy Thresholding", "Fuzzy Contrast Enhancement", "Fuzzy Brightness Enhancement",
    "Thresholding", "Histogram Equalization",
    "Canny Edge Detection", "Gaussian Blur",
    "Sharpening", "Invert Colors", "Adaptive Thresholding"
])

threshold = st.sidebar.slider("Threshold", 0, 255, 127)
low_w = st.sidebar.slider("Fuzzy Low Weight", 0.0, 1.0, 0.3)
med_w = st.sidebar.slider("Fuzzy Medium Weight", 0.0, 1.0, 0.7)
high_w = st.sidebar.slider("Fuzzy High Weight", 0.0, 1.0, 0.4)
canny_t1 = st.sidebar.slider("Canny Threshold 1", 0, 500, 100)
canny_t2 = st.sidebar.slider("Canny Threshold 2", 0, 500, 200)
blur_k = st.sidebar.slider("Blur Kernel Size", 1, 25, 5, step=2)

# File Upload
uploaded_file = st.file_uploader("\U0001F4C1 Upload Image", type=["png", "jpg", "jpeg"])

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

def fuzzy_thresholding(img):
    gray = apply_grayscale(img).astype(np.float32)
    normalized = gray / 255.0
    low = fuzz.interp_membership([0, 0.5], [1, 0], normalized)
    high = fuzz.interp_membership([0.5, 1], [0, 1], normalized)
    combined = np.fmax(low, high)
    return np.uint8(combined * 255)

def fuzzy_contrast_enhancement(img):
    gray = apply_grayscale(img).astype(np.float32)
    normalized = gray / 255.0
    enhanced = fuzz.sigmf(normalized, 0.5, 10)
    return np.uint8(enhanced * 255)

def fuzzy_brightness_boost(img):
    gray = apply_grayscale(img).astype(np.float32) / 255.0
    dark = fuzz.interp_membership([0, 0.5], [1, 0], gray)
    boost = dark * 0.5 + gray
    return np.uint8(np.clip(boost, 0, 1) * 255)

def apply_canny(img, t1, t2):
    gray = apply_grayscale(img)
    return cv2.Canny(gray, t1, t2)

def apply_blur(img, ksize):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def apply_sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def apply_invert(img):
    return cv2.bitwise_not(img)

def apply_adaptive_thresh(img):
    gray = apply_grayscale(img)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

# Run Processor
if uploaded_file:
    image = Image.open(uploaded_file)
    img_np = np.array(image.convert("RGB"))
    st.image(img_np, caption="Original Image", use_column_width=True)

    if st.button("\u2728 Apply Tool"):
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
        elif tool == "Fuzzy Thresholding":
            processed = fuzzy_thresholding(img_np)
        elif tool == "Fuzzy Contrast Enhancement":
            processed = fuzzy_contrast_enhancement(img_np)
        elif tool == "Fuzzy Brightness Enhancement":
            processed = fuzzy_brightness_boost(img_np)
        elif tool == "Canny Edge Detection":
            processed = apply_canny(img_np, canny_t1, canny_t2)
        elif tool == "Gaussian Blur":
            processed = apply_blur(img_np, blur_k)
        elif tool == "Sharpening":
            processed = apply_sharpen(img_np)
        elif tool == "Invert Colors":
            processed = apply_invert(img_np)
        elif tool == "Adaptive Thresholding":
            processed = apply_adaptive_thresh(img_np)
        else:
            processed = img_np

        st.image(processed, caption="Processed Image", use_column_width=True)

        # Download button
        is_gray = len(processed.shape) == 2
        processed_bgr = processed if is_gray else cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".png", processed_bgr)
        b64 = base64.b64encode(buffer).decode()
        href = f'<a href="data:file/png;base64,{b64}" download="processed.png">\U0001F4E5 Download Result</a>'
        st.markdown(href, unsafe_allow_html=True)
