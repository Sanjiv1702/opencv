streamlit_app.py

import streamlit as st import sqlite3 import hashlib import numpy as np import cv2 from PIL import Image from io import BytesIO import base64 from datetime import datetime import os import skfuzzy as fuzz

---------- Database Setup ----------

def create_users_table(): conn = sqlite3.connect("users.db") c = conn.cursor() c.execute('''CREATE TABLE IF NOT EXISTS users ( username TEXT PRIMARY KEY, password TEXT )''') c.execute('''CREATE TABLE IF NOT EXISTS uploads ( username TEXT, filename TEXT, timestamp TEXT )''') conn.commit() conn.close()

def hash_password(password): return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, password): conn = sqlite3.connect("users.db") c = conn.cursor() c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password))) conn.commit() conn.close()

def verify_user(username, password): conn = sqlite3.connect("users.db") c = conn.cursor() c.execute("SELECT password FROM users WHERE username = ?", (username,)) result = c.fetchone() conn.close() return result and result[0] == hash_password(password)

def log_upload(username, filename): conn = sqlite3.connect("users.db") c = conn.cursor() c.execute("INSERT INTO uploads (username, filename, timestamp) VALUES (?, ?, ?)", (username, filename, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))) conn.commit() conn.close()

def get_user_uploads(username): conn = sqlite3.connect("users.db") c = conn.cursor() c.execute("SELECT filename, timestamp FROM uploads WHERE username = ? ORDER BY timestamp DESC", (username,)) data = c.fetchall() conn.close() return data

---------- Image Processing Functions ----------

def apply_grayscale(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_gradient(img): gray = apply_grayscale(img) sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1) grad = np.hypot(sobelx, sobely) return np.uint8(grad / grad.max() * 255)

def fuzzy_edge_detection(img, low_w, med_w, high_w): gray = apply_grayscale(img) blurred = cv2.GaussianBlur(gray, (5, 5), 0) norm_img = blurred.astype(np.float32) / 255.0 low = np.clip((0.5 - norm_img) / 0.5, 0, 1) med = 1 - np.abs(norm_img - 0.5) * 2 high = np.clip((norm_img - 0.5) / 0.5, 0, 1) edge_strength = np.maximum(low * low_w, np.maximum(med * med_w, high * high_w)) return np.uint8(edge_strength * 255)

def fuzzy_thresholding(img): gray = apply_grayscale(img).astype(np.float32) normalized = gray / 255.0 low = fuzz.interp_membership([0, 0.5], [1, 0], normalized) high = fuzz.interp_membership([0.5, 1], [0, 1], normalized) combined = np.fmax(low, high) return np.uint8(combined * 255)

def fuzzy_contrast_enhancement(img): gray = apply_grayscale(img).astype(np.float32) normalized = gray / 255.0 enhanced = fuzz.sigmf(normalized, 0.5, 10) return np.uint8(enhanced * 255)

def fuzzy_brightness_boost(img): gray = apply_grayscale(img).astype(np.float32) / 255.0 dark = fuzz.interp_membership([0, 0.5], [1, 0], gray) boost = dark * 0.5 + gray return np.uint8(np.clip(boost, 0, 1) * 255)

def apply_canny(img, t1=100, t2=200): gray = apply_grayscale(img) return cv2.Canny(gray, t1, t2)

---------- UI & Main App Logic ----------

create_users_table() if 'username' not in st.session_state: st.session_state.username = None st.session_state.authenticated = False

Login/Signup block in MAIN AREA (not sidebar)

if not st.session_state.authenticated: st.title("🔐 Welcome to Fuzzy Image Processor") tab1, tab2 = st.tabs(["Login", "Signup"])

with tab1:
    st.subheader("Login")
    user = st.text_input("Username", key="login_user")
    passwd = st.text_input("Password", type='password', key="login_pass")
    if st.button("Login"):
        if verify_user(user, passwd):
            st.session_state.username = user
            st.session_state.authenticated = True
            st.success(f"✅ Logged in as {user}")
        else:
            st.error("❌ Invalid username or password")

with tab2:
    st.subheader("Signup")
    new_user = st.text_input("New Username", key="signup_user")
    new_pass = st.text_input("New Password", type='password', key="signup_pass")
    if st.button("Create Account"):
        try:
            add_user(new_user, new_pass)
            st.success("🎉 Account created! You can now log in.")
        except:
            st.error("⚠️ Username already exists.")

Main application UI (visible only after login)

if st.session_state.authenticated: st.title("🧠 Fuzzy Logic-Based Image Processing") st.markdown("Logged in as: %s" % st.session_state.username) if st.button("🔓 Logout"): st.session_state.authenticated = False st.session_state.username = None st.rerun()

st.sidebar.header("Tools & Parameters")
tool = st.sidebar.selectbox("Select Tool", [
    "Grayscale", "Gradient", "Fuzzy Edge Detection",
    "Fuzzy Thresholding", "Fuzzy Contrast Enhancement",
    "Fuzzy Brightness Enhancement", "Canny Edge Detection"
])

low_w = st.sidebar.slider("Fuzzy Low Weight", 0.0, 1.0, 0.3)
med_w = st.sidebar.slider("Fuzzy Medium Weight", 0.0, 1.0, 0.7)
high_w = st.sidebar.slider("Fuzzy High Weight", 0.0, 1.0, 0.4)

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_np = np.array(image.convert("RGB"))
    st.image(img_np, caption="Original Image", use_column_width=True)

    processed = None
    if tool == "Grayscale":
        processed = apply_grayscale(img_np)
    elif tool == "Gradient":
        processed = apply_gradient(img_np)
    elif tool == "Fuzzy Edge Detection":
        processed = fuzzy_edge_detection(img_np, low_w, med_w, high_w)
    elif tool == "Fuzzy Thresholding":
        processed = fuzzy_thresholding(img_np)
    elif tool == "Fuzzy Contrast Enhancement":
        processed = fuzzy_contrast_enhancement(img_np)
    elif tool == "Fuzzy Brightness Enhancement":
        processed = fuzzy_brightness_boost(img_np)
    elif tool == "Canny Edge Detection":
        processed = apply_canny(img_np)

    if processed is not None:
        st.image(processed, caption="Processed Image", use_column_width=True)
        log_upload(st.session_state.username, uploaded_file.name)

        ext = ".jpg"
        mime = "image/jpeg"
        processed_bgr = processed if len(processed.shape) == 2 else cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(ext, processed_bgr)
        b64 = base64.b64encode(buffer).decode()
        href = f'<a href="data:{mime};base64,{b64}" download="{uploaded_file.name}">📥 Download Result (JPG)</a>'
        st.markdown(href, unsafe_allow_html=True)

# Upload History
st.subheader("📜 Upload History")
uploads = get_user_uploads(st.session_state.username)
for filename, timestamp in uploads:
    st.markdown(f"- **{filename}** uploaded at _{timestamp}_")

