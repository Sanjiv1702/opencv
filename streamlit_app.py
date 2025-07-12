import streamlit as st
from streamlit_authenticator import Authenticate
import yaml
from yaml.loader import SafeLoader
import os
import sqlite3
import hashlib
from datetime import datetime
import base64
from PIL import Image
import numpy as np
import cv2
import skfuzzy as fuzz

# ========== Authentication Functions ==========
def init_db():
    """Initialize the SQLite database for user storage"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, 
                  name TEXT, 
                  password TEXT, 
                  email TEXT,
                  created_at TIMESTAMP)''')
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, name, password, email):
    """Register a new user in the database"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_pw = hash_password(password)
    try:
        c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                  (username, name, hashed_pw, email, datetime.now()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    """Verify user credentials against the database"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        return hash_password(password) == result[0]
    return False

def get_auth_config():
    """Get or create authentication configuration file"""
    if not os.path.exists('auth_config.yaml'):
        with open('auth_config.yaml', 'w') as file:
            yaml.dump({
                'credentials': {
                    'usernames': {}
                },
                'cookie': {
                    'expiry_days': 30,
                    'key': 'random_signature_key',
                    'name': 'fuzzy_image_auth'
                },
                'preauthorized': {
                    'emails': []
                }
            }, file)
    
    with open('auth_config.yaml') as file:
        return yaml.load(file, Loader=SafeLoader)

def init_authenticator():
    """Initialize the authenticator with updated parameters"""
    config = get_auth_config()
    authenticator = Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
    return authenticator

def login_widget():
    """Display login widget with updated authenticator usage"""
    authenticator = init_authenticator()
    name, authentication_status, username = authenticator.login('Login', 'main')
    
    if authentication_status:
        st.session_state['authenticated'] = True
        st.session_state['username'] = username
        st.session_state['name'] = name
        return True
    elif authentication_status is False:
        st.error('Username/password is incorrect')
        return False
    elif authentication_status is None:
        st.warning('Please enter your username and password')
        return False

def registration_widget(authenticator):
    """Display registration form"""
    with st.expander("Don't have an account? Register here"):
        with st.form("registration_form"):
            username = st.text_input("Username")
            name = st.text_input("Full name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            
            submitted = st.form_submit_button("Register")
            if submitted:
                if password != confirm_password:
                    st.error("Passwords don't match")
                else:
                    if register_user(username, name, password, email):
                        st.success("Registration successful! Please login.")
                        # Update auth config
                        config = get_auth_config()
                        config['credentials']['usernames'][username] = {
                            'name': name,
                            'password': hash_password(password),
                            'email': email
                        }
                        with open('auth_config.yaml', 'w') as file:
                            yaml.dump(config, file)
                        # Register with authenticator
                        authenticator.register_user(username, name, password, email)
                    else:
                        st.error("Username already exists")

def authenticate():
    """Main authentication function with improved error handling"""
    init_db()
    authenticator = init_authenticator()
    
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    if not st.session_state['authenticated']:
        st.title("Fuzzy Image Processor - Login")
        login_success = login_widget()
        registration_widget(authenticator)
        if login_success:
            st.experimental_rerun()
        st.stop()
    
    return st.session_state['username'], st.session_state['name']

# ========== Image Processing Functions ==========
def apply_grayscale(img):
    """Convert image to grayscale"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_gradient(img):
    """Apply Sobel gradient"""
    gray = apply_grayscale(img)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad = np.hypot(sobelx, sobely)
    return np.uint8(grad / grad.max() * 255)

def apply_threshold(img, t):
    """Apply simple thresholding"""
    gray = apply_grayscale(img)
    _, thresh = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    return thresh

def apply_hist_eq(img):
    """Apply histogram equalization"""
    gray = apply_grayscale(img)
    return cv2.equalizeHist(gray)

def fuzzy_edge_detection(img, low_w, med_w, high_w):
    """Fuzzy logic-based edge detection"""
    gray = apply_grayscale(img)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    norm_img = blurred.astype(np.float32) / 255.0
    low = np.clip((0.5 - norm_img) / 0.5, 0, 1)
    med = 1 - np.abs(norm_img - 0.5) * 2
    high = np.clip((norm_img - 0.5) / 0.5, 0, 1)
    edge_strength = np.maximum(low * low_w, np.maximum(med * med_w, high * high_w))
    return np.uint8(edge_strength * 255)

def fuzzy_thresholding(img):
    """Fuzzy logic-based thresholding"""
    gray = apply_grayscale(img).astype(np.float32)
    normalized = gray / 255.0
    low = fuzz.interp_membership([0, 0.5], [1, 0], normalized)
    high = fuzz.interp_membership([0.5, 1], [0, 1], normalized)
    combined = np.fmax(low, high)
    return np.uint8(combined * 255)

def fuzzy_contrast_enhancement(img):
    """Fuzzy logic-based contrast enhancement"""
    gray = apply_grayscale(img).astype(np.float32)
    normalized = gray / 255.0
    enhanced = fuzz.sigmf(normalized, 0.5, 10)
    return np.uint8(enhanced * 255)

def fuzzy_brightness_boost(img):
    """Fuzzy logic-based brightness enhancement"""
    gray = apply_grayscale(img).astype(np.float32) / 255.0
    dark = fuzz.interp_membership([0, 0.5], [1, 0], gray)
    boost = dark * 0.5 + gray
    return np.uint8(np.clip(boost, 0, 1) * 255)

def apply_canny(img, t1, t2):
    """Apply Canny edge detection"""
    gray = apply_grayscale(img)
    return cv2.Canny(gray, t1, t2)

def apply_blur(img, ksize):
    """Apply Gaussian blur"""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def apply_sharpen(img):
    """Apply sharpening filter"""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def apply_invert(img):
    """Invert image colors"""
    return cv2.bitwise_not(img)

def apply_adaptive_thresh(img):
    """Apply adaptive thresholding"""
    gray = apply_grayscale(img)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

# ========== Main Application ==========
def main():
    # Authenticate user
    username, name = authenticate()
    
    # Configure page
    st.set_page_config(page_title="Fuzzy Image Processor", layout="centered")
    
    # Display welcome message and logout button
    st.sidebar.title(f"Welcome, {name}!")
    authenticator = init_authenticator()
    if st.sidebar.button("Logout"):
        authenticator.logout('Logout', 'main')
        st.session_state['authenticated'] = False
        st.experimental_rerun()
    
    # Main title and description
    st.title("🧠 Fuzzy Logic-Based Image Processing")
    st.markdown(f"""
    Welcome back, **{name}**! This application demonstrates soft computing principles using fuzzy logic and OpenCV. 
    It includes various image processing tools like fuzzy edge detection, thresholding, and contrast enhancement.
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
    
    # Parameters for different tools
    threshold = st.sidebar.slider("Threshold", 0, 255, 127)
    low_w = st.sidebar.slider("Fuzzy Low Weight", 0.0, 1.0, 0.3)
    med_w = st.sidebar.slider("Fuzzy Medium Weight", 0.0, 1.0, 0.7)
    high_w = st.sidebar.slider("Fuzzy High Weight", 0.0, 1.0, 0.4)
    canny_t1 = st.sidebar.slider("Canny Threshold 1", 0, 500, 100)
    canny_t2 = st.sidebar.slider("Canny Threshold 2", 0, 500, 200)
    blur_k = st.sidebar.slider("Blur Kernel Size", 1, 25, 5, step=2)
    
    # Download format
    download_format = st.sidebar.selectbox("Download Format", ["JPG", "PNG"])
    
    # File Upload
    uploaded_file = st.file_uploader("\U0001F4C1 Upload Image", type=["png", "jpg", "jpeg"])
    
    # Process image when uploaded
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
            
            # Download button with format option
            is_gray = len(processed.shape) == 2
            processed_bgr = processed if is_gray else cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
            ext = ".jpg" if download_format == "JPG" else ".png"
            mime = "image/jpeg" if download_format == "JPG" else "image/png"
            _, buffer = cv2.imencode(ext, processed_bgr)
            b64 = base64.b64encode(buffer).decode()
            href = f'<a href="data:{mime};base64,{b64}" download="processed{ext}">📥 Download Result ({download_format})</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
