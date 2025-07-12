# streamlit_app.py
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
import sqlite3
from datetime import datetime, timedelta
import base64
from PIL import Image
import numpy as np
import cv2
import skfuzzy as fuzz
import smtplib
from email.mime.text import MIMEText
import secrets
import string
from skimage import io, exposure
import mahotas as mh
import pandas as pd
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
# TO (lazy load inside functions):
def get_dl_features(img, model_name="VGG16"):
    import tensorflow as tf  # ← Import only when needed
    from tensorflow.keras.applications import VGG16
    model = VGG16(weights='imagenet')
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
import imageio.v3 as iio

# ========== Configuration ==========
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
APP_URL = os.getenv("APP_URL", "https://your-app-name.onrender.com")

MODELS = {
    "VGG16": VGG16,
    "ResNet50": ResNet50
}

# ========== Database Setup ==========
def get_db_path():
    """Get persistent database path for Render or local development"""
    if os.environ.get('RENDER'):
        path = '/var/lib/render/users.db'
    else:
        path = os.path.join(os.getcwd(), 'data', 'users.db')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def init_db():
    try:
        conn = sqlite3.connect(get_db_path())
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (username TEXT PRIMARY KEY, 
                      name TEXT, 
                      password TEXT, 
                      email TEXT,
                      reset_token TEXT,
                      token_expiry TIMESTAMP,
                      created_at TIMESTAMP)''')
        conn.commit()
    except Exception as e:
        st.error(f"Database initialization failed: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

# ========== Authentication Functions ==========
def hash_password(password):
    return stauth.Hasher([password]).generate()[0]

def register_user(username, name, password, email):
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    hashed_pw = hash_password(password)
    try:
        c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)",
                 (username, name, hashed_pw, email, None, None, datetime.now()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        return stauth.authenticate_user(username, password, {'usernames': {username: {'password': result[0]}}})
    return False

def get_user_by_email(email):
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("SELECT username, name FROM users WHERE email=?", (email,))
    result = c.fetchone()
    conn.close()
    return result if result else None

def update_password(username, new_password):
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    hashed_pw = hash_password(new_password)
    c.execute("UPDATE users SET password=? WHERE username=?", (hashed_pw, username))
    conn.commit()
    conn.close()

def set_reset_token(email, token, expiry):
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("UPDATE users SET reset_token=?, token_expiry=? WHERE email=?", (token, expiry, email))
    conn.commit()
    conn.close()

def validate_reset_token(email, token):
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    c.execute("SELECT reset_token, token_expiry FROM users WHERE email=?", (email,))
    result = c.fetchone()
    conn.close()
    
    if not result or not result[0] or not result[1]:
        return False
    
    stored_token, expiry = result
    if stored_token != token or datetime.now() > datetime.strptime(expiry, "%Y-%m-%d %H:%M:%S.%f"):
        return False
    return True

# ========== Email Functions ==========
def send_email(to_email, subject, body):
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to_email

        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False

def generate_reset_token():
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))

# ========== UI Components ==========
def show_forgot_password():
    with st.expander("🔑 Forgot Password"):
        email = st.text_input("Enter your registered email", key="reset_email")
        
        if st.button("Send Reset Link"):
            user = get_user_by_email(email)
            if user:
                username, name = user
                token = generate_reset_token()
                expiry = datetime.now() + timedelta(hours=1)
                set_reset_token(email, token, expiry)
                
                reset_link = f"{APP_URL}/?email={email}&token={token}"
                email_body = f"""
                Hello {name},
                
                You requested a password reset for your account (username: {username}).
                Click this link to reset your password (expires in 1 hour):
                
                {reset_link}
                
                If you didn't request this, please ignore this email.
                """
                
                if send_email(email, "Password Reset Request", email_body):
                    st.success("📩 Reset link sent! Check your email.")
                else:
                    st.error("❌ Failed to send email. Please try again later.")
            else:
                st.error("❌ No account found with this email.")

def show_forgot_username():
    with st.expander("🧠 Forgot Username"):
        email = st.text_input("Enter your registered email", key="recover_username")
        
        if st.button("Recover Username"):
            user = get_user_by_email(email)
            if user:
                username, name = user
                email_body = f"""
                Hello {name},
                
                Your username is: {username}
                
                If you didn't request this, please ignore this email.
                """
                
                if send_email(email, "Username Recovery", email_body):
                    st.success("📩 Username sent to your email!")
                else:
                    st.error("❌ Failed to send email. Please try again later.")
            else:
                st.error("❌ No account found with this email.")

def show_reset_password(email, token):
    st.title("🔐 Reset Password")
    if not validate_reset_token(email, token):
        st.error("Invalid or expired reset token.")
        return
    
    with st.form("reset_password_form"):
        new_password = st.text_input("New Password", type="password", key="new_pass")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_pass")
        
        if st.form_submit_button("Reset Password"):
            if new_password != confirm_password:
                st.error("Passwords don't match!")
            else:
                user = get_user_by_email(email)
                if user:
                    username, _ = user
                    update_password(username, new_password)
                    set_reset_token(email, None, None)
                    st.success("✅ Password reset successfully! Please login.")
                    st.balloons()
                    st.experimental_rerun()

def show_registration(authenticator, config):
    with st.expander("📝 Register New Account"):
        with st.form("register_form"):
            username = st.text_input("Username", key="reg_user")
            name = st.text_input("Full Name", key="reg_name")
            email = st.text_input("Email", key="reg_email")
            password = st.text_input("Password", type="password", key="reg_pass")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_conf_pass")
            
            if st.form_submit_button("Register"):
                if password != confirm_password:
                    st.error("Passwords don't match!")
                else:
                    if register_user(username, name, password, email):
                        st.success("🎉 Registration successful! Please login.")
                        config['credentials']['usernames'][username] = {
                            'name': name,
                            'password': hash_password(password),
                            'email': email
                        }
                        config_path = get_config_path()
                        with open(config_path, 'w') as file:
                            yaml.dump(config, file)
                        st.experimental_rerun()
                    else:
                        st.error("❌ Username already exists!")

# ========== Image Processing Functions ==========
class FuzzySystems:
    @staticmethod
    def fuzzy_edge_detection(img, low_w=0.3, med_w=0.7, high_w=0.4):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        norm_img = blurred.astype(np.float32) / 255.0
        
        # Fuzzy membership functions
        low = np.clip((0.5 - norm_img) / 0.5, 0, 1)
        med = 1 - np.abs(norm_img - 0.5) * 2
        high = np.clip((norm_img - 0.5) / 0.5, 0, 1)
        
        # Fuzzy inference
        edge_strength = np.maximum(low * low_w, np.maximum(med * med_w, high * high_w))
        return (edge_strength * 255).astype(np.uint8)

    @staticmethod
    def fuzzy_contrast_enhancement(img, a=0.5, b=10):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        enhanced = fuzz.sigmf(gray, a, b)
        return (enhanced * 255).astype(np.uint8)

    @staticmethod
    def fuzzy_image_segmentation(img, n_clusters=3):
        pixels = img.reshape((-1, 3)).astype(np.float32)
        
        # Initialize fuzzy c-means
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            pixels.T, n_clusters, 2, error=0.005, maxiter=1000
        )
        
        # Assign clusters
        cluster_membership = np.argmax(u, axis=0)
        return cntr[cluster_membership].reshape(img.shape).astype(np.uint8)

class AdvancedImageProcessor:
    def __init__(self):
        self.dl_models = {
            "VGG16": VGG16(weights='imagenet'),
            "ResNet50": ResNet50(weights='imagenet')
        }

    # ===== Basic Operations =====
    @staticmethod
    def apply_grayscale(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def apply_gradient(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.hypot(sobelx, sobely)
        return np.uint8(grad / grad.max() * 255)

    @staticmethod
    def apply_threshold(img, t):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        return thresh

    @staticmethod
    def apply_hist_eq(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)

    @staticmethod
    def apply_canny(img, t1, t2):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, t1, t2)

    @staticmethod
    def apply_blur(img, ksize):
        return cv2.GaussianBlur(img, (ksize, ksize), 0)

    @staticmethod
    def apply_sharpen(img):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def apply_invert(img):
        return cv2.bitwise_not(img)

    @staticmethod
    def apply_adaptive_thresh(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    @staticmethod
    def adjust_gamma(img, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)

    # ===== Feature Extraction =====
    @staticmethod
    def extract_haralick_features(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = mh.features.haralick(gray).mean(axis=0)
        return pd.DataFrame([features], columns=[f"Haralick_{i}" for i in range(13)])

    # ===== Deep Learning =====
    def extract_dl_features(self, img, model_name="VGG16"):
        model = self.dl_models[model_name]
        img = cv2.resize(img, (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        return features.flatten()

    # ===== Panorama Stitching =====
    @staticmethod
    def stitch_images(images):
        stitcher = cv2.Stitcher_create()
        status, panorama = stitcher.stitch(images)
        if status == cv2.Stitcher_OK:
            return panorama
        else:
            raise ValueError("Image stitching failed")

    # ===== 3D Visualization =====
    @staticmethod
    def create_3d_projection(img, elevation=30, azimuth=45):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x, y = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, gray, cmap='viridis')
        ax.view_init(elev=elevation, azim=azimuth)
        return fig

    # ===== Batch Processing =====
    @staticmethod
    def batch_process(files, operation, params={}):
        results = []
        for file in files:
            img = iio.imread(file)
            processed = getattr(AdvancedImageProcessor, operation)(img, **params)
            results.append(processed)
        return results

def show_image_processing_tools():
    st.sidebar.header("🔧 Image Processing Tools")
    tool = st.sidebar.selectbox("Select Tool", [
        "Basic Operations", "Fuzzy Logic", "Feature Extraction", 
        "Deep Learning", "Panorama Stitching", "3D Visualization",
        "Batch Processing", "Classic Tools"
    ])

    processor = AdvancedImageProcessor()
    fuzzy = FuzzySystems()
    
    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        try:
            img = np.array(Image.open(uploaded_file).convert("RGB"))
            st.image(img, caption="Original Image", use_column_width=True)
            
            if st.button("✨ Process Image"):
                try:
                    with st.spinner("Processing..."):
                        processed = None
                        
                        if tool == "Basic Operations":
                            col1, col2 = st.columns(2)
                            with col1:
                                gamma = st.slider("Gamma Correction", 0.1, 3.0, 1.0, 0.1)
                            with col2:
                                blur = st.slider("Gaussian Blur", 0, 15, 0, 2)
                            
                            processed = processor.adjust_gamma(img, gamma)
                            if blur > 0:
                                processed = cv2.GaussianBlur(processed, (blur, blur), 0)
                        
                        elif tool == "Fuzzy Logic":
                            method = st.selectbox("Fuzzy Method", [
                                "Edge Detection", "Contrast Enhancement", "Image Segmentation"
                            ])
                            
                            if method == "Edge Detection":
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    low_w = st.slider("Low Weight", 0.0, 1.0, 0.3)
                                with col2:
                                    med_w = st.slider("Medium Weight", 0.0, 1.0, 0.7)
                                with col3:
                                    high_w = st.slider("High Weight", 0.0, 1.0, 0.4)
                                processed = fuzzy.fuzzy_edge_detection(img, low_w, med_w, high_w)
                            
                            elif method == "Contrast Enhancement":
                                a = st.slider("Alpha", 0.1, 1.0, 0.5)
                                b = st.slider("Beta", 1, 20, 10)
                                processed = fuzzy.fuzzy_contrast_enhancement(img, a, b)
                            
                            else:  # Segmentation
                                n_clusters = st.slider("Number of Clusters", 2, 10, 3)
                                processed = fuzzy.fuzzy_image_segmentation(img, n_clusters)
                        
                        elif tool == "Feature Extraction":
                            features = processor.extract_haralick_features(img)
                            st.dataframe(features)
                            return
                        
                        elif tool == "Deep Learning":
                            model = st.selectbox("Select Model", list(MODELS.keys()))
                            features = processor.extract_dl_features(img, model)
                            st.text(f"Feature Vector Length: {len(features)}")
                            st.line_chart(features[:50])
                            return
                        
                        elif tool == "Panorama Stitching":
                            uploaded_files = st.file_uploader(
                                "Upload multiple images for stitching", 
                                type=["jpg", "jpeg", "png"], 
                                accept_multiple_files=True
                            )
                            if uploaded_files and len(uploaded_files) >= 2:
                                images = [np.array(Image.open(file).convert("RGB")) for file in uploaded_files]
                                processed = processor.stitch_images(images)
                        
                        elif tool == "3D Visualization":
                            col1, col2 = st.columns(2)
                            with col1:
                                elevation = st.slider("Elevation", 0, 90, 30)
                            with col2:
                                azimuth = st.slider("Azimuth", 0, 360, 45)
                            
                            fig = processor.create_3d_projection(img, elevation, azimuth)
                            st.pyplot(fig)
                            return
                        
                        elif tool == "Batch Processing":
                            uploaded_files = st.file_uploader(
                                "Upload multiple images", 
                                type=["jpg", "jpeg", "png"], 
                                accept_multiple_files=True
                            )
                            if uploaded_files:
                                operation = st.selectbox("Select Operation", [
                                    "adjust_gamma", "apply_grayscale", "apply_threshold"
                                ])
                                params = {}
                                if operation == "adjust_gamma":
                                    params['gamma'] = st.slider("Gamma", 0.1, 3.0, 1.0)
                                elif operation == "apply_threshold":
                                    params['t'] = st.slider("Threshold", 0, 255, 127)
                                
                                results = processor.batch_process(uploaded_files, operation, params)
                                for i, result in enumerate(results):
                                    st.image(result, caption=f"Processed Image {i+1}", use_column_width=True)
                        
                        elif tool == "Classic Tools":
                            st.sidebar.header("⚙️ Parameters")
                            threshold = st.sidebar.slider("Threshold", 0, 255, 127)
                            canny_t1 = st.sidebar.slider("Canny Threshold 1", 0, 500, 100)
                            canny_t2 = st.sidebar.slider("Canny Threshold 2", 0, 500, 200)
                            blur_k = st.sidebar.slider("Blur Kernel Size", 1, 25, 5, step=2)
                            
                            classic_tool = st.selectbox("Select Classic Tool", [
                                "Grayscale", "Gradient", "Thresholding", 
                                "Histogram Equalization", "Canny Edge Detection", 
                                "Gaussian Blur", "Sharpening", "Invert Colors", 
                                "Adaptive Thresholding"
                            ])
                            
                            if classic_tool == "Grayscale":
                                processed = processor.apply_grayscale(img)
                            elif classic_tool == "Gradient":
                                processed = processor.apply_gradient(img)
                            elif classic_tool == "Thresholding":
                                processed = processor.apply_threshold(img, threshold)
                            elif classic_tool == "Histogram Equalization":
                                processed = processor.apply_hist_eq(img)
                            elif classic_tool == "Canny Edge Detection":
                                processed = processor.apply_canny(img, canny_t1, canny_t2)
                            elif classic_tool == "Gaussian Blur":
                                processed = processor.apply_blur(img, blur_k)
                            elif classic_tool == "Sharpening":
                                processed = processor.apply_sharpen(img)
                            elif classic_tool == "Invert Colors":
                                processed = processor.apply_invert(img)
                            elif classic_tool == "Adaptive Thresholding":
                                processed = processor.apply_adaptive_thresh(img)
                        
                        # Display processed image
                        if tool not in ["Feature Extraction", "Deep Learning", "3D Visualization"] and processed is not None:
                            st.image(processed, caption="Processed Image", use_column_width=True)
                            
                            # Download option
                            is_gray = len(processed.shape) == 2
                            processed_bgr = processed if is_gray else cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                            _, buffer = cv2.imencode(".jpg", processed_bgr)
                            b64 = base64.b64encode(buffer).decode()
                            href = f'<a href="data:image/jpeg;base64,{b64}" download="processed.jpg">💾 Download Result</a>'
                            st.markdown(href, unsafe_allow_html=True)
                            
                except cv2.error as e:
                    st.error(f"OpenCV Error: {str(e)}")
                except Exception as e:
                    st.error(f"Processing Error: {str(e)}")
                    
        except Exception as e:
            st.error(f"Invalid image file: {str(e)}")
            st.stop()  # Halt execution on critical errors

# ========== Main Application ==========
def get_config_path():
    if os.environ.get('RENDER'):
        path = '/var/lib/render/auth_config.yaml'
    else:
        path = os.path.join(os.getcwd(), 'data', 'auth_config.yaml')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def main():
    # Initialize page config
    st.set_page_config(page_title="Advanced Fuzzy Image Processor", layout="wide")
    
    # Check for password reset
    query_params = st.experimental_get_query_params()
    if 'email' in query_params and 'token' in query_params:
        show_reset_password(query_params['email'][0], query_params['token'][0])
        return
    
    # Initialize authentication
    init_db()
    config_path = get_config_path()
    
    if not os.path.exists(config_path):
        with open(config_path, 'w') as file:
            yaml.dump({
                'credentials': {'usernames': {}},
                'cookie': {
                    'expiry_days': 30,
                    'key': 'random_signature_key',
                    'name': 'fuzzy_image_auth'
                }
            }, file)
    
    with open(config_path) as file:
        config = yaml.load(file, Loader=SafeLoader)
    
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
    
    # Authentication check
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    if not st.session_state['authenticated']:
        st.title("Advanced Fuzzy Image Processor - Login")
        name, authentication_status, username = authenticator.login('Login', 'main')
        
        if authentication_status:
            st.session_state['authenticated'] = True
            st.session_state['username'] = username
            st.session_state['name'] = name
            st.experimental_rerun()
        elif authentication_status is False:
            st.error('Username/password is incorrect')
        elif authentication_status is None:
            st.warning('Please enter your username and password')
        
        show_forgot_password()
        show_forgot_username()
        show_registration(authenticator, config)
        st.stop()
    
    # Main app after authentication
    st.title("🧠 Advanced Fuzzy Logic-Based Image Processing")
    st.markdown(f"Welcome, **{st.session_state['name']}**!")
    
    if st.sidebar.button("🚪 Logout"):
        authenticator.logout('Logout', 'main')
        st.session_state['authenticated'] = False
        st.experimental_rerun()
    
    show_image_processing_tools()

if __name__ == "__main__":
    main()
