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

# ========== Configuration ==========
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
APP_URL = os.getenv("APP_URL", "https://your-app-name.onrender.com")

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
        return stauth.authenticate_user(username, password, {'usernames': {username: {'password': result[0]}})
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
def apply_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_gradient(img):
    gray = apply_grayscale(img)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
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
    st.set_page_config(page_title="Fuzzy Image Processor", layout="centered")
    
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
        st.title("Fuzzy Image Processor - Login")
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
    st.title("🧠 Fuzzy Logic-Based Image Processing")
    st.markdown(f"Welcome, **{st.session_state['name']}**!")
    
    if st.sidebar.button("🚪 Logout"):
        authenticator.logout('Logout', 'main')
        st.session_state['authenticated'] = False
        st.experimental_rerun()
    
    # Tool selection
    st.sidebar.header("⚙️ Tools & Parameters")
    tool = st.sidebar.selectbox("Select Tool", [
        "Grayscale", "Gradient", "Fuzzy Edge Detection",
        "Fuzzy Thresholding", "Fuzzy Contrast Enhancement", 
        "Fuzzy Brightness Enhancement", "Thresholding", 
        "Histogram Equalization", "Canny Edge Detection", 
        "Gaussian Blur", "Sharpening", "Invert Colors", 
        "Adaptive Thresholding"
    ])
    
    # Parameters
    threshold = st.sidebar.slider("Threshold", 0, 255, 127)
    low_w = st.sidebar.slider("Fuzzy Low Weight", 0.0, 1.0, 0.3)
    med_w = st.sidebar.slider("Fuzzy Medium Weight", 0.0, 1.0, 0.7)
    high_w = st.sidebar.slider("Fuzzy High Weight", 0.0, 1.0, 0.4)
    canny_t1 = st.sidebar.slider("Canny Threshold 1", 0, 500, 100)
    canny_t2 = st.sidebar.slider("Canny Threshold 2", 0, 500, 200)
    blur_k = st.sidebar.slider("Blur Kernel Size", 1, 25, 5, step=2)
    
    # File upload
    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_np = np.array(image.convert("RGB"))
        st.image(img_np, caption="Original Image", use_column_width=True)
        
        if st.button("✨ Process Image"):
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
            
            st.image(processed, caption="Processed Image", use_column_width=True)
            
            # Download option
            is_gray = len(processed.shape) == 2
            processed_bgr = processed if is_gray else cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode(".jpg", processed_bgr)
            b64 = base64.b64encode(buffer).decode()
            href = f'<a href="data:image/jpeg;base64,{b64}" download="processed.jpg">💾 Download Result</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
