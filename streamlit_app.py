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

# Initialize database
def init_db():
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

# Hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Register new user
def register_user(username, name, password, email):
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

# Verify user
def verify_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        return hash_password(password) == result[0]
    return False

# Authentication configuration
def get_auth_config():
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

# Initialize authenticator
def init_authenticator():
    config = get_auth_config()
    authenticator = Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    return authenticator

# Login widget
def login_widget():
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
        return False

# Registration widget
def registration_widget():
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
                    else:
                        st.error("Username already exists")

# Social login options
def social_login():
    st.markdown("### Or login with:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Google"):
            st.info("Google login coming soon!")
    
    with col2:
        if st.button("GitHub"):
            st.info("GitHub login coming soon!")
    
    with col3:
        if st.button("Twitter"):
            st.info("Twitter login coming soon!")

# Main auth function
def authenticate():
    init_db()
    
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    if not st.session_state['authenticated']:
        st.title("Fuzzy Image Processor - Login")
        if login_widget():
            st.experimental_rerun()
        registration_widget()
        social_login()
        st.stop()
    
    return st.session_state['username'], st.session_state['name']

# === Image Processing with Fuzzy Logic ===
st.title("Fuzzy Image Processor")
st.subheader("Upload an image to apply edge detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Sobel edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))

    st.image(image, caption='Original Image', use_column_width=True)
    st.image(magnitude, caption='Edge Detected Image', use_column_width=True)