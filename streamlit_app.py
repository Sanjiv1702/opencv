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
# Use environment variables (set in Render Dashboard)
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")  # Default to Gmail
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
APP_URL = os.getenv("APP_URL", "https://your-app-name.onrender.com")  # Your Render URL

# ========== Database Setup ==========
def get_db_path():
    # Store DB in a persistent directory
    db_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, 'users.db')

def init_db():
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
    conn.close()

# ========== Authentication Functions ==========
def hash_password(password):
    return stauth.Hasher([password]).generate()[0]  # Uses bcrypt

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

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
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
    with st.expander("Forgot Password"):
        email = st.text_input("Enter your registered email")
        
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
                Please click the link below to reset your password:
                
                {reset_link}
                
                This link will expire in 1 hour.
                
                If you didn't request this, please ignore this email.
                """
                
                if send_email(email, "Password Reset Request", email_body):
                    st.success("Password reset link sent to your email!")
                else:
                    st.error("Failed to send reset email. Please try again later.")
            else:
                st.error("No account found with this email.")

def show_forgot_username():
    with st.expander("Forgot Username"):
        email = st.text_input("Enter your registered email")
        
        if st.button("Recover Username"):
            user = get_user_by_email(email)
            if user:
                username, name = user
                email_body = f"""
                Hello {name},
                
                You requested to recover your username.
                Your username is: {username}
                
                If you didn't request this, please ignore this email.
                """
                
                if send_email(email, "Username Recovery", email_body):
                    st.success("Username sent to your registered email!")
                else:
                    st.error("Failed to send email. Please try again later.")
            else:
                st.error("No account found with this email.")

def show_reset_password(email, token):
    if not validate_reset_token(email, token):
        st.error("Invalid or expired reset token.")
        return
    
    with st.form("reset_password_form"):
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        
        if st.form_submit_button("Reset Password"):
            if new_password != confirm_password:
                st.error("Passwords don't match!")
            else:
                user = get_user_by_email(email)
                if user:
                    username, _ = user
                    update_password(username, new_password)
                    set_reset_token(email, None, None)  # Clear the token
                    st.success("Password reset successfully! Please login with your new password.")
                    st.experimental_rerun()

def show_registration(authenticator, config):
    with st.expander("Register New Account"):
        with st.form("register_form"):
            username = st.text_input("Username")
            name = st.text_input("Full Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            if st.form_submit_button("Register"):
                if password != confirm_password:
                    st.error("Passwords do not match!")
                else:
                    if register_user(username, name, password, email):
                        st.success("Registration successful! Please login.")
                        # Update config
                        config['credentials']['usernames'][username] = {
                            'name': name,
                            'password': hash_password(password),
                            'email': email
                        }
                        # Save config to persistent storage
                        config_path = os.path.join(os.getcwd(), 'data/auth_config.yaml')
                        os.makedirs(os.path.dirname(config_path), exist_ok=True)
                        with open(config_path, 'w') as file:
                            yaml.dump(config, file)
                        st.experimental_rerun()
                    else:
                        st.error("Username already exists!")

# ========== Image Processing Functions ==========
# [Keep all your existing image processing functions here]
# ... (Copy all your image processing functions unchanged)

# ========== Main Application ==========
def main():
    st.set_page_config(page_title="Fuzzy Image Processor", layout="centered")
    
    # Handle port for Render
    if 'PORT' in os.environ:
        st.set_option('server.port', int(os.environ['PORT']))
    
    # Check for password reset
    query_params = st.experimental_get_query_params()
    if 'email' in query_params and 'token' in query_params:
        email = query_params['email'][0]
        token = query_params['token'][0]
        show_reset_password(email, token)
        return
    
    # Initialize authentication
    init_db()
    config_path = os.path.join(os.getcwd(), 'data/auth_config.yaml')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    if not os.path.exists(config_path):
        with open(config_path, 'w') as file:
            yaml.dump({
                'credentials': {
                    'usernames': {},
                    'cookie': {
                        'expiry_days': 30,
                        'key': 'random_signature_key',
                        'name': 'fuzzy_image_auth'
                    }
                }
            }, file)
    
    with open(config_path) as file:
        config = yaml.load(file, Loader=SafeLoader)
    
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config.get('preauthorized', None)
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
    
    if st.sidebar.button("Logout"):
        authenticator.logout('Logout', 'main')
        st.session_state['authenticated'] = False
        st.experimental_rerun()
    
    # [Rest of your existing main application code]
    # ... (Copy all your image processing UI code here)

if __name__ == "__main__":
    main()
