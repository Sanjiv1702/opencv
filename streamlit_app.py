
import streamlit as st
import streamlit_authenticator as stauth
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

# DB setup
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        name TEXT,
        password TEXT,
        email TEXT,
        created_at TIMESTAMP
    )""")
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

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

def verify_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        return hash_password(password) == result[0]
    return False

def setup_authentication():
    init_db()
    if not os.path.exists('auth_config.yaml'):
        with open('auth_config.yaml', 'w') as file:
            yaml.dump({
                'credentials': {
                    'usernames': {
                        'demo_user': {
                            'name': 'Demo User',
                            'email': 'demo@example.com',
                            'password': hash_password('password')
                        }
                    },
                    'cookie': {
                        'expiry_days': 30,
                        'key': 'fuzzy_key',
                        'name': 'fuzzy_cookie'
                    }
                }
            }, file)

    with open('auth_config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['credentials']['cookie']['name'],
        config['credentials']['cookie']['key'],
        config['credentials']['cookie']['expiry_days']
    )
    return authenticator, config

# UI
def show_login(authenticator):
    name, auth_status, username = authenticator.login("Login", "main")
    if auth_status:
        st.session_state['authenticated'] = True
        st.session_state['username'] = username
        st.session_state['name'] = name
        st.experimental_rerun()
    elif auth_status is False:
        st.error("Incorrect username or password.")
    elif auth_status is None:
        st.warning("Enter your credentials.")

def apply_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def main():
    st.set_page_config(page_title="Fuzzy Image App", layout="centered")
    authenticator, config = setup_authentication()

    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        st.title("🔐 Fuzzy Image Processor Login")
        show_login(authenticator)
        st.stop()

    st.title("🧠 Fuzzy Image Processing")
    st.markdown(f"Welcome, **{st.session_state['name']}**!")

    if st.sidebar.button("Logout"):
        authenticator.logout("Logout", "main")
        st.session_state['authenticated'] = False
        st.experimental_rerun()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_np = np.array(image.convert("RGB"))
        st.image(img_np, caption="Original Image", use_column_width=True)

        if st.button("Convert to Grayscale"):
            result = apply_grayscale(img_np)
            st.image(result, caption="Processed Image", use_column_width=True)

if __name__ == "__main__":
    main()
