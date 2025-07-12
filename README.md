# Fuzzy Logic-Based Image Processor

This project is a Streamlit web app that provides a set of image processing tools powered by traditional and fuzzy logic-based techniques.

## Features

- User authentication and registration
- Grayscale conversion
- Gradient and edge detection
- Fuzzy logic for thresholding, brightness, contrast, and edge enhancement
- Multiple traditional filters (blur, sharpen, etc.)
- Easy image upload and processing interface

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

Or build with Docker (if provided):

```bash
docker build -t fuzzy-image-app .
docker run -p 8501:8501 fuzzy-image-app
```

## Additional System Dependencies

Install via:

```bash
sudo apt install -y $(< packages.txt)
```

## Run the App

```bash
streamlit run streamlit_app.py
```

## Author

Your Name - Mini Project 2025
