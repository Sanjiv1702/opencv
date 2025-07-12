#!/bin/bash
# Initialize data directory
mkdir -p data

# Set environment variables for Render
export RENDER=true
export PORT=${PORT:-10000}

# Start Streamlit
streamlit run streamlit_app.py \
  --server.port=$PORT \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
