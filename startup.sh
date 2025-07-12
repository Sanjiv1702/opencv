#!/bin/bash
mkdir -p /var/lib/render  # For persistent storage on Render
streamlit run streamlit_app.py \
  --server.port=$PORT \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
