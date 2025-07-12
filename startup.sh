#!/bin/bash

# 1. First create persistent storage directory
mkdir -p /var/lib/render

# 2. Convert tabs to spaces in Python file to prevent indentation errors
sed -i 's/\t/    /g' streamlit_app.py

# 3. Start Streamlit with all required server configurations
streamlit run streamlit_app.py \
  --server.port=$PORT \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --server.fileWatcherType=none \
  --browser.serverAddress="0.0.0.0" \
  --browser.gatherUsageStats=false
