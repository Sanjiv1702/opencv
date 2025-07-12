#!/bin/bash
mkdir -p /var/lib/render

# Set all Streamlit server configurations via CLI
streamlit run streamlit_app.py \
  --server.port=$PORT \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --server.fileWatcherType=none \
  --browser.serverAddress="0.0.0.0" \
  --browser.gatherUsageStats=false
