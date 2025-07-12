#!/bin/bash
# 1. Create persistent storage
mkdir -p /var/lib/render && chmod 755 /var/lib/render

# 2. Then run your app
exec streamlit run streamlit_app.py \
  --server.port=$PORT \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --server.fileWatcherType=none \
  --browser.serverAddress="0.0.0.0" \
  --browser.gatherUsageStats=false
