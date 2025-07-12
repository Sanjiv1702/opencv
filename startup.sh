#!/bin/bash

# 1. Create persistent storage
mkdir -p /var/lib/render

# 2. FIX INDENTATION ERRORS (critical for Render)
sed -i 's/\t/    /g' streamlit_app.py          # Convert tabs to spaces
sed -i 's/[[:space:]]*$//' streamlit_app.py    # Remove trailing whitespace
dos2unix streamlit_app.py                      # Fix line endings (if needed)

# 3. Start Streamlit with Render-optimized settings
exec streamlit run streamlit_app.py \
  --server.port=$PORT \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --server.fileWatcherType=none \
  --browser.serverAddress="0.0.0.0" \
  --browser.gatherUsageStats=false
