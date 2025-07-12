#!/bin/bash

# 1. Create directories
mkdir -p /var/lib/render

# 2. Force-clean the Python file
sed -i 's/\t/    /g' streamlit_app.py  # Convert tabs to spaces
sed -i 's/^    /\t/g' streamlit_app.py  # Convert back if needed (safety)
sed -i 's/\t/    /g' streamlit_app.py  # Final conversion to spaces

# 3. Verify indentation
if grep -n $'\t' streamlit_app.py; then
    echo "ERROR: Tabs still found in these lines:"
    grep -n $'\t' streamlit_app.py
    exit 1
fi

# 4. Start Streamlit
exec streamlit run streamlit_app.py \
  --server.port=$PORT \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --server.fileWatcherType=none \
  --browser.serverAddress="0.0.0.0" \
  --browser.gatherUsageStats=false
