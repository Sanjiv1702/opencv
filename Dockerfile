FROM python:3.10-slim

# ===== OPTION 3 PART 1: System Dependencies =====
# Install system dependencies for OpenCV and Render
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ===== OPTION 3 PART 2: PORT Handling =====
# Fix for Render's PORT variable
RUN echo '#!/bin/sh\nstreamlit run streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0' > entrypoint.sh \
    && chmod +x entrypoint.sh

# ===== OPTION 3 PART 3: Memory Optimization =====
# (Optional) Add memory limits for TensorFlow
RUN echo '#!/bin/sh\n' \
    'if [ -f "/proc/meminfo" ]; then\n' \
    '  MEM_AVAIL=$(grep MemAvailable /proc/meminfo | awk "{print \$2}")\n' \
    '  if [ "$MEM_AVAIL" -lt 1000000 ]; then\n' \
    '    echo "WARNING: Low memory detected - reducing image size"\n' \
    '    export TF_FORCE_GPU_ALLOW_GROWTH=true\n' \
    '  fi\n' \
    'fi\n' \
    'exec streamlit run streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0' >> entrypoint.sh

CMD ["./entrypoint.sh"]
