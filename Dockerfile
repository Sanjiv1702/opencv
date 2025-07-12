FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use a shell script to handle the PORT variable
RUN echo '#!/bin/sh\nstreamlit run streamlit_app.py --server.port ${PORT:-8501} --server.address=0.0.0.0' > entrypoint.sh \
    && chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
