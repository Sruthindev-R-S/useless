FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install minimal system dependencies for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight Python dependencies and clean caches
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /usr/local/lib/python3.10/site-packages/sympy \
    && find /usr/local/lib/python3.10 -name "*.pyc" -delete \
    && find /usr/local/lib/python3.10 -name "__pycache__" -delete

# Copy trained ONNX model weights and application code
COPY model/best.onnx ./model/
COPY app.py .

# Expose default port
EXPOSE 8000

# Healthcheck to verify the server is healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Start server (supports dynamic PORT variable for Render / GCP Cloud Run / AWS / Railway)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
