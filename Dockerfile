FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first to keep docker image lightweight and fast for deployment
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Copy and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy trained model weights and application code
COPY model/ ./model/
COPY app.py .
COPY test/ ./test/

# Expose the default port
EXPOSE 8000

# Healthcheck to verify the server is healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Start server (supports dynamic PORT variable for Render / GCP Cloud Run / AWS / Railway)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
