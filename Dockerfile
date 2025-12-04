# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure service account key file is available (for Cloud Run signed URLs)
# The key file should be in the build context directory
# Note: In production, consider using Secret Manager instead of embedding the key
RUN if [ -f voucher-storage-key.json ]; then \
      echo "✅ Service account key file found"; \
      chmod 600 voucher-storage-key.json; \
    else \
      echo "⚠️  Warning: voucher-storage-key.json not found - signed URLs will not work"; \
    fi

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]