# Multi-stage Dockerfile for RizanAI Backend
# Optimized for AWS ECS deployment

# ========================================
# Stage 1: Builder
# ========================================
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ========================================
# Stage 2: Runtime
# ========================================
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH \
    PORT=8000

# Set working directory (will be changed to backend/ after copying files)
WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code (backend and ai-services)
COPY backend/ ./backend/
COPY ai-services/ ./ai-services/
WORKDIR /app/backend

# Create necessary directories in backend
RUN mkdir -p \
    backend/uploads \
    backend/processed \
    backend/batch_queue \
    backend/converted_images \
    backend/logs \
    backend/data/organized_vouchers

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
# Temporarily using 1 worker to debug environment variable issue
# TODO: Restore to --workers 4 after fixing env var propagation
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "120", "--limit-max-requests", "10000"]

