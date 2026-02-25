# Use Python 3.10 slim image
FROM python:3.10-slim

# OCI image metadata
LABEL org.opencontainers.image.title="Aircraft Engine RUL Prediction API" \
    org.opencontainers.image.description="Predictive maintenance API for turbofan engines using NASA C-MAPSS" \
    org.opencontainers.image.version="2.0.1" \
    org.opencontainers.image.source="https://github.com/jitesh523/Aircraft-Engine-Degradation-RUL-Prediction" \
    org.opencontainers.image.licenses="MIT"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models and results
RUN mkdir -p models/saved results plots logs

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --no-create-home appuser && \
    chown -R appuser:appuser /app

# Expose API port
EXPOSE 8000

# Health check using curl (lightweight, no extra Python deps)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER appuser

# Run API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
