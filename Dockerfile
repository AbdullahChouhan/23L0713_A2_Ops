# Task 6.2 & 6.4: Optimized Production Dockerfile for VisionGuard (23L0713_A2_Ops)
FROM python:3.11-slim

# Prevent Python from writing .pyc files, buffer stdout/stderr, and cap TensorFlow CPU memory footprint for 512MB cloud tiers (Render)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    OMP_NUM_THREADS=1 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    PORT=5000

WORKDIR /app

# Create a non-root user and group before copying application files, and grant /app ownership
RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --create-home --home-dir /home/appuser appuser && \
    chown -R appuser:appgroup /app

# Install dependencies BEFORE copying source code to maximize Docker layer caching
COPY requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code and champion model (no secrets or .env copied due to .dockerignore)
COPY --chown=appuser:appgroup app/ ./app/
COPY --chown=appuser:appgroup models/champion_model.keras ./models/champion_model.keras

# Switch to non-root user
USER appuser

EXPOSE 5000

# Container-level health check querying Flask's /health endpoint
HEALTHCHECK --interval=20s --timeout=10s --start-period=25s --retries=3 \
    CMD python -c "import os, urllib.request, sys; p = os.getenv('PORT', '5000'); r = urllib.request.urlopen(f'http://127.0.0.1:{p}/health', timeout=5); sys.exit(0 if r.getcode() == 200 else 1)"

# Run production WSGI server (gunicorn) with debug mode off, respecting dynamic $PORT on Render/Cloud
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 2 --timeout 120 app.main:app"]
