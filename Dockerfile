# ==============================================================================
# Production Dockerfile for AI Data Analyst Agent
# Multi-stage / minimal footprint Python 3.10 deployment with non-root security
# ==============================================================================

FROM python:3.10-slim

# Set environment variables for production execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    FLASK_DEBUG=False \
    PORT=5000 \
    HOST=0.0.0.0

# Set working directory
WORKDIR /app

# Install minimal OS dependencies for compilation and health check curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure storage directories exist with secure permissions
RUN mkdir -p data/uploads data/processed data/reports data/temp logs && \
    chmod -R 755 data logs

# Run as dedicated unprivileged user for container isolation
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose HTTP port
EXPOSE 5000

# Container health monitoring
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Production WSGI server command (Waitress)
CMD ["sh", "-c", "waitress-serve --host=0.0.0.0 --port=${PORT:-5000} wsgi:app"]
