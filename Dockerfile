# Sentinel AI - Production Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY examples/ ./examples/
COPY config.yaml ./
COPY README.md ./

# Install Python dependencies
RUN uv sync --frozen

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash sentinel
RUN chown -R sentinel:sentinel /app
USER sentinel

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Expose port
EXPOSE 8080

# Default command (can be overridden)
CMD ["uv", "run", "python", "-m", "src.main"]
