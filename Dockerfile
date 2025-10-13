# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies (with fallback for SSL issues)
RUN pip install --no-cache-dir -r requirements.txt || \
    pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')" || true

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models logs

# Expose port for API server
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Default command - run API server
CMD ["python", "main.py", "api", "--host", "0.0.0.0", "--port", "5000"]
