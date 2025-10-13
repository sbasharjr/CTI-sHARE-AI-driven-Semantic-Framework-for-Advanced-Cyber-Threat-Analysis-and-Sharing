# Docker Deployment Guide

This guide provides detailed instructions for deploying the AI-driven Semantic Framework for Advanced Cyber Threat Analysis and Sharing using Docker.

## Prerequisites

- Docker Engine 20.10+ or Docker Desktop
- Docker Compose (included with Docker Desktop)
- 4GB+ RAM available for the container
- 10GB+ disk space for images and data

## Quick Start

### Using Docker Compose (Recommended)

1. **Start the application:**
```bash
docker compose up -d
```

2. **View logs:**
```bash
docker compose logs -f threat-analysis-api
```

3. **Stop the application:**
```bash
docker compose down
```

### Using Docker CLI

1. **Build the image:**
```bash
docker build -t threat-analysis-framework:latest .
```

2. **Run the API server:**
```bash
docker run -d \
  --name threat-analysis-api \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  threat-analysis-framework:latest
```

3. **View logs:**
```bash
docker logs -f threat-analysis-api
```

4. **Stop and remove the container:**
```bash
docker stop threat-analysis-api
docker rm threat-analysis-api
```

## Running Different Modes

The framework supports multiple operation modes:

### API Server (Default)
```bash
docker run -p 5000:5000 threat-analysis-framework:latest
```

### Threat Analysis Mode
```bash
docker run threat-analysis-framework:latest python main.py analyze
```

### Real-time Detection Mode
```bash
docker run threat-analysis-framework:latest python main.py realtime
```

### Model Training Mode
```bash
docker run -v $(pwd)/models:/app/models \
  threat-analysis-framework:latest python main.py train
```

## Configuration

### Environment Variables

You can pass environment variables to customize the behavior:

```bash
docker run -p 5000:5000 \
  -e PYTHONUNBUFFERED=1 \
  -e LOG_LEVEL=INFO \
  threat-analysis-framework:latest
```

### Persistent Data

Mount volumes to persist data across container restarts:

- `/app/data` - Threat data storage
- `/app/models` - Trained ML/DL models
- `/app/logs` - Application logs
- `/app/config` - Configuration files

```bash
docker run -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/config:/app/config \
  threat-analysis-framework:latest
```

## Accessing the API

Once the container is running, the API is available at:

- Health Check: `http://localhost:5000/api/health`
- API Documentation: See README.md for all endpoints

### Test the API

```bash
# Health check
curl http://localhost:5000/api/health

# Submit a threat
curl -X POST http://localhost:5000/api/threats/submit \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Suspicious malware detected in network traffic",
    "source": "network-ids",
    "metadata": {}
  }'

# Get threats
curl http://localhost:5000/api/threats?limit=10

# Get statistics
curl http://localhost:5000/api/statistics
```

## Docker Compose Configuration

The `docker-compose.yml` file provides a complete setup with:

- API server on port 5000
- Volume mounts for data persistence
- Automatic restart policy
- Optional Redis service for caching

To enable Redis (uncomment in docker-compose.yml):
```bash
docker compose up -d
```

## Building for Production

### Multi-stage Build (Optional)

For smaller production images, you can create a multi-stage Dockerfile:

```dockerfile
# Build stage
FROM python:3.9-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "main.py", "api"]
```

### Image Optimization

To reduce image size:

```bash
# Build with no cache
docker build --no-cache -t threat-analysis-framework:latest .

# Remove dangling images
docker image prune -f
```

## Troubleshooting

### Container won't start

1. Check logs:
```bash
docker logs threat-analysis-api
```

2. Verify port availability:
```bash
lsof -i :5000
```

3. Check container status:
```bash
docker ps -a
```

### Out of memory

Increase Docker memory limit in Docker Desktop settings or:

```bash
docker run -m 4g -p 5000:5000 threat-analysis-framework:latest
```

### Permission issues with volumes

Ensure the host directories exist and have proper permissions:

```bash
mkdir -p data models logs
chmod -R 755 data models logs
```

## Publishing to Docker Hub

To publish the image to Docker Hub:

1. **Tag the image:**
```bash
docker tag threat-analysis-framework:latest yourusername/threat-analysis-framework:latest
docker tag threat-analysis-framework:latest yourusername/threat-analysis-framework:1.0.0
```

2. **Login to Docker Hub:**
```bash
docker login
```

3. **Push the image:**
```bash
docker push yourusername/threat-analysis-framework:latest
docker push yourusername/threat-analysis-framework:1.0.0
```

4. **Pull and run from Docker Hub:**
```bash
docker pull yourusername/threat-analysis-framework:latest
docker run -p 5000:5000 yourusername/threat-analysis-framework:latest
```

## Publishing to GitHub Container Registry

To publish to GitHub Container Registry (ghcr.io):

1. **Create a personal access token** with `write:packages` scope

2. **Login to GHCR:**
```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

3. **Tag the image:**
```bash
docker tag threat-analysis-framework:latest ghcr.io/USERNAME/threat-analysis-framework:latest
```

4. **Push the image:**
```bash
docker push ghcr.io/USERNAME/threat-analysis-framework:latest
```

## CI/CD Integration

### GitHub Actions Example

Create `.github/workflows/docker-publish.yml`:

```yaml
name: Docker Build and Publish

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: docker build -t threat-analysis-framework:latest .
    
    - name: Test Docker image
      run: |
        docker run --rm threat-analysis-framework:latest python -c "import sys; sys.exit(0)"
    
    - name: Login to Docker Hub
      if: github.event_name != 'pull_request'
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Push to Docker Hub
      if: github.event_name != 'pull_request'
      run: |
        docker tag threat-analysis-framework:latest ${{ secrets.DOCKER_USERNAME }}/threat-analysis-framework:latest
        docker push ${{ secrets.DOCKER_USERNAME }}/threat-analysis-framework:latest
```

## Security Considerations

1. **Don't include secrets in the image:**
   - Use environment variables or Docker secrets
   - Add `.env` files to `.dockerignore`

2. **Run as non-root user (optional):**
   Add to Dockerfile:
   ```dockerfile
   RUN useradd -m -u 1000 appuser
   USER appuser
   ```

3. **Scan images for vulnerabilities:**
   ```bash
   docker scan threat-analysis-framework:latest
   ```

4. **Keep base images updated:**
   ```bash
   docker pull python:3.9-slim
   docker build --no-cache -t threat-analysis-framework:latest .
   ```

## Support

For issues or questions about Docker deployment:
- Check the [main README](../README.md)
- Review Docker logs
- Open an issue on GitHub
