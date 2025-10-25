# Production WSGI Server Deployment Guide

## 🎯 CTI-sHARE Dashboard Production Deployment

This guide covers deploying the CTI-sHARE threat intelligence dashboard using production-grade WSGI servers.

## 📋 Prerequisites

### Dependencies Installed
- ✅ **Gunicorn**: Linux/Unix production WSGI server
- ✅ **Waitress**: Windows-compatible production WSGI server  
- ✅ **psutil**: System monitoring for live data APIs

### Installation Command
```bash
pip install gunicorn waitress psutil
```

## 🚀 Deployment Options

### 1. Windows Production Server (Waitress)

**Quick Start:**
```bash
python run_production_server.py
```

**Using Batch File:**
```bash
deploy_production_windows.bat
```

**Features:**
- Windows-native WSGI server
- Multi-threaded request handling
- Built-in load balancing
- Production-ready configuration

### 2. Linux/Unix Production Server (Gunicorn)

**Quick Start:**
```bash
gunicorn --config gunicorn_config.py wsgi:application
```

**Using Shell Script:**
```bash
./deploy_production_linux.sh
```

**Alternative Simple Command:**
```bash
gunicorn --bind 0.0.0.0:5001 --workers 4 --timeout 30 wsgi:application
```

**Features:**
- Multi-process worker model
- Graceful worker restarts
- Advanced configuration options
- High performance under load

### 3. Docker Production Deployment

**Build and Run:**
```bash
# Build production image
docker build -f Dockerfile.production -t cti-share-dashboard .

# Run container
docker run -p 5001:5001 cti-share-dashboard

# Or use docker-compose
docker-compose -f docker-compose.production.yml up -d
```

**Features:**
- Containerized deployment
- Nginx reverse proxy (optional)
- Redis caching (optional)
- Health checks and auto-restart

## ⚙️ Configuration Files

### WSGI Application (`wsgi.py`)
```python
# Production WSGI entry point
from src.dashboard.dashboard import ThreatDashboard

def create_app():
    dashboard = ThreatDashboard()
    # Add sample threat data
    # Configure production settings
    return dashboard.app

application = create_app()
```

### Gunicorn Config (`gunicorn_config.py`)
```python
# Production server configuration
bind = "0.0.0.0:5001"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
timeout = 30
preload_app = True
```

### Waitress Server (`run_production_server.py`)
```python
# Windows-compatible production server
from waitress import serve
serve(application, 
      host='0.0.0.0', 
      port=5001, 
      threads=6)
```

## 🔧 Production Features

### Performance Optimizations
- **Multi-worker/Multi-thread**: Parallel request handling
- **Connection Pooling**: Efficient resource management
- **Request Timeouts**: Prevent hanging connections
- **Graceful Shutdowns**: Clean server restarts

### Security Features
- **Process Isolation**: Separate worker processes
- **Resource Limits**: Memory and connection limits
- **Error Handling**: Production error logging
- **Environment Separation**: Production vs development configs

### Monitoring & Health Checks
- **Health Endpoints**: `/api/dashboard/health`
- **Process Monitoring**: Worker status tracking
- **Request Logging**: Access and error logs
- **Performance Metrics**: Response time monitoring

## 🌐 Reverse Proxy Setup (Optional)

### Nginx Configuration
```nginx
upstream cti_share_backend {
    server localhost:5001;
}

server {
    listen 80;
    location / {
        proxy_pass http://cti_share_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**Benefits:**
- SSL termination
- Load balancing
- Static file serving
- Rate limiting
- Security headers

## 📊 Scaling Options

### Horizontal Scaling
```bash
# Multiple Gunicorn instances
gunicorn --bind 127.0.0.1:5001 wsgi:application &
gunicorn --bind 127.0.0.1:5002 wsgi:application &
gunicorn --bind 127.0.0.1:5003 wsgi:application &
```

### Vertical Scaling
```python
# Increase workers based on CPU cores
workers = multiprocessing.cpu_count() * 2 + 1
```

### Container Scaling
```yaml
# Docker compose scaling
version: '3.8'
services:
  cti-share-dashboard:
    deploy:
      replicas: 3
```

## 🔍 Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Check Python path
export PYTHONPATH=/app/src

# Install missing dependencies
pip install -r requirements.txt
```

**Port Conflicts:**
```bash
# Check port usage
netstat -tulpn | grep 5001

# Change port in config
bind = "0.0.0.0:8080"
```

**Permission Issues:**
```bash
# Run as non-root user
useradd --create-home app
su - app
```

### Health Check Commands
```bash
# Test health endpoint
curl http://localhost:5001/api/dashboard/health

# Test main dashboard
curl http://localhost:5001/

# Test live data APIs
curl http://localhost:5001/api/dashboard/live/system-performance
```

## 📈 Production Monitoring

### Key Metrics to Monitor
- **Response Time**: API endpoint latency
- **Request Rate**: Requests per second
- **Error Rate**: 4xx/5xx response percentage
- **Memory Usage**: Worker memory consumption
- **CPU Usage**: Server CPU utilization

### Logging Configuration
```python
# Production logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/dashboard.log'),
        logging.StreamHandler()
    ]
)
```

## 🎉 Deployment Summary

The CTI-sHARE dashboard is now configured with professional production WSGI servers:

### ✅ **Completed Setup**
- **WSGI Application**: Production-ready Flask app
- **Waitress Server**: Windows-compatible production server
- **Gunicorn Server**: Linux/Unix high-performance server
- **Docker Setup**: Containerized deployment option
- **Nginx Config**: Reverse proxy configuration
- **Deployment Scripts**: Automated setup for Windows/Linux

### 🚀 **Ready for Production**
- **High Performance**: Multi-worker/multi-thread handling
- **Scalability**: Horizontal and vertical scaling options
- **Security**: Process isolation and resource limits
- **Monitoring**: Health checks and logging
- **Flexibility**: Multiple deployment options

### 📋 **Quick Commands**
```bash
# Windows Production
python run_production_server.py

# Linux Production  
gunicorn --config gunicorn_config.py wsgi:application

# Docker Production
docker-compose -f docker-compose.production.yml up -d
```

Your CTI-sHARE threat intelligence dashboard is now enterprise-ready for production deployment! 🛡️