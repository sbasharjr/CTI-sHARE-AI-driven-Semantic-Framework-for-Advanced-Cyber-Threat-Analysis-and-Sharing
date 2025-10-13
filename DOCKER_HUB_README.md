# AI-driven Semantic Framework for Advanced Cyber Threat Analysis

[![Docker Build](https://img.shields.io/docker/automated/USERNAME/threat-analysis-framework)](https://hub.docker.com/r/USERNAME/threat-analysis-framework)
[![Docker Pulls](https://img.shields.io/docker/pulls/USERNAME/threat-analysis-framework)](https://hub.docker.com/r/USERNAME/threat-analysis-framework)

A comprehensive machine learning and deep learning framework for real-time cyber threat detection, analysis, and intelligence sharing.

## Quick Start

Pull and run the latest image:

```bash
docker pull USERNAME/threat-analysis-framework:latest
docker run -p 5000:5000 USERNAME/threat-analysis-framework:latest
```

Access the API at: http://localhost:5000/api/health

## What's Included

- **Machine Learning Models**: Random Forest, SVM, Gradient Boosting
- **Deep Learning Models**: LSTM, CNN, BERT, GPT
- **Semantic Analysis**: Threat categorization and entity extraction
- **Real-time Detection**: Asynchronous threat processing engine
- **RESTful API**: Complete API for threat intelligence sharing
- **Automated Response**: Configurable threat response capabilities

## Usage Examples

### API Server (Default)
```bash
docker run -p 5000:5000 threat-analysis-framework:latest
```

### With Persistent Data
```bash
docker run -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  threat-analysis-framework:latest
```

### Threat Analysis Mode
```bash
docker run threat-analysis-framework:latest python main.py analyze
```

### Train Models
```bash
docker run -v $(pwd)/models:/app/models \
  threat-analysis-framework:latest python main.py train
```

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/threats/submit` - Submit a threat
- `GET /api/threats` - Get threats (paginated)
- `POST /api/threats/analyze` - Analyze a threat
- `GET /api/statistics` - Get statistics
- `GET /api/threats/search` - Search threats

## Configuration

Environment variables:
- `PYTHONUNBUFFERED=1` - Enable unbuffered output
- `LOG_LEVEL` - Set logging level (INFO, DEBUG, etc.)

Volumes:
- `/app/data` - Threat data storage
- `/app/models` - Trained ML/DL models
- `/app/logs` - Application logs
- `/app/config` - Configuration files

## Using Docker Compose

Create a `docker-compose.yml`:

```yaml
services:
  threat-analysis-api:
    image: USERNAME/threat-analysis-framework:latest
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
```

Run with:
```bash
docker compose up -d
```

## Documentation

- [GitHub Repository](https://github.com/sbasharjr/Semantic-Framework-for-Advanced-Cyber-Threat-Analysis-and-Sharing)
- [Full Documentation](https://github.com/sbasharjr/Semantic-Framework-for-Advanced-Cyber-Threat-Analysis-and-Sharing/blob/main/README.md)
- [Docker Deployment Guide](https://github.com/sbasharjr/Semantic-Framework-for-Advanced-Cyber-Threat-Analysis-and-Sharing/blob/main/docs/DOCKER.md)
- [Architecture Overview](https://github.com/sbasharjr/Semantic-Framework-for-Advanced-Cyber-Threat-Analysis-and-Sharing/blob/main/docs/ARCHITECTURE.md)

## Requirements

- Docker Engine 20.10+
- 4GB+ RAM
- 10GB+ disk space

## Tags

- `latest` - Latest stable release
- `1.0.0` - Specific version
- `main` - Development version

## License

MIT License - See [LICENSE](https://github.com/sbasharjr/Semantic-Framework-for-Advanced-Cyber-Threat-Analysis-and-Sharing/blob/main/LICENSE)

## Support

- [GitHub Issues](https://github.com/sbasharjr/Semantic-Framework-for-Advanced-Cyber-Threat-Analysis-and-Sharing/issues)
- [GitHub Discussions](https://github.com/sbasharjr/Semantic-Framework-for-Advanced-Cyber-Threat-Analysis-and-Sharing/discussions)

---

**Note**: Replace `USERNAME` with your Docker Hub username when using these commands.
