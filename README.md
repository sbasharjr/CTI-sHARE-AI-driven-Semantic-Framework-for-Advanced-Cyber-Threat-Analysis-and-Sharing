# CTI-sHARE-AI-driven Semantic Framework for Advanced Cyber Threat Analysis and Sharing

A comprehensive machine learning and deep learning framework for real-time cyber threat detection, analysis, and intelligence sharing.

## Overview

This framework leverages advanced artificial intelligence techniques including machine learning, deep learning, and semantic analysis to detect, categorize, and respond to cyber threats in real-time. It provides a complete solution for organizations to enhance their cybersecurity posture through automated threat intelligence.

## Key Features

### 1. **Machine Learning Models**
- Random Forest Classifier
- Support Vector Machines (SVM)
- Gradient Boosting
- Naive Bayes
- Cross-validation and hyperparameter tuning support

### 2. **Deep Learning Models**
- Bidirectional LSTM for sequential threat analysis
- CNN for pattern recognition in threat data
- **Transformer models (BERT, GPT)** for advanced NLP
- **Graph Neural Networks** for relationship analysis
- Custom architectures with dropout and regularization
- Early stopping and model checkpointing

### 3. **Semantic Analysis**
- Threat categorization (malware, phishing, DDoS, data breach, APT, etc.)
- Entity extraction (IPs, domains, hashes, emails, CVEs)
- Threat similarity analysis and clustering
- Severity assessment (CRITICAL, HIGH, MEDIUM, LOW, INFORMATIONAL)
- Trend analysis and reporting

### 4. **Real-time Threat Detection**
- Asynchronous threat processing
- Multi-threaded detection engine
- Alert generation and monitoring
- Configurable detection thresholds
- Event callbacks and notifications
- **Automated threat response** with configurable rules
- **Web-based dashboard** for visualization
- **Multi-language support** (8+ languages)

### 5. **Threat Intelligence Sharing**
- RESTful API for threat submission and retrieval
- Search and filtering capabilities
- Real-time threat analysis endpoints
- Statistics and reporting
- CORS-enabled for web integration
- **Federated learning** for privacy-preserving collaboration
- **Blockchain-based verification** for immutable records
- **SIEM integration** (Splunk, QRadar)

## Architecture

```
Dataset-Semantic-Framework/
├── src/
│   ├── preprocessing/          # Data preprocessing and feature extraction
│   ├── models/                 # ML and DL models
│   ├── semantic_analysis/      # Semantic threat analysis
│   ├── realtime/              # Real-time detection engine
│   ├── threat_sharing/        # API for threat sharing
│   └── utils/                 # Utility functions
├── config/                     # Configuration files
├── examples/                   # Usage examples
├── tests/                      # Unit and integration tests
├── data/                       # Data storage
│   ├── raw/                   # Raw threat data
│   └── processed/             # Processed data
└── docs/                       # Documentation
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone [https://github.com/sbasharjr/CTI-sHARE-AI-driven-Semantic-Framework-for-Advanced-Cyber-Threat-Analysis-and-Sharing.git]
cd CTI-sHARE-AI-driven-Semantic-Framework-for-Advanced-Cyber-Threat-Analysis-and-Sharing
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data (required for text processing):
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### Docker Deployment

The easiest way to deploy the framework is using Docker:

#### Quick Start with Script
```bash
# Make the script executable (first time only)
chmod +x docker-start.sh

# Start the framework
./docker-start.sh up

# View logs
./docker-start.sh logs

# Stop the framework
./docker-start.sh stop
```

#### Manual Docker Compose
```bash
# Start the framework
docker compose up -d

# View logs
docker compose logs -f

# Stop the framework
docker compose down
```

#### Manual Docker Commands
```bash
# Build the image
docker build -t threat-analysis-framework .

# Run the API server
docker run -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  threat-analysis-framework

# Run in different modes
docker run threat-analysis-framework python main.py analyze
docker run threat-analysis-framework python main.py train
```

Access the API at `http://localhost:5000/api/health`

For detailed Docker deployment instructions, see [Docker Deployment Guide](docs/DOCKER.md).

## Quick Start

### Basic Usage

```python
from src.preprocessing.data_preprocessor import ThreatDataPreprocessor
from src.models.ml_models import ThreatDetectionML
from src.semantic_analysis.semantic_analyzer import ThreatSemanticAnalyzer
from src.utils.data_loader import ThreatDataLoader

# Load data
loader = ThreatDataLoader()
data = loader.create_sample_data(num_samples=100)

# Preprocess
preprocessor = ThreatDataPreprocessor()
processed_data = preprocessor.extract_features(data)

# Semantic analysis
analyzer = ThreatSemanticAnalyzer()
categories = analyzer.categorize_threat("Ransomware attack targeting healthcare")
severity = analyzer.assess_threat_severity("Critical zero-day vulnerability")

# Train ML model
ml_model = ThreatDetectionML(model_type='random_forest')
X = preprocessor.create_feature_matrix(processed_data)
y = processed_data['severity'].values
metrics = ml_model.train(X, y)
```

### Real-time Detection

```python
from src.realtime.detector import RealTimeThreatDetector, ThreatMonitor
from src.semantic_analysis.semantic_analyzer import ThreatSemanticAnalyzer

# Initialize
analyzer = ThreatSemanticAnalyzer()
detector = RealTimeThreatDetector(semantic_analyzer=analyzer)
monitor = ThreatMonitor(detector)

# Start detection
detector.start()

# Submit threat data
detector.add_threat_data({
    'text': 'Suspicious activity detected on network',
    'features': [...]
})

# Get statistics
stats = detector.get_statistics()
alerts = monitor.get_alerts()

# Stop detection
detector.stop()
```

### API Server

```python
from src.threat_sharing.api import ThreatSharingAPI
from src.semantic_analysis.semantic_analyzer import ThreatSemanticAnalyzer

# Initialize
analyzer = ThreatSemanticAnalyzer()
api = ThreatSharingAPI(semantic_analyzer=analyzer)

# Run server
api.run(host='0.0.0.0', port=5000)
```

Or run the example script:
```bash
python examples/api_server.py
```

## API Endpoints

### Health Check
```
GET /api/health
```

### Submit Threat
```
POST /api/threats/submit
Body: {
  "text": "Threat description",
  "source": "source_id",
  "metadata": {}
}
```

### Get Threats
```
GET /api/threats?limit=10&offset=0
```

### Analyze Threat
```
POST /api/threats/analyze
Body: {
  "text": "Threat description to analyze"
}
```

### Get Statistics
```
GET /api/statistics
```

### Search Threats
```
GET /api/threats/search?q=keyword&category=malware
```

## Examples

Run the included examples to see the framework in action:

```bash
# Basic usage example
python examples/basic_usage.py

# Real-time detection example
python examples/realtime_detection.py

# API server example
python examples/api_server.py

# Comprehensive integration (demonstrates all advanced features together)
python examples/comprehensive_integration_example.py

# Individual advanced features
python examples/transformer_example.py       # BERT/GPT models
python examples/gnn_example.py              # Graph neural networks
python examples/federated_learning_example.py  # Federated learning
python examples/automated_response_example.py  # Automated response
python examples/blockchain_example.py        # Blockchain verification
python examples/i18n_example.py             # Multi-language support
```

## Configuration

Edit `config/config.yaml` to customize:
- Data directories
- Model parameters
- Detection thresholds
- API settings
- Logging configuration

## Threat Categories

The framework categorizes threats into the following types:
- **Malware**: Ransomware, trojans, viruses, worms, spyware
- **Phishing**: Credential harvesting, social engineering
- **DDoS**: Distributed denial of service attacks
- **Data Breach**: Unauthorized access, data exfiltration
- **APT**: Advanced persistent threats, nation-state attacks
- **Vulnerability Exploit**: Zero-days, CVEs, code execution
- **Insider Threat**: Malicious insiders, privilege abuse
- **Supply Chain**: Third-party compromises, software attacks

## Severity Levels

Threats are assessed with the following severity levels:
1. **INFORMATIONAL**: Low-impact events
2. **LOW**: Minor threats with limited impact
3. **MEDIUM**: Notable threats requiring attention
4. **HIGH**: Serious threats requiring immediate action
5. **CRITICAL**: Severe threats with major impact

## Machine Learning Models

### Random Forest
- Ensemble learning with 100 decision trees
- Robust to overfitting
- Feature importance analysis
- Fast training and prediction

### Deep Learning (LSTM)
- Bidirectional LSTM layers
- Word embeddings for text representation
- Dropout regularization
- Early stopping for optimal performance

### Deep Learning (CNN)
- 1D Convolutional layers
- Max pooling for feature reduction
- Global pooling for sequence aggregation
- Efficient pattern recognition

## Testing

Run tests to verify the framework:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## Performance Metrics

The framework provides comprehensive metrics:
- Accuracy, precision, recall, F1-score
- Confusion matrices
- ROC curves and AUC scores
- Training and validation curves
- Real-time detection statistics

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{threat_analysis_framework,
  title={CTI-sHARE-AI-driven Semantic Framework for Advanced Cyber Threat Analysis and Sharing},
  author={Suleiman Sani Bashar},
  year={2025},
  url={https://github.com/sbasharjr/Dataset-Semantic-Framework-for-Advanced-Cyber-Threat-Analysis-and-Sharing}
}
```

## Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation

## Acknowledgments

This framework leverages:
- scikit-learn for machine learning
- TensorFlow/Keras for deep learning
- NLTK for natural language processing
- Flask for API development
- pandas for data manipulation
- PyTorch for transformer models and GNNs
- Transformers library for BERT/GPT models

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [Advanced Features Guide](docs/ADVANCED_FEATURES.md)
- [Docker Deployment Guide](docs/DOCKER.md)
- API documentation in `docs/` directory
- Example scripts in `examples/` directory

## Roadmap

Future enhancements:
- [x] Transformer-based models (BERT, GPT)
- [x] Graph neural networks for relationship analysis
- [x] Federated learning for privacy-preserving threat sharing
- [x] Integration with SIEM systems
- [x] Automated threat response capabilities
- [x] Web-based dashboard for visualization
- [x] Multi-language support
- [x] Blockchain for threat intelligence verification

---

**Version**: 1.0.0  
**Last Updated**: Sept, 2025
**Status**: Production Ready
