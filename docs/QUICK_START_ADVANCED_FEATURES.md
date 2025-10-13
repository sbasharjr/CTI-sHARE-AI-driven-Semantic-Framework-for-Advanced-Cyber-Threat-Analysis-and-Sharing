# Quick Start Guide: Advanced Features

## Overview

This guide helps you get started with the advanced features of the AI-driven Semantic Framework for Advanced Cyber Threat Analysis and Sharing.

## ✅ All Features Implemented

The framework includes **8 advanced features**, all fully implemented and ready to use:

1. 🤖 **Transformer-based models (BERT, GPT)**
2. 🔗 **Graph Neural Networks**
3. 🔐 **Federated Learning**
4. 🔔 **SIEM Integration**
5. ⚡ **Automated Response**
6. 📊 **Web Dashboard**
7. 🌍 **Multi-language Support**
8. ⛓️ **Blockchain Verification**

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/sbasharjr/Semantic-Framework-for-Advanced-Cyber-Threat-Analysis-and-Sharing.git
cd Semantic-Framework-for-Advanced-Cyber-Threat-Analysis-and-Sharing

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### 2. Validate Installation

```bash
# Run the validation script
python scripts/validate_features.py
```

Expected output:
```
✓ Transformer-based models (BERT, GPT)
✓ Graph Neural Networks
✓ Federated Learning
✓ SIEM Integration
✓ Automated Threat Response
✓ Web-based Dashboard
✓ Multi-language Support
✓ Blockchain Verification

✓ ALL VALIDATIONS PASSED!
```

### 3. Try the Examples

#### Comprehensive Integration (Recommended First Step)
```bash
python examples/comprehensive_integration_example.py
```

This demonstrates all features working together in a complete workflow.

#### Individual Features

**Transformer Models:**
```bash
python examples/transformer_example.py
```

**Graph Neural Networks:**
```bash
python examples/gnn_example.py
```

**Federated Learning:**
```bash
python examples/federated_learning_example.py
```

**Automated Response:**
```bash
python examples/automated_response_example.py
```

**Blockchain Verification:**
```bash
python examples/blockchain_example.py
```

**Multi-language Support:**
```bash
python examples/i18n_example.py
```

### 4. Run Tests

```bash
# Run all tests
python -m unittest discover tests/

# Run specific integration tests
python -m unittest tests.test_advanced_features_integration
```

## Feature-Specific Quick Guides

### 🤖 Transformer Models (BERT, GPT)

**Use Case:** Advanced threat text analysis and classification

```python
from src.models.transformer_models import ThreatDetectionBERT

# Initialize BERT model
bert = ThreatDetectionBERT(num_classes=5)

# Train on threat data
bert.train(train_texts, train_labels, epochs=3)

# Predict threat severity
predictions = bert.predict(["Ransomware attack detected"])
```

📖 Full documentation: `docs/ADVANCED_FEATURES.md#transformer-models`

---

### 🔗 Graph Neural Networks

**Use Case:** Analyzing threat relationships and attack patterns

```python
from src.models.gnn_models import ThreatGraphAnalyzer

# Initialize GNN
gnn = ThreatGraphAnalyzer(input_dim=128)

# Define threat relationships
relationships = [(0, 1), (1, 2), (2, 3)]

# Find related threats
related = gnn.find_related_threats(features, relationships, query_idx=0)
```

📖 Full documentation: `docs/ADVANCED_FEATURES.md#graph-neural-networks`

---

### 🔐 Federated Learning

**Use Case:** Privacy-preserving collaborative threat intelligence

```python
from src.models.federated_learning import FederatedLearningOrchestrator
import torch.nn as nn

# Define model architecture
class ThreatModel(nn.Module):
    # ... your model

# Initialize orchestrator
orchestrator = FederatedLearningOrchestrator(
    model_architecture=ThreatModel(),
    num_clients=5
)

# Train across organizations
history = orchestrator.train_federated(client_data, num_rounds=10)
```

📖 Full documentation: `docs/ADVANCED_FEATURES.md#federated-learning`

---

### 🔔 SIEM Integration

**Use Case:** Centralized security monitoring

```python
from src.integrations.siem_integration import SIEMIntegration, SplunkConnector

# Initialize SIEM
siem = SIEMIntegration()

# Add Splunk connector
splunk = SplunkConnector(host='splunk.example.com', port=8089, api_key='key')
siem.add_siem('splunk', splunk)

# Send alert
alert = siem.format_threat_alert(threat_data)
siem.send_alert('splunk', alert)
```

📖 Full documentation: `docs/ADVANCED_FEATURES.md#siem-integration`

---

### ⚡ Automated Response

**Use Case:** Rapid threat response

```python
from src.response.automated_response import AutomatedResponseSystem

# Initialize system
response_system = AutomatedResponseSystem()
response_system.create_default_rules()

# Process threat
response = response_system.process_threat(threat_data)
print(f"Rules matched: {len(response['rules_matched'])}")
```

📖 Full documentation: `docs/ADVANCED_FEATURES.md#automated-response`

---

### 📊 Web Dashboard

**Use Case:** Real-time threat visualization

```python
from src.dashboard.dashboard import ThreatDashboard

# Initialize dashboard
dashboard = ThreatDashboard()

# Add threats
dashboard.add_threat(threat_data)

# Run server
dashboard.run(host='0.0.0.0', port=5001)
```

Access at: `http://localhost:5001`

📖 Full documentation: `docs/ADVANCED_FEATURES.md#web-dashboard`

---

### 🌍 Multi-language Support

**Use Case:** Global threat intelligence sharing

```python
from src.i18n.translation import TranslationManager

# Initialize translator
translator = TranslationManager()

# Set language
translator.set_language('es')

# Translate threat data
translated = translator.translate_threat_data(threat_data, language='es')
```

Supported: EN, ES, FR, DE, ZH, JA, AR, RU

📖 Full documentation: `docs/ADVANCED_FEATURES.md#multi-language`

---

### ⛓️ Blockchain Verification

**Use Case:** Immutable threat intelligence records

```python
from src.blockchain.verification import BlockchainVerificationService

# Initialize blockchain
blockchain = BlockchainVerificationService(difficulty=2)

# Submit threat
threat_hash = blockchain.submit_threat(threat_data)

# Commit to blockchain
block = blockchain.commit_threats()

# Verify authenticity
is_authentic = blockchain.verify_threat_authenticity(threat_id, threat_hash)
```

📖 Full documentation: `docs/ADVANCED_FEATURES.md#blockchain`

---

## Docker Deployment

### Quick Start with Docker

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Services Included

- API Server (Port 5000)
- Dashboard (Port 5001)
- All advanced features enabled

📖 Full documentation: `docs/DOCKER.md`

---

## Documentation Resources

| Document | Description |
|----------|-------------|
| `README.md` | Main overview and getting started |
| `docs/ADVANCED_FEATURES.md` | Detailed feature documentation |
| `docs/FEATURES_IMPLEMENTATION_STATUS.md` | Implementation status and metrics |
| `docs/ARCHITECTURE.md` | System architecture |
| `docs/DOCKER.md` | Docker deployment guide |

---

## Example Files

| Example | Description |
|---------|-------------|
| `comprehensive_integration_example.py` | All features together |
| `transformer_example.py` | BERT/GPT models |
| `gnn_example.py` | Graph neural networks |
| `federated_learning_example.py` | Privacy-preserving learning |
| `automated_response_example.py` | Automated threat response |
| `blockchain_example.py` | Blockchain verification |
| `i18n_example.py` | Multi-language support |
| `basic_usage.py` | Basic framework usage |
| `realtime_detection.py` | Real-time threat detection |
| `api_server.py` | API server |

---

## Troubleshooting

### Common Issues

**Import errors with transformers/torch:**
```bash
pip install torch transformers
```

**CUDA not available:**
- CPU mode is automatic fallback
- For GPU support, install CUDA toolkit

**Dependencies missing:**
```bash
pip install -r requirements.txt --upgrade
```

**Port already in use:**
```python
# Use different port
dashboard.run(port=5002)
```

### Get Help

- 📖 Check documentation in `docs/`
- 💻 Run examples in `examples/`
- 🧪 Review tests in `tests/`
- 🐛 Open issue on GitHub

---

## Next Steps

1. ✅ Complete installation and validation
2. ✅ Run comprehensive integration example
3. ✅ Explore individual feature examples
4. ✅ Review documentation
5. ✅ Integrate with your security infrastructure
6. ✅ Customize for your use case

---

## Contributing

Contributions welcome! Please see `README.md` for guidelines.

---

## License

MIT License - See LICENSE file for details.

---

**Version:** 1.0.0  
**Status:** ✅ Production Ready  
**All Features:** ✅ Implemented and Tested
