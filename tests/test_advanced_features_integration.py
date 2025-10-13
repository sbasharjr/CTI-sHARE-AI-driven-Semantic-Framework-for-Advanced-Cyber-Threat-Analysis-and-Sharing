"""
Integration tests for advanced features

Tests the integration of all advanced features:
1. Transformer-based models (BERT, GPT)
2. Graph neural networks
3. Federated learning
4. SIEM integration
5. Automated response
6. Blockchain verification
7. Multi-language support
8. Dashboard
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestTransformerModels(unittest.TestCase):
    """Test transformer-based models"""
    
    def test_bert_initialization(self):
        """Test BERT model initialization"""
        from src.models.transformer_models import ThreatDetectionBERT
        
        model = ThreatDetectionBERT(num_classes=5)
        self.assertIsNotNone(model)
        self.assertEqual(model.num_classes, 5)
        self.assertFalse(model.is_trained)
    
    def test_gpt_initialization(self):
        """Test GPT model initialization"""
        from src.models.transformer_models import ThreatDetectionGPT
        
        model = ThreatDetectionGPT()
        self.assertIsNotNone(model)


class TestGraphNeuralNetworks(unittest.TestCase):
    """Test graph neural network features"""
    
    def test_gnn_initialization(self):
        """Test GNN analyzer initialization"""
        from src.models.gnn_models import ThreatGraphAnalyzer
        
        analyzer = ThreatGraphAnalyzer(input_dim=128, hidden_dim=64, output_dim=32)
        self.assertIsNotNone(analyzer)
        self.assertEqual(analyzer.input_dim, 128)
    
    def test_threat_graph_building(self):
        """Test building threat graph"""
        from src.models.gnn_models import ThreatGraphAnalyzer
        
        analyzer = ThreatGraphAnalyzer(input_dim=10, hidden_dim=8, output_dim=4)
        threat_features = np.random.randn(5, 10)
        relationships = [(0, 1), (1, 2), (2, 3)]
        
        x, adj = analyzer.build_threat_graph(threat_features, relationships)
        self.assertEqual(x.shape[0], 5)
        self.assertEqual(adj.shape, (5, 5))


class TestFederatedLearning(unittest.TestCase):
    """Test federated learning features"""
    
    def test_orchestrator_initialization(self):
        """Test federated learning orchestrator initialization"""
        from src.models.federated_learning import FederatedLearningOrchestrator
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(10, 2)
            
            def forward(self, x):
                return self.fc(x)
        
        orchestrator = FederatedLearningOrchestrator(
            model_architecture=SimpleModel(),
            num_clients=3
        )
        self.assertIsNotNone(orchestrator)
        self.assertEqual(len(orchestrator.clients), 3)


class TestSIEMIntegration(unittest.TestCase):
    """Test SIEM integration features"""
    
    def test_siem_integration_initialization(self):
        """Test SIEM integration initialization"""
        from src.integrations.siem_integration import SIEMIntegration
        
        siem = SIEMIntegration()
        self.assertIsNotNone(siem)
        self.assertEqual(len(siem.connectors), 0)
    
    def test_format_threat_alert(self):
        """Test threat alert formatting"""
        from src.integrations.siem_integration import SIEMIntegration
        from datetime import datetime
        
        siem = SIEMIntegration()
        threat_data = {
            'id': 'TEST-001',
            'timestamp': datetime.now().isoformat(),
            'severity': 'CRITICAL',
            'category': 'malware',
            'description': 'Test threat'
        }
        
        alert = siem.format_threat_alert(threat_data)
        self.assertIn('title', alert)
        self.assertIn('severity', alert)
        self.assertIn('timestamp', alert)


class TestAutomatedResponse(unittest.TestCase):
    """Test automated response features"""
    
    def test_response_system_initialization(self):
        """Test automated response system initialization"""
        from src.response.automated_response import AutomatedResponseSystem
        
        system = AutomatedResponseSystem()
        self.assertIsNotNone(system)
        self.assertEqual(len(system.rules), 0)
    
    def test_default_rules_creation(self):
        """Test creating default response rules"""
        from src.response.automated_response import AutomatedResponseSystem
        
        system = AutomatedResponseSystem()
        system.create_default_rules()
        self.assertGreater(len(system.rules), 0)
    
    def test_process_threat(self):
        """Test processing threat with response system"""
        from src.response.automated_response import AutomatedResponseSystem
        from datetime import datetime
        
        system = AutomatedResponseSystem()
        system.create_default_rules()
        
        threat_data = {
            'id': 'TEST-001',
            'timestamp': datetime.now().isoformat(),
            'severity': 'CRITICAL',
            'category': 'malware',
            'confidence': 0.95
        }
        
        response = system.process_threat(threat_data)
        self.assertIn('threat_id', response)
        self.assertIn('rules_matched', response)
        self.assertIn('actions_executed', response)


class TestBlockchainVerification(unittest.TestCase):
    """Test blockchain verification features"""
    
    def test_blockchain_initialization(self):
        """Test blockchain service initialization"""
        from src.blockchain.verification import BlockchainVerificationService
        
        blockchain = BlockchainVerificationService(difficulty=1)
        self.assertIsNotNone(blockchain)
        self.assertEqual(len(blockchain.blockchain.chain), 1)  # Genesis block
    
    def test_submit_threat(self):
        """Test submitting threat to blockchain"""
        from src.blockchain.verification import BlockchainVerificationService
        from datetime import datetime
        
        blockchain = BlockchainVerificationService(difficulty=1)
        threat_data = {
            'id': 'TEST-001',
            'timestamp': datetime.now().isoformat(),
            'severity': 'HIGH',
            'category': 'phishing'
        }
        
        threat_hash = blockchain.submit_threat(threat_data)
        self.assertIsNotNone(threat_hash)
        self.assertIsInstance(threat_hash, str)
    
    def test_blockchain_verification(self):
        """Test blockchain integrity verification"""
        from src.blockchain.verification import BlockchainVerificationService
        
        blockchain = BlockchainVerificationService(difficulty=1)
        is_valid = blockchain.blockchain.verify_chain()
        self.assertTrue(is_valid)


class TestMultiLanguageSupport(unittest.TestCase):
    """Test multi-language support features"""
    
    def test_translation_manager_initialization(self):
        """Test translation manager initialization"""
        from src.i18n.translation import TranslationManager
        
        tm = TranslationManager(default_language='en')
        self.assertIsNotNone(tm)
        self.assertEqual(tm.current_language, 'en')
    
    def test_supported_languages(self):
        """Test getting supported languages"""
        from src.i18n.translation import TranslationManager
        
        tm = TranslationManager()
        languages = tm.get_supported_languages()
        
        self.assertIn('en', languages)
        self.assertIn('es', languages)
        self.assertIn('fr', languages)
        self.assertGreater(len(languages), 5)
    
    def test_set_language(self):
        """Test setting language"""
        from src.i18n.translation import TranslationManager
        
        tm = TranslationManager()
        success = tm.set_language('es')
        
        self.assertTrue(success)
        self.assertEqual(tm.current_language, 'es')
    
    def test_translation(self):
        """Test getting translation"""
        from src.i18n.translation import TranslationManager
        
        tm = TranslationManager()
        tm.set_language('es')
        
        # Get translation (should return Spanish version or key)
        translation = tm.get_translation('severity')
        self.assertIsNotNone(translation)


class TestDashboard(unittest.TestCase):
    """Test dashboard features"""
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization"""
        from src.dashboard.dashboard import ThreatDashboard
        
        dashboard = ThreatDashboard()
        self.assertIsNotNone(dashboard)
        self.assertIsNotNone(dashboard.app)
    
    def test_add_threat(self):
        """Test adding threat to dashboard"""
        from src.dashboard.dashboard import ThreatDashboard
        from datetime import datetime
        
        dashboard = ThreatDashboard()
        threat_data = {
            'id': 'TEST-001',
            'timestamp': datetime.now().isoformat(),
            'severity': 'HIGH',
            'category': 'malware'
        }
        
        # Should not raise exception
        dashboard.add_threat(threat_data)
        self.assertGreater(len(dashboard.threat_history), 0)


class TestFeatureIntegration(unittest.TestCase):
    """Test integration of multiple features together"""
    
    def test_end_to_end_workflow(self):
        """Test end-to-end workflow using multiple features"""
        from datetime import datetime
        
        # Initialize components
        from src.integrations.siem_integration import SIEMIntegration
        from src.response.automated_response import AutomatedResponseSystem
        from src.blockchain.verification import BlockchainVerificationService
        from src.i18n.translation import TranslationManager
        
        siem = SIEMIntegration()
        response_system = AutomatedResponseSystem()
        response_system.create_default_rules()
        blockchain = BlockchainVerificationService(difficulty=1)
        translator = TranslationManager()
        
        # Create threat data
        threat_data = {
            'id': 'INTEG-001',
            'timestamp': datetime.now().isoformat(),
            'severity': 'CRITICAL',
            'category': 'ransomware',
            'description': 'Critical ransomware attack',
            'confidence': 0.95
        }
        
        # 1. Format for SIEM
        alert = siem.format_threat_alert(threat_data)
        self.assertIsNotNone(alert)
        
        # 2. Process with automated response
        response = response_system.process_threat(threat_data)
        self.assertIsNotNone(response)
        
        # 3. Submit to blockchain
        threat_hash = blockchain.submit_threat(threat_data)
        self.assertIsNotNone(threat_hash)
        
        # 4. Translate to Spanish
        translator.set_language('es')
        translated = translator.translate_threat_data(threat_data)
        self.assertIsNotNone(translated)
        
        # Verify all steps completed
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
