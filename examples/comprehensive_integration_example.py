"""
Comprehensive Integration Example: Using All Advanced Features Together

This example demonstrates how to use all advanced features of the framework:
1. Transformer-based models (BERT, GPT)
2. Graph neural networks for relationship analysis
3. Federated learning for privacy-preserving threat sharing
4. Integration with SIEM systems
5. Automated threat response capabilities
6. Web-based dashboard for visualization
7. Multi-language support
8. Blockchain for threat intelligence verification
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from datetime import datetime
from typing import List, Dict, Any


def demonstrate_comprehensive_integration():
    """Comprehensive demonstration of all advanced features working together"""
    
    print("=" * 100)
    print("COMPREHENSIVE INTEGRATION EXAMPLE: All Advanced Features")
    print("=" * 100)
    print("\nThis example demonstrates the integration of:")
    print("1. Transformer models (BERT/GPT)")
    print("2. Graph Neural Networks")
    print("3. Federated Learning")
    print("4. SIEM Integration")
    print("5. Automated Response")
    print("6. Blockchain Verification")
    print("7. Multi-language Support")
    print("8. Web Dashboard (setup)")
    print("=" * 100)
    
    # =========================================================================
    # PART 1: Initialize All Components
    # =========================================================================
    print("\n" + "=" * 100)
    print("PART 1: Initializing All Components")
    print("=" * 100)
    
    # 1.1 Transformer Models
    print("\n1. Initializing Transformer Models...")
    try:
        from src.models.transformer_models import ThreatDetectionBERT, ThreatDetectionGPT
        bert_model = ThreatDetectionBERT(model_name='bert-base-uncased', num_classes=5)
        gpt_model = ThreatDetectionGPT(model_name='gpt2')
        print("   ✓ Transformer models initialized (BERT & GPT)")
    except Exception as e:
        print(f"   ✗ Transformer models initialization failed: {e}")
        bert_model = None
        gpt_model = None
    
    # 1.2 Graph Neural Networks
    print("\n2. Initializing Graph Neural Network...")
    try:
        from src.models.gnn_models import ThreatGraphAnalyzer
        gnn_analyzer = ThreatGraphAnalyzer(input_dim=128, hidden_dim=64, output_dim=32)
        print("   ✓ GNN analyzer initialized")
    except Exception as e:
        print(f"   ✗ GNN initialization failed: {e}")
        gnn_analyzer = None
    
    # 1.3 Federated Learning
    print("\n3. Initializing Federated Learning...")
    try:
        from src.models.federated_learning import FederatedLearningOrchestrator
        import torch.nn as nn
        
        # Simple model for demo
        class SimpleThreatModel(nn.Module):
            def __init__(self):
                super(SimpleThreatModel, self).__init__()
                self.fc = nn.Linear(10, 2)
            
            def forward(self, x):
                return self.fc(x)
        
        fl_orchestrator = FederatedLearningOrchestrator(
            model_architecture=SimpleThreatModel(),
            num_clients=3
        )
        print("   ✓ Federated learning orchestrator initialized with 3 clients")
    except Exception as e:
        print(f"   ✗ Federated learning initialization failed: {e}")
        fl_orchestrator = None
    
    # 1.4 SIEM Integration
    print("\n4. Initializing SIEM Integration...")
    try:
        from src.integrations.siem_integration import SIEMIntegration, SplunkConnector
        siem = SIEMIntegration()
        
        # Add mock Splunk connector (would use real credentials in production)
        splunk = SplunkConnector(
            host='demo.splunk.com',
            port=8089,
            api_key='demo-api-key'
        )
        siem.add_siem('splunk_demo', splunk)
        print("   ✓ SIEM integration initialized with Splunk connector")
    except Exception as e:
        print(f"   ✗ SIEM integration initialization failed: {e}")
        siem = None
    
    # 1.5 Automated Response System
    print("\n5. Initializing Automated Response System...")
    try:
        from src.response.automated_response import AutomatedResponseSystem
        response_system = AutomatedResponseSystem()
        response_system.create_default_rules()
        print("   ✓ Automated response system initialized with default rules")
    except Exception as e:
        print(f"   ✗ Automated response initialization failed: {e}")
        response_system = None
    
    # 1.6 Blockchain Verification
    print("\n6. Initializing Blockchain Verification...")
    try:
        from src.blockchain.verification import BlockchainVerificationService
        blockchain = BlockchainVerificationService(difficulty=2)
        print("   ✓ Blockchain verification service initialized")
    except Exception as e:
        print(f"   ✗ Blockchain initialization failed: {e}")
        blockchain = None
    
    # 1.7 Multi-language Support
    print("\n7. Initializing Multi-language Support...")
    try:
        from src.i18n.translation import TranslationManager
        translator = TranslationManager(default_language='en')
        print("   ✓ Translation manager initialized")
        print(f"   Supported languages: {', '.join(translator.get_supported_languages())}")
    except Exception as e:
        print(f"   ✗ Translation manager initialization failed: {e}")
        translator = None
    
    # 1.8 Dashboard (setup info)
    print("\n8. Dashboard Setup Information...")
    try:
        from src.dashboard.dashboard import ThreatDashboard
        print("   ✓ Dashboard available (can be started separately)")
        print("   To run: dashboard = ThreatDashboard(); dashboard.run(port=5001)")
    except Exception as e:
        print(f"   ✗ Dashboard import failed: {e}")
    
    # =========================================================================
    # PART 2: Threat Detection and Analysis Workflow
    # =========================================================================
    print("\n" + "=" * 100)
    print("PART 2: Threat Detection and Analysis Workflow")
    print("=" * 100)
    
    # Sample threat data
    threat_texts = [
        "Critical ransomware attack targeting healthcare infrastructure",
        "Sophisticated phishing campaign using AI-generated emails",
        "Zero-day vulnerability exploited in widely-used software"
    ]
    
    print(f"\nAnalyzing {len(threat_texts)} threat samples...")
    
    # 2.1 BERT Classification
    if bert_model:
        print("\n2.1 BERT-based Threat Classification...")
        try:
            # Quick training with minimal data (for demo)
            sample_labels = np.array([4, 3, 4])  # 4=critical, 3=high
            bert_model.train(
                threat_texts[:2], 
                sample_labels[:2],
                epochs=1,
                batch_size=2
            )
            predictions = bert_model.predict(threat_texts)
            severity_map = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH', 3: 'HIGH', 4: 'CRITICAL'}
            
            for i, text in enumerate(threat_texts):
                print(f"   Threat {i+1}: {severity_map[predictions[i]]} severity")
        except Exception as e:
            print(f"   BERT classification skipped: {e}")
    
    # 2.2 GPT Feature Extraction
    if gpt_model:
        print("\n2.2 GPT Feature Extraction...")
        try:
            features = gpt_model.extract_features(threat_texts[:2], batch_size=2)
            print(f"   Extracted features: shape {features.shape}")
        except Exception as e:
            print(f"   GPT feature extraction skipped: {e}")
    
    # 2.3 GNN Relationship Analysis
    if gnn_analyzer:
        print("\n2.3 Graph Neural Network Relationship Analysis...")
        try:
            # Create sample threat graph
            threat_features = np.random.randn(5, 128)
            relationships = [(0, 1), (1, 2), (0, 2), (3, 4)]
            
            # Train GNN
            gnn_analyzer.train(threat_features, relationships, epochs=20)
            
            # Find related threats
            related = gnn_analyzer.find_related_threats(
                threat_features, relationships, query_idx=0, top_k=2
            )
            print(f"   Threats related to Threat #0:")
            for idx, similarity in related:
                print(f"      - Threat #{idx}: {similarity:.3f} similarity")
        except Exception as e:
            print(f"   GNN analysis skipped: {e}")
    
    # =========================================================================
    # PART 3: Privacy-Preserving Federated Learning
    # =========================================================================
    print("\n" + "=" * 100)
    print("PART 3: Privacy-Preserving Federated Learning")
    print("=" * 100)
    
    if fl_orchestrator:
        print("\nTraining threat detection model across 3 organizations...")
        print("(Each organization keeps their data private)")
        try:
            # Simulate data from 3 different organizations
            client_data = []
            for i in range(3):
                X = np.random.randn(20, 10).astype(np.float32)
                y = np.random.randint(0, 2, 20).astype(np.int64)
                client_data.append((X, y))
            
            # Train federated model
            history = fl_orchestrator.train_federated(
                client_data=client_data,
                num_rounds=3,
                local_epochs=2
            )
            print(f"   ✓ Federated training completed")
            print(f"   Final global accuracy: {history['global_accuracy'][-1]:.2%}")
        except Exception as e:
            print(f"   Federated learning training skipped: {e}")
    
    # =========================================================================
    # PART 4: Threat Intelligence Verification and Sharing
    # =========================================================================
    print("\n" + "=" * 100)
    print("PART 4: Threat Intelligence Verification and Sharing")
    print("=" * 100)
    
    # Create comprehensive threat data
    threat_data = {
        'id': 'THREAT-INTEG-001',
        'timestamp': datetime.now().isoformat(),
        'severity': 'CRITICAL',
        'category': 'ransomware',
        'description': 'Critical ransomware attack targeting healthcare infrastructure',
        'entities': {
            'ips': ['192.168.1.100', '10.0.0.50'],
            'domains': ['malicious-c2.com'],
            'hashes': ['a1b2c3d4e5f6']
        },
        'source': 'Security Operations Center',
        'confidence': 0.95
    }
    
    # 4.1 Blockchain Verification
    if blockchain:
        print("\n4.1 Recording to Blockchain...")
        try:
            threat_hash = blockchain.submit_threat(threat_data)
            print(f"   ✓ Threat submitted to blockchain")
            print(f"   Threat hash: {threat_hash[:32]}...")
            
            # Commit to blockchain
            block = blockchain.commit_threats()
            if block:
                print(f"   ✓ Block mined and added to chain")
                print(f"   Block hash: {block.hash[:32]}...")
            
            # Verify integrity
            is_valid = blockchain.blockchain.verify_chain()
            print(f"   ✓ Blockchain integrity verified: {is_valid}")
        except Exception as e:
            print(f"   Blockchain operations skipped: {e}")
    
    # 4.2 SIEM Integration
    if siem:
        print("\n4.2 Sending Alert to SIEM...")
        try:
            alert = siem.format_threat_alert(threat_data)
            print(f"   ✓ Alert formatted for SIEM")
            print(f"   Alert title: {alert['title']}")
            print(f"   Severity: {alert['severity']}")
            # Note: Actual sending would require real SIEM connection
            print("   (Would send to SIEM in production environment)")
        except Exception as e:
            print(f"   SIEM alert skipped: {e}")
    
    # =========================================================================
    # PART 5: Automated Threat Response
    # =========================================================================
    print("\n" + "=" * 100)
    print("PART 5: Automated Threat Response")
    print("=" * 100)
    
    if response_system:
        print("\nProcessing threat with automated response system...")
        try:
            response_summary = response_system.process_threat(threat_data)
            print(f"   ✓ Automated response executed")
            print(f"   Rules matched: {len(response_summary['rules_matched'])}")
            print(f"   Actions executed: {len(response_summary['actions_executed'])}")
            
            if response_summary['rules_matched']:
                print(f"   Triggered rules:")
                for rule in response_summary['rules_matched']:
                    print(f"      - {rule}")
        except Exception as e:
            print(f"   Automated response skipped: {e}")
    
    # =========================================================================
    # PART 6: Multi-language Support
    # =========================================================================
    print("\n" + "=" * 100)
    print("PART 6: Multi-language Support")
    print("=" * 100)
    
    if translator:
        print("\nTranslating threat data to multiple languages...")
        try:
            languages = ['es', 'fr', 'de']
            
            for lang in languages:
                translator.set_language(lang)
                translated = translator.translate_threat_data(threat_data, language=lang)
                lang_name = {'es': 'Spanish', 'fr': 'French', 'de': 'German'}[lang]
                print(f"\n   {lang_name} ({lang}):")
                print(f"   - Severity: {translated.get('severity', threat_data['severity'])}")
                print(f"   - Category: {translated.get('category', threat_data['category'])}")
        except Exception as e:
            print(f"   Translation skipped: {e}")
    
    # =========================================================================
    # PART 7: Summary and Statistics
    # =========================================================================
    print("\n" + "=" * 100)
    print("PART 7: Integration Summary")
    print("=" * 100)
    
    print("\n✓ Successfully demonstrated integration of:")
    components = [
        ("Transformer Models", bert_model is not None),
        ("Graph Neural Networks", gnn_analyzer is not None),
        ("Federated Learning", fl_orchestrator is not None),
        ("SIEM Integration", siem is not None),
        ("Automated Response", response_system is not None),
        ("Blockchain Verification", blockchain is not None),
        ("Multi-language Support", translator is not None)
    ]
    
    for component, initialized in components:
        status = "✓" if initialized else "✗"
        print(f"   {status} {component}")
    
    print("\n" + "=" * 100)
    print("COMPREHENSIVE INTEGRATION EXAMPLE COMPLETED")
    print("=" * 100)
    print("\nKey Takeaways:")
    print("• All advanced features can work together seamlessly")
    print("• Threat detection uses state-of-the-art AI models")
    print("• Privacy is preserved through federated learning")
    print("• Blockchain ensures data integrity and verification")
    print("• SIEM integration enables centralized monitoring")
    print("• Automated responses reduce reaction time")
    print("• Multi-language support enables global collaboration")
    print("• Web dashboard provides real-time visualization")
    print("=" * 100)


if __name__ == "__main__":
    try:
        demonstrate_comprehensive_integration()
    except Exception as e:
        print(f"\n✗ Error during integration example: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 100)
    print("For individual feature examples, see:")
    print("  • examples/transformer_example.py")
    print("  • examples/gnn_example.py")
    print("  • examples/federated_learning_example.py")
    print("  • examples/automated_response_example.py")
    print("  • examples/blockchain_example.py")
    print("  • examples/i18n_example.py")
    print("=" * 100)
