#!/usr/bin/env python3
"""
Feature Validation Script

This script validates that all advanced features are properly implemented
and can be imported without errors.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def validate_imports():
    """Validate that all feature modules can be imported"""
    print("=" * 80)
    print("Advanced Features Validation Script")
    print("=" * 80)
    print("\nValidating feature implementations...\n")
    
    features = []
    
    # 1. Transformer Models
    try:
        from src.models.transformer_models import ThreatDetectionBERT, ThreatDetectionGPT
        features.append(("✓", "Transformer-based models (BERT, GPT)", "src/models/transformer_models.py"))
        print("✓ Transformer-based models (BERT, GPT)")
    except ImportError as e:
        features.append(("✗", "Transformer-based models (BERT, GPT)", f"FAILED: {e}"))
        print(f"✗ Transformer-based models: {e}")
    
    # 2. Graph Neural Networks
    try:
        from src.models.gnn_models import ThreatGraphAnalyzer
        features.append(("✓", "Graph Neural Networks", "src/models/gnn_models.py"))
        print("✓ Graph Neural Networks")
    except ImportError as e:
        features.append(("✗", "Graph Neural Networks", f"FAILED: {e}"))
        print(f"✗ Graph Neural Networks: {e}")
    
    # 3. Federated Learning
    try:
        from src.models.federated_learning import FederatedLearningOrchestrator
        features.append(("✓", "Federated Learning", "src/models/federated_learning.py"))
        print("✓ Federated Learning")
    except ImportError as e:
        features.append(("✗", "Federated Learning", f"FAILED: {e}"))
        print(f"✗ Federated Learning: {e}")
    
    # 4. SIEM Integration
    try:
        from src.integrations.siem_integration import SIEMIntegration
        features.append(("✓", "SIEM Integration", "src/integrations/siem_integration.py"))
        print("✓ SIEM Integration")
    except ImportError as e:
        features.append(("✗", "SIEM Integration", f"FAILED: {e}"))
        print(f"✗ SIEM Integration: {e}")
    
    # 5. Automated Response
    try:
        from src.response.automated_response import AutomatedResponseSystem
        features.append(("✓", "Automated Threat Response", "src/response/automated_response.py"))
        print("✓ Automated Threat Response")
    except ImportError as e:
        features.append(("✗", "Automated Threat Response", f"FAILED: {e}"))
        print(f"✗ Automated Threat Response: {e}")
    
    # 6. Dashboard
    try:
        from src.dashboard.dashboard import ThreatDashboard
        features.append(("✓", "Web-based Dashboard", "src/dashboard/dashboard.py"))
        print("✓ Web-based Dashboard")
    except ImportError as e:
        features.append(("✗", "Web-based Dashboard", f"FAILED: {e}"))
        print(f"✗ Web-based Dashboard: {e}")
    
    # 7. Multi-language Support
    try:
        from src.i18n.translation import TranslationManager
        features.append(("✓", "Multi-language Support", "src/i18n/translation.py"))
        print("✓ Multi-language Support")
    except ImportError as e:
        features.append(("✗", "Multi-language Support", f"FAILED: {e}"))
        print(f"✗ Multi-language Support: {e}")
    
    # 8. Blockchain Verification
    try:
        from src.blockchain.verification import BlockchainVerificationService
        features.append(("✓", "Blockchain Verification", "src/blockchain/verification.py"))
        print("✓ Blockchain Verification")
    except ImportError as e:
        features.append(("✗", "Blockchain Verification", f"FAILED: {e}"))
        print(f"✗ Blockchain Verification: {e}")
    
    return features


def validate_examples():
    """Validate that all example files exist"""
    print("\n" + "=" * 80)
    print("Validating Example Files")
    print("=" * 80 + "\n")
    
    examples = [
        "examples/transformer_example.py",
        "examples/gnn_example.py",
        "examples/federated_learning_example.py",
        "examples/automated_response_example.py",
        "examples/blockchain_example.py",
        "examples/i18n_example.py",
        "examples/comprehensive_integration_example.py",
        "examples/basic_usage.py",
        "examples/realtime_detection.py",
        "examples/api_server.py"
    ]
    
    missing = []
    for example in examples:
        if os.path.exists(example):
            print(f"✓ {example}")
        else:
            print(f"✗ {example} - NOT FOUND")
            missing.append(example)
    
    return missing


def validate_tests():
    """Validate that test files exist"""
    print("\n" + "=" * 80)
    print("Validating Test Files")
    print("=" * 80 + "\n")
    
    tests = [
        "tests/test_semantic_analyzer.py",
        "tests/test_preprocessor.py",
        "tests/test_advanced_features_integration.py"
    ]
    
    missing = []
    for test in tests:
        if os.path.exists(test):
            print(f"✓ {test}")
        else:
            print(f"✗ {test} - NOT FOUND")
            missing.append(test)
    
    return missing


def main():
    """Main validation function"""
    # Validate imports
    features = validate_imports()
    
    # Validate examples
    missing_examples = validate_examples()
    
    # Validate tests
    missing_tests = validate_tests()
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for status, _, _ in features if status == "✓")
    total = len(features)
    
    print(f"\nFeature Implementations: {passed}/{total} validated")
    print(f"Example Files: {10 - len(missing_examples)}/10 present")
    print(f"Test Files: {3 - len(missing_tests)}/3 present")
    
    if passed == total and not missing_examples and not missing_tests:
        print("\n✓ ALL VALIDATIONS PASSED!")
        print("All advanced features are properly implemented and available.")
        return 0
    else:
        print("\n✗ SOME VALIDATIONS FAILED")
        if passed < total:
            print(f"  - {total - passed} feature(s) failed import validation")
        if missing_examples:
            print(f"  - {len(missing_examples)} example file(s) missing")
        if missing_tests:
            print(f"  - {len(missing_tests)} test file(s) missing")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n✗ Validation script error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
