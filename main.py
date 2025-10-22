"""
Main entry point for AI-driven Semantic Framework for Cyber Threat Analysis
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing.data_preprocessor import ThreatDataPreprocessor
from src.models.ml_models import ThreatDetectionML
from src.models.dl_models import ThreatDetectionLSTM, ThreatDetectionCNN
from src.semantic_analysis.semantic_analyzer import ThreatSemanticAnalyzer
from src.realtime.detector import RealTimeThreatDetector, ThreatMonitor
from src.threat_sharing.api import ThreatSharingAPI
from src.utils.data_loader import ThreatDataLoader


def run_basic_analysis():
    """Run basic threat analysis"""
    print("\n=== Running Basic Threat Analysis ===\n")
    
    loader = ThreatDataLoader()
    data = loader.create_sample_data(num_samples=50)
    
    analyzer = ThreatSemanticAnalyzer()
    
    print(f"Analyzing {len(data)} threat samples...\n")
    
    for i in range(min(5, len(data))):
        threat_text = data.iloc[i]['summary']
        print(f"Threat {i+1}: {threat_text}")
        
        categories = analyzer.categorize_threat(threat_text)
        top_cat = max(categories.items(), key=lambda x: x[1])
        print(f"  Category: {top_cat[0]} ({top_cat[1]:.1%})")
        
        severity = analyzer.assess_threat_severity(threat_text)
        print(f"  Severity: {severity['severity']}\n")


def run_realtime_detection():
    """Run real-time threat detection"""
    print("\n=== Starting Real-time Threat Detection ===\n")
    
    analyzer = ThreatSemanticAnalyzer()
    detector = RealTimeThreatDetector(semantic_analyzer=analyzer)
    monitor = ThreatMonitor(detector)
    
    detector.start()
    print("Real-time detection started. Press Ctrl+C to stop.\n")
    
    try:
        # Simulate threat feed
        loader = ThreatDataLoader()
        sample_data = loader.create_sample_data(num_samples=10)
        
        for i, row in sample_data.iterrows():
            detector.add_threat_data({'text': row['summary']})
            print(f"Submitted threat {i+1}/10")
            import time
            time.sleep(1)
        
        time.sleep(2)
        
        stats = detector.get_statistics()
        print(f"\nStatistics:")
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Threats detected: {stats['threats_detected']}")
        print(f"  Detection rate: {stats['detection_rate']:.1%}")
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        detector.stop()


def run_api_server(host='0.0.0.0', port=5000):
    """Run API server"""
    print("\n=== Starting Threat Sharing API Server ===\n")
    
    analyzer = ThreatSemanticAnalyzer()
    detector = RealTimeThreatDetector(semantic_analyzer=analyzer)
    detector.start()
    
    api = ThreatSharingAPI(detector=detector, semantic_analyzer=analyzer)
    
    print(f"Server running on http://{host}:{port}")
    print("\nAvailable endpoints:")
    print("  GET  /api/health")
    print("  POST /api/threats/submit")
    print("  GET  /api/threats")
    print("  POST /api/threats/analyze")
    print("  GET  /api/statistics")
    print("  GET  /api/threats/search")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        api.run(host=host, port=port, debug=False)
    except KeyboardInterrupt:
        print("\nStopping server...")
        detector.stop()


def train_models():
    """Train ML and DL models"""
    print("\n=== Training Threat Detection Models ===\n")
    
    # Load data
    loader = ThreatDataLoader()
    data = loader.create_sample_data(num_samples=200)
    
    # Preprocess
    preprocessor = ThreatDataPreprocessor()
    processed_data = preprocessor.extract_features(data)
    
    # Prepare features
    X = preprocessor.create_feature_matrix(processed_data)
    y = processed_data['severity'].values
    
    # Train ML model
    print("Training Random Forest model...")
    ml_model = ThreatDetectionML(model_type='random_forest')
    metrics = ml_model.train(X, y, validation_split=0.2)
    print(f"  Training accuracy: {metrics['train_accuracy']:.2%}")
    print(f"  Validation accuracy: {metrics.get('val_accuracy', 0):.2%}")
    
    # Save model
    ml_model.save_model('models/random_forest.pkl')
    print("  Model saved to models/random_forest.pkl\n")
    
    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(
        description='CTI-sHARE-AI-driven Semantic Framework for Cyber Threat Analysis and Sharing'
    )
    
    parser.add_argument(
        'mode',
        choices=['analyze', 'realtime', 'api', 'train'],
        help='Operation mode'
    )
    
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='API server host (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='API server port (default: 5000)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CTI-sHARE-AI-driven Semantic Framework for Cyber Threat Analysis and Sharing")
    print("=" * 80)
    
    if args.mode == 'analyze':
        run_basic_analysis()
    elif args.mode == 'realtime':
        run_realtime_detection()
    elif args.mode == 'api':
        run_api_server(host=args.host, port=args.port)
    elif args.mode == 'train':
        train_models()
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
