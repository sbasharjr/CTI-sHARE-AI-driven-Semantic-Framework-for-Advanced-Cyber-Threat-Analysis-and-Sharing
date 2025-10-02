"""
Basic usage example of the AI-driven Semantic Framework
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.data_preprocessor import ThreatDataPreprocessor
from src.models.ml_models import ThreatDetectionML
from src.semantic_analysis.semantic_analyzer import ThreatSemanticAnalyzer
from src.utils.data_loader import ThreatDataLoader


def main():
    print("=" * 80)
    print("AI-driven Semantic Framework for Cyber Threat Analysis")
    print("=" * 80)
    print()
    
    # 1. Load or create sample data
    print("1. Loading sample threat data...")
    loader = ThreatDataLoader()
    data = loader.create_sample_data(num_samples=100)
    print(f"   Loaded {len(data)} threat samples")
    print()
    
    # 2. Preprocess data
    print("2. Preprocessing threat data...")
    preprocessor = ThreatDataPreprocessor()
    processed_data = preprocessor.extract_features(data)
    print(f"   Extracted features from {len(processed_data)} samples")
    print()
    
    # 3. Semantic analysis
    print("3. Performing semantic analysis...")
    analyzer = ThreatSemanticAnalyzer()
    
    # Analyze first few threats
    for i in range(min(3, len(data))):
        threat_text = data.iloc[i]['summary']
        print(f"\n   Threat {i+1}: {threat_text[:60]}...")
        
        # Categorize threat
        categories = analyzer.categorize_threat(threat_text)
        top_category = max(categories.items(), key=lambda x: x[1])
        print(f"   Category: {top_category[0]} (confidence: {top_category[1]:.2%})")
        
        # Extract entities
        entities = analyzer.extract_threat_entities(threat_text)
        if any(entities.values()):
            print(f"   Entities: {entities}")
        
        # Assess severity
        severity = analyzer.assess_threat_severity(threat_text)
        print(f"   Severity: {severity['severity']} (level {severity['level']}/5)")
    
    print()
    
    # 4. Train ML model
    print("4. Training machine learning model...")
    try:
        # Create feature matrix
        X = preprocessor.create_feature_matrix(processed_data)
        y = processed_data['severity'].values if 'severity' in processed_data.columns else None
        
        if y is not None:
            ml_model = ThreatDetectionML(model_type='random_forest')
            metrics = ml_model.train(X, y, validation_split=0.2)
            print(f"   Training accuracy: {metrics['train_accuracy']:.2%}")
            if 'val_accuracy' in metrics:
                print(f"   Validation accuracy: {metrics['val_accuracy']:.2%}")
        else:
            print("   Skipping ML training (no labels available)")
    except Exception as e:
        print(f"   ML training error: {e}")
    
    print()
    
    # 5. Threat similarity analysis
    print("5. Analyzing threat similarities...")
    threat_texts = data['summary'].tolist()[:10]
    similarity_matrix = analyzer.compute_threat_similarity(threat_texts)
    clusters = analyzer.cluster_similar_threats(threat_texts, threshold=0.7)
    print(f"   Found {len(clusters)} threat clusters")
    print()
    
    # 6. Trend analysis
    print("6. Analyzing threat trends...")
    threats = [{'text': row['summary'], 'timestamp': row['event_date']} 
               for _, row in data.iterrows()]
    trends = analyzer.analyze_threat_trends(threats)
    print(f"   Total threats analyzed: {trends['total_threats']}")
    print(f"   Unique categories: {trends['unique_categories']}")
    print("   Category distribution:")
    for cat, count in sorted(trends['category_distribution'].items(), 
                            key=lambda x: x[1], reverse=True):
        print(f"     - {cat}: {count}")
    
    print()
    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
