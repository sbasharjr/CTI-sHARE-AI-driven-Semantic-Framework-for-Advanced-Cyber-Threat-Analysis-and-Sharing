"""
Example: Using Transformer-based Models (BERT, GPT) for Threat Detection
"""

from src.models.transformer_models import ThreatDetectionBERT, ThreatDetectionGPT
import numpy as np


def example_bert_classification():
    """Example of using BERT for threat classification"""
    print("=" * 80)
    print("BERT-based Threat Detection Example")
    print("=" * 80)
    
    # Sample threat texts
    train_texts = [
        "Ransomware attack detected on healthcare network",
        "Phishing campaign targeting financial institutions",
        "DDoS attack from botnet targeting web servers",
        "Malware infection spreading through email attachments",
        "SQL injection vulnerability found in web application"
    ]
    
    # Labels (0: low, 1: medium, 2: high, 3: critical)
    train_labels = np.array([3, 2, 2, 2, 1])
    
    print("\n1. Initializing BERT model...")
    bert_model = ThreatDetectionBERT(model_name='bert-base-uncased', num_classes=4)
    print("   Model initialized successfully")
    
    print("\n2. Training BERT model...")
    print("   Note: This requires GPU for faster training")
    # For demo purposes, we'll use a very small training set
    # In production, use more data and more epochs
    history = bert_model.train(
        train_texts[:3],
        train_labels[:3],
        val_texts=train_texts[3:],
        val_labels=train_labels[3:],
        epochs=1,
        batch_size=2
    )
    print("   Training completed")
    
    print("\n3. Making predictions...")
    test_texts = [
        "Critical zero-day vulnerability exploited in the wild",
        "Suspicious network activity detected"
    ]
    
    predictions = bert_model.predict(test_texts)
    probabilities = bert_model.predict_proba(test_texts)
    
    severity_map = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH', 3: 'CRITICAL'}
    
    for i, text in enumerate(test_texts):
        print(f"\n   Text: {text}")
        print(f"   Predicted Severity: {severity_map[predictions[i]]}")
        print(f"   Confidence: {probabilities[i][predictions[i]]:.2%}")
    
    print("\n" + "=" * 80)


def example_gpt_feature_extraction():
    """Example of using GPT for threat feature extraction"""
    print("\n" + "=" * 80)
    print("GPT-based Feature Extraction Example")
    print("=" * 80)
    
    print("\n1. Initializing GPT model...")
    gpt_model = ThreatDetectionGPT(model_name='gpt2')
    print("   Model initialized successfully")
    
    print("\n2. Extracting features from threat descriptions...")
    texts = [
        "Advanced persistent threat actor using custom malware",
        "Data breach exposed customer information"
    ]
    
    features = gpt_model.extract_features(texts, batch_size=2)
    
    print(f"   Extracted feature vectors with shape: {features.shape}")
    print(f"   Feature vector dimension: {features.shape[1]}")
    
    print("\n3. Generating threat report...")
    prompt = "Threat Alert: Ransomware attack detected."
    report = gpt_model.generate_threat_report(prompt, max_length=100)
    
    print(f"\n   Generated Report:")
    print(f"   {report}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Transformer-based Threat Detection Examples")
    print("=" * 80)
    print("\nNote: These examples require transformer models which are large.")
    print("Make sure you have sufficient memory and GPU for faster processing.")
    print("The models will be downloaded on first use.")
    
    try:
        example_bert_classification()
    except Exception as e:
        print(f"\nBERT example failed: {e}")
        print("This is expected if transformers/torch are not installed")
    
    try:
        example_gpt_feature_extraction()
    except Exception as e:
        print(f"\nGPT example failed: {e}")
        print("This is expected if transformers/torch are not installed")
    
    print("\n" + "=" * 80)
    print("Examples completed")
    print("=" * 80)
