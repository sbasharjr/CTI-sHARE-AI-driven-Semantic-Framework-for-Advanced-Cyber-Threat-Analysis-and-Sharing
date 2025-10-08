"""
Example: Using Graph Neural Networks for Threat Relationship Analysis
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.gnn_models import ThreatGraphAnalyzer
import numpy as np


def example_threat_graph_analysis():
    """Example of using GNN for threat relationship analysis"""
    print("=" * 80)
    print("Graph Neural Network - Threat Relationship Analysis")
    print("=" * 80)
    
    # Create sample threat features
    # Each row represents a threat with some features
    num_threats = 10
    feature_dim = 128
    
    print(f"\n1. Creating sample threat data...")
    print(f"   Number of threats: {num_threats}")
    print(f"   Feature dimension: {feature_dim}")
    
    # Random features for demo purposes
    threat_features = np.random.randn(num_threats, feature_dim)
    
    # Define relationships between threats (edges in the graph)
    # Format: (source_threat_idx, target_threat_idx)
    relationships = [
        (0, 1), (0, 2),  # Threat 0 related to threats 1 and 2
        (1, 3), (2, 3),  # Threats 1 and 2 both related to threat 3
        (3, 4), (3, 5),  # Threat 3 related to threats 4 and 5
        (6, 7), (7, 8),  # Separate cluster: 6-7-8
        (8, 9)           # 8 connected to 9
    ]
    
    print(f"   Number of relationships: {len(relationships)}")
    
    # Initialize GNN analyzer
    print("\n2. Initializing GNN analyzer...")
    gnn_analyzer = ThreatGraphAnalyzer(
        input_dim=feature_dim,
        hidden_dim=64,
        output_dim=32,
        num_layers=3
    )
    print("   GNN analyzer initialized")
    
    # Train GNN (unsupervised learning)
    print("\n3. Training GNN on threat graph...")
    history = gnn_analyzer.train(
        threat_features=threat_features,
        relationships=relationships,
        labels=None,  # Unsupervised
        epochs=50,
        learning_rate=0.01
    )
    print(f"   Training completed - Final loss: {history['loss'][-1]:.4f}")
    
    # Get threat embeddings
    print("\n4. Extracting threat embeddings...")
    embeddings = gnn_analyzer.get_embeddings(threat_features, relationships)
    print(f"   Embedding shape: {embeddings.shape}")
    
    # Find related threats
    print("\n5. Finding threats related to Threat #0...")
    related = gnn_analyzer.find_related_threats(
        threat_features, relationships, query_idx=0, top_k=3
    )
    
    print("   Most related threats:")
    for threat_idx, similarity in related:
        print(f"      Threat #{threat_idx}: Similarity = {similarity:.4f}")
    
    # Detect threat communities
    print("\n6. Detecting threat communities...")
    communities = gnn_analyzer.detect_threat_communities(
        threat_features, relationships, num_communities=3
    )
    
    print("   Community assignments:")
    for i, community in enumerate(communities):
        print(f"      Threat #{i} -> Community {community}")
    
    # Count threats per community
    from collections import Counter
    community_counts = Counter(communities)
    print("\n   Community distribution:")
    for community, count in sorted(community_counts.items()):
        print(f"      Community {community}: {count} threats")
    
    print("\n" + "=" * 80)
    print("GNN Analysis Complete")
    print("=" * 80)


if __name__ == "__main__":
    try:
        example_threat_graph_analysis()
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure PyTorch and scikit-learn are installed")
        import traceback
        traceback.print_exc()
