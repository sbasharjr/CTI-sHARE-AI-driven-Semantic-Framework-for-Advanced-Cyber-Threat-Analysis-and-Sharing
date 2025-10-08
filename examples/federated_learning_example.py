"""
Example: Federated Learning for Privacy-Preserving Threat Sharing
"""

import torch
import torch.nn as nn
import numpy as np
from src.models.federated_learning import FederatedLearningOrchestrator


# Simple neural network for demonstration
class SimpleThreatClassifier(nn.Module):
    """Simple neural network for threat classification"""
    
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=3):
        super(SimpleThreatClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def example_federated_learning():
    """Example of federated learning for threat detection"""
    print("=" * 80)
    print("Federated Learning for Privacy-Preserving Threat Sharing")
    print("=" * 80)
    
    # Simulation parameters
    num_clients = 5
    input_dim = 10
    num_classes = 3
    samples_per_client = 100
    
    print(f"\n1. Setting up federated learning simulation...")
    print(f"   Number of clients: {num_clients}")
    print(f"   Samples per client: {samples_per_client}")
    print(f"   Feature dimension: {input_dim}")
    print(f"   Number of classes: {num_classes}")
    
    # Create model architecture
    model = SimpleThreatClassifier(input_dim, hidden_dim=20, output_dim=num_classes)
    
    # Initialize federated learning orchestrator
    print("\n2. Initializing federated learning orchestrator...")
    orchestrator = FederatedLearningOrchestrator(
        model_architecture=model,
        num_clients=num_clients
    )
    print("   Orchestrator initialized with global model and client models")
    
    # Generate synthetic data for each client (simulating local threat data)
    print("\n3. Generating local data for each client...")
    print("   (In real scenario, each organization would have their own threat data)")
    
    client_data = []
    for i in range(num_clients):
        # Generate random data for demonstration
        X = np.random.randn(samples_per_client, input_dim).astype(np.float32)
        y = np.random.randint(0, num_classes, samples_per_client)
        client_data.append((X, y))
        print(f"   Client {i}: {X.shape[0]} samples")
    
    # Generate test data
    X_test = np.random.randn(50, input_dim).astype(np.float32)
    y_test = np.random.randint(0, num_classes, 50)
    test_data = (X_test, y_test)
    
    # Run federated learning
    print("\n4. Running federated learning...")
    print("   Each round:")
    print("   - Server broadcasts global model to clients")
    print("   - Clients train on local data (privacy preserved)")
    print("   - Server aggregates client models")
    print()
    
    history = orchestrator.train_federated(
        client_data=client_data,
        num_rounds=5,
        local_epochs=3,
        test_data=test_data
    )
    
    # Display results
    print("\n5. Federated Learning Results:")
    print("-" * 80)
    print(f"{'Round':<10} {'Accuracy':<15} {'Loss':<15}")
    print("-" * 80)
    
    for i in range(len(history['round'])):
        round_num = history['round'][i]
        accuracy = history['global_accuracy'][i]
        loss = history['global_loss'][i]
        print(f"{round_num:<10} {accuracy*100:>6.2f}%        {loss:>6.4f}")
    
    print("-" * 80)
    
    # Get final global model
    global_model = orchestrator.get_global_model()
    print("\n6. Final Global Model Statistics:")
    total_params = sum(p.numel() for p in global_model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Final accuracy: {history['global_accuracy'][-1]*100:.2f}%")
    print(f"   Final loss: {history['global_loss'][-1]:.4f}")
    
    # Save global model
    print("\n7. Saving global model...")
    model_path = "/tmp/federated_global_model.pth"
    orchestrator.save_global_model(model_path)
    print(f"   Model saved to: {model_path}")
    
    print("\n" + "=" * 80)
    print("Key Benefits of Federated Learning:")
    print("=" * 80)
    print("✓ Privacy: Raw data never leaves client organizations")
    print("✓ Collaboration: Organizations benefit from collective intelligence")
    print("✓ Compliance: Meets data protection regulations (GDPR, etc.)")
    print("✓ Security: No central data storage risk")
    print("=" * 80)


if __name__ == "__main__":
    try:
        example_federated_learning()
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure PyTorch is installed")
        import traceback
        traceback.print_exc()
