"""
Federated Learning framework for privacy-preserving threat intelligence sharing
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import logging
import copy

logger = logging.getLogger(__name__)


class FederatedClient:
    """
    Federated learning client for local threat detection model training
    """
    
    def __init__(self, client_id: str, model: nn.Module, learning_rate: float = 0.01):
        """
        Initialize federated client
        
        Args:
            client_id: Unique identifier for this client
            model: Local model (should match server model architecture)
            learning_rate: Learning rate for local training
        """
        self.client_id = client_id
        self.model = model
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def set_model_parameters(self, parameters: Dict[str, np.ndarray]) -> None:
        """
        Update local model with parameters from server
        
        Args:
            parameters: Model parameters as numpy arrays
        """
        state_dict = {}
        for name, param in parameters.items():
            state_dict[name] = torch.FloatTensor(param)
        
        self.model.load_state_dict(state_dict)
        logger.info(f"Client {self.client_id}: Updated model parameters from server")
    
    def get_model_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get local model parameters
        
        Returns:
            Model parameters as numpy arrays
        """
        parameters = {}
        for name, param in self.model.state_dict().items():
            parameters[name] = param.cpu().numpy()
        return parameters
    
    def train_local(self, X: np.ndarray, y: np.ndarray,
                   epochs: int = 5, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train model on local data
        
        Args:
            X: Local training features
            y: Local training labels
            epochs: Number of local training epochs
            batch_size: Batch size
            
        Returns:
            Training metrics
        """
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        metrics = {'loss': [], 'accuracy': []}
        
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            # Mini-batch training
            for i in range(0, len(X), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            accuracy = correct / total
            metrics['loss'].append(epoch_loss / (len(X) / batch_size))
            metrics['accuracy'].append(accuracy)
        
        logger.info(f"Client {self.client_id}: Local training completed - "
                   f"Final accuracy: {accuracy:.4f}")
        
        return metrics


class FederatedServer:
    """
    Federated learning server for aggregating client models
    """
    
    def __init__(self, model: nn.Module, aggregation_method: str = 'fedavg'):
        """
        Initialize federated server
        
        Args:
            model: Global model architecture
            aggregation_method: Method for aggregating client models ('fedavg', 'fedprox')
        """
        self.model = model
        self.aggregation_method = aggregation_method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.round_number = 0
        
    def get_model_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get global model parameters
        
        Returns:
            Model parameters as numpy arrays
        """
        parameters = {}
        for name, param in self.model.state_dict().items():
            parameters[name] = param.cpu().numpy()
        return parameters
    
    def aggregate_models(self, client_parameters: List[Dict[str, np.ndarray]],
                        client_weights: Optional[List[float]] = None) -> None:
        """
        Aggregate client models using FedAvg or weighted averaging
        
        Args:
            client_parameters: List of client model parameters
            client_weights: Optional weights for each client (e.g., based on data size)
        """
        if not client_parameters:
            logger.warning("No client parameters to aggregate")
            return
        
        # Equal weights if not provided
        if client_weights is None:
            client_weights = [1.0 / len(client_parameters)] * len(client_parameters)
        else:
            # Normalize weights
            total_weight = sum(client_weights)
            client_weights = [w / total_weight for w in client_weights]
        
        # FedAvg: weighted average of client parameters
        aggregated_params = {}
        
        for param_name in client_parameters[0].keys():
            # Weighted sum of parameters
            weighted_sum = np.zeros_like(client_parameters[0][param_name])
            for client_param, weight in zip(client_parameters, client_weights):
                weighted_sum += weight * client_param[param_name]
            
            aggregated_params[param_name] = weighted_sum
        
        # Update global model
        state_dict = {}
        for name, param in aggregated_params.items():
            state_dict[name] = torch.FloatTensor(param)
        
        self.model.load_state_dict(state_dict)
        self.round_number += 1
        
        logger.info(f"Server: Aggregated {len(client_parameters)} client models "
                   f"(Round {self.round_number})")
    
    def evaluate_global_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate global model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        y_tensor = torch.LongTensor(y_test).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            
            accuracy = (predicted == y_tensor).sum().item() / len(y_test)
            
            # Calculate loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, y_tensor).item()
        
        metrics = {
            'accuracy': accuracy,
            'loss': loss
        }
        
        logger.info(f"Global model evaluation - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        
        return metrics


class FederatedLearningOrchestrator:
    """
    Orchestrator for federated learning process
    """
    
    def __init__(self, model_architecture: nn.Module, num_clients: int = 5):
        """
        Initialize federated learning orchestrator
        
        Args:
            model_architecture: Model architecture to use
            num_clients: Number of federated clients
        """
        self.server = FederatedServer(model_architecture)
        self.clients = []
        
        # Create clients with copies of the model
        for i in range(num_clients):
            client_model = copy.deepcopy(model_architecture)
            client = FederatedClient(f"client_{i}", client_model)
            self.clients.append(client)
        
        logger.info(f"Initialized federated learning with {num_clients} clients")
    
    def train_federated(self, client_data: List[tuple[np.ndarray, np.ndarray]],
                       num_rounds: int = 10, local_epochs: int = 5,
                       test_data: Optional[tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Run federated learning process
        
        Args:
            client_data: List of (X, y) tuples for each client
            num_rounds: Number of federated learning rounds
            local_epochs: Number of local training epochs per round
            test_data: Optional test data (X_test, y_test) for evaluation
            
        Returns:
            Training history
        """
        if len(client_data) != len(self.clients):
            raise ValueError(f"Number of client data sets ({len(client_data)}) "
                           f"must match number of clients ({len(self.clients)})")
        
        history = {
            'round': [],
            'global_accuracy': [],
            'global_loss': []
        }
        
        for round_num in range(num_rounds):
            logger.info(f"\n=== Federated Learning Round {round_num + 1}/{num_rounds} ===")
            
            # Broadcast global model to clients
            global_params = self.server.get_model_parameters()
            for client in self.clients:
                client.set_model_parameters(global_params)
            
            # Local training on each client
            client_params = []
            client_weights = []
            
            for i, client in enumerate(self.clients):
                X_local, y_local = client_data[i]
                
                # Train locally
                client.train_local(X_local, y_local, epochs=local_epochs)
                
                # Collect updated parameters
                client_params.append(client.get_model_parameters())
                client_weights.append(len(X_local))  # Weight by data size
            
            # Aggregate client models
            self.server.aggregate_models(client_params, client_weights)
            
            # Evaluate global model if test data provided
            if test_data:
                X_test, y_test = test_data
                metrics = self.server.evaluate_global_model(X_test, y_test)
                
                history['round'].append(round_num + 1)
                history['global_accuracy'].append(metrics['accuracy'])
                history['global_loss'].append(metrics['loss'])
        
        logger.info("\nFederated learning completed")
        return history
    
    def get_global_model(self) -> nn.Module:
        """
        Get the trained global model
        
        Returns:
            Global model
        """
        return self.server.model
    
    def save_global_model(self, filepath: str) -> None:
        """
        Save global model to disk
        
        Args:
            filepath: Path to save model
        """
        torch.save(self.server.model.state_dict(), filepath)
        logger.info(f"Global model saved to {filepath}")
    
    def load_global_model(self, filepath: str) -> None:
        """
        Load global model from disk
        
        Args:
            filepath: Path to load model from
        """
        self.server.model.load_state_dict(torch.load(filepath))
        logger.info(f"Global model loaded from {filepath}")
