"""
Graph Neural Networks for threat relationship analysis
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class GraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer for threat relationship modeling
    """
    
    def __init__(self, in_features: int, out_features: int):
        """
        Initialize Graph Convolutional Layer
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
        """
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through graph convolution
        
        Args:
            x: Node features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Normalize adjacency matrix
        degree = torch.sum(adj, dim=1, keepdim=True)
        degree = torch.pow(degree, -0.5)
        degree[torch.isinf(degree)] = 0
        normalized_adj = degree * adj * degree.t()
        
        # Graph convolution: D^-1/2 A D^-1/2 X W
        support = self.linear(x)
        output = torch.mm(normalized_adj, support)
        
        return output


class ThreatRelationshipGNN(nn.Module):
    """
    Graph Neural Network for analyzing threat relationships
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64,
                 output_dim: int = 32, num_layers: int = 3, dropout: float = 0.3):
        """
        Initialize GNN model
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super(ThreatRelationshipGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GNN layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GraphConvLayer(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GraphConvLayer(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(GraphConvLayer(hidden_dim, output_dim))
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GNN
        
        Args:
            x: Node features
            adj: Adjacency matrix
            
        Returns:
            Node embeddings
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            
            # Apply ReLU and dropout to all but last layer
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class ThreatGraphAnalyzer:
    """
    Analyzer for threat relationships using Graph Neural Networks
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64,
                 output_dim: int = 32, num_layers: int = 3):
        """
        Initialize Threat Graph Analyzer
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of GNN layers
        """
        self.model = ThreatRelationshipGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers
        )
        
        self.device = self.model.device
        self.is_trained = False
        
    def build_threat_graph(self, threat_features: np.ndarray,
                          relationships: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph representation of threats
        
        Args:
            threat_features: Feature matrix for threats [num_threats, feature_dim]
            relationships: List of threat relationships (source, target) tuples
            
        Returns:
            Node features and adjacency matrix
        """
        num_threats = threat_features.shape[0]
        
        # Convert features to tensor
        x = torch.FloatTensor(threat_features).to(self.device)
        
        # Build adjacency matrix
        adj = torch.zeros((num_threats, num_threats), device=self.device)
        
        # Add edges from relationships
        for source, target in relationships:
            if 0 <= source < num_threats and 0 <= target < num_threats:
                adj[source, target] = 1.0
                adj[target, source] = 1.0  # Undirected graph
        
        # Add self-loops
        adj = adj + torch.eye(num_threats, device=self.device)
        
        return x, adj
    
    def train(self, threat_features: np.ndarray, relationships: List[Tuple[int, int]],
              labels: np.ndarray = None, epochs: int = 100, learning_rate: float = 0.01) -> Dict[str, Any]:
        """
        Train GNN model
        
        Args:
            threat_features: Feature matrix for threats
            relationships: List of threat relationships
            labels: Optional node labels for supervised training
            epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        x, adj = self.build_threat_graph(threat_features, relationships)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        history = {'loss': []}
        
        # If labels provided, use supervised learning
        if labels is not None:
            labels_tensor = torch.LongTensor(labels).to(self.device)
            criterion = nn.CrossEntropyLoss()
            
            self.model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                embeddings = self.model(x, adj)
                
                # For node classification
                loss = criterion(embeddings, labels_tensor)
                
                loss.backward()
                optimizer.step()
                
                history['loss'].append(loss.item())
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        else:
            # Unsupervised learning using reconstruction loss
            self.model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                embeddings = self.model(x, adj)
                
                # Reconstruction loss: try to reconstruct adjacency from embeddings
                reconstructed = torch.mm(embeddings, embeddings.t())
                loss = F.mse_loss(reconstructed, adj)
                
                loss.backward()
                optimizer.step()
                
                history['loss'].append(loss.item())
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        self.is_trained = True
        return history
    
    def get_embeddings(self, threat_features: np.ndarray,
                      relationships: List[Tuple[int, int]]) -> np.ndarray:
        """
        Get threat embeddings from GNN
        
        Args:
            threat_features: Feature matrix for threats
            relationships: List of threat relationships
            
        Returns:
            Threat embeddings
        """
        self.model.eval()
        
        x, adj = self.build_threat_graph(threat_features, relationships)
        
        with torch.no_grad():
            embeddings = self.model(x, adj)
        
        return embeddings.cpu().numpy()
    
    def find_related_threats(self, threat_features: np.ndarray,
                            relationships: List[Tuple[int, int]],
                            query_idx: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find threats most related to a query threat
        
        Args:
            threat_features: Feature matrix for threats
            relationships: List of threat relationships
            query_idx: Index of query threat
            top_k: Number of related threats to return
            
        Returns:
            List of (threat_idx, similarity_score) tuples
        """
        embeddings = self.get_embeddings(threat_features, relationships)
        
        # Compute cosine similarity
        query_embedding = embeddings[query_idx]
        query_norm = np.linalg.norm(query_embedding)
        
        similarities = []
        for i, emb in enumerate(embeddings):
            if i != query_idx:
                sim = np.dot(query_embedding, emb) / (query_norm * np.linalg.norm(emb) + 1e-8)
                similarities.append((i, float(sim)))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def detect_threat_communities(self, threat_features: np.ndarray,
                                  relationships: List[Tuple[int, int]],
                                  num_communities: int = 3) -> np.ndarray:
        """
        Detect threat communities using GNN embeddings
        
        Args:
            threat_features: Feature matrix for threats
            relationships: List of threat relationships
            num_communities: Number of communities to detect
            
        Returns:
            Community assignments for each threat
        """
        embeddings = self.get_embeddings(threat_features, relationships)
        
        # Use simple k-means clustering on embeddings
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=num_communities, random_state=42)
        communities = kmeans.fit_predict(embeddings)
        
        return communities
    
    def save_model(self, filepath: str) -> None:
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.layers[0].linear.in_features,
            'hidden_dim': self.model.layers[0].linear.out_features,
            'output_dim': self.model.layers[-1].linear.out_features,
            'num_layers': self.model.num_layers
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model from disk"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
