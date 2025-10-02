"""
Deep learning models for threat detection
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Tuple, Dict, Any, List
import os


class ThreatDetectionLSTM:
    """
    LSTM-based deep learning model for threat detection from sequential data
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128, 
                 max_length: int = 100, num_classes: int = 5):
        """
        Initialize LSTM model
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            max_length: Maximum sequence length
            num_classes: Number of threat classes
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.num_classes = num_classes
        self.model = self._build_model()
        self.is_trained = False
    
    def _build_model(self) -> keras.Model:
        """
        Build LSTM model architecture
        """
        model = models.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              validation_data: Tuple[np.ndarray, np.ndarray] = None,
              epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the LSTM model
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            validation_data: Validation data tuple (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        return {
            'history': history.history,
            'final_train_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1] if validation_data else None
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input sequences
            
        Returns:
            Predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Input sequences
            
        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        self.is_trained = True


class ThreatDetectionCNN:
    """
    CNN-based deep learning model for threat detection
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128,
                 max_length: int = 100, num_classes: int = 5):
        """
        Initialize CNN model
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            max_length: Maximum sequence length
            num_classes: Number of threat classes
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.num_classes = num_classes
        self.model = self._build_model()
        self.is_trained = False
    
    def _build_model(self) -> keras.Model:
        """
        Build CNN model architecture
        """
        model = models.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            layers.Conv1D(128, 5, activation='relu'),
            layers.MaxPooling1D(5),
            layers.Conv1D(128, 5, activation='relu'),
            layers.MaxPooling1D(5),
            layers.Conv1D(128, 5, activation='relu'),
            layers.GlobalMaxPooling1D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              validation_data: Tuple[np.ndarray, np.ndarray] = None,
              epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the CNN model
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            validation_data: Validation data tuple (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        return {
            'history': history.history,
            'final_train_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1] if validation_data else None
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input sequences
            
        Returns:
            Predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Input sequences
            
        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
