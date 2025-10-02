"""
Traditional machine learning models for threat detection
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Tuple, Dict, Any
import joblib
import os


class ThreatDetectionML:
    """
    Machine learning models for cyber threat detection
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize ML model
        
        Args:
            model_type: Type of model ('random_forest', 'svm', 'gradient_boosting', 'naive_bayes')
        """
        self.model_type = model_type
        self.model = self._initialize_model(model_type)
        self.is_trained = False
    
    def _initialize_model(self, model_type: str):
        """
        Initialize the specified model
        """
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'svm':
            return SVC(
                kernel='rbf',
                C=1.0,
                random_state=42,
                probability=True
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif model_type == 'naive_bayes':
            return MultinomialNB(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            validation_split: Validation split ratio
            
        Returns:
            Training metrics
        """
        # Split data for validation
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=validation_split, random_state=42
            )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on training set
        train_score = self.model.score(X_train, y_train)
        
        metrics = {
            'train_accuracy': train_score,
            'model_type': self.model_type
        }
        
        # Evaluate on validation set if available
        if validation_split > 0:
            val_score = self.model.score(X_val, y_val)
            y_pred = self.model.predict(X_val)
            metrics['val_accuracy'] = val_score
            metrics['classification_report'] = classification_report(y_val, y_pred)
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(f"Model {self.model_type} does not support probability prediction")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred)
        }
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation
        
        Args:
            X: Feature matrix
            y: Labels
            cv: Number of cross-validation folds
            
        Returns:
            Cross-validation scores
        """
        scores = cross_val_score(self.model, X, y, cv=cv, n_jobs=-1)
        
        return {
            'scores': scores.tolist(),
            'mean_score': scores.mean(),
            'std_score': scores.std()
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from disk
        
        Args:
            filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
