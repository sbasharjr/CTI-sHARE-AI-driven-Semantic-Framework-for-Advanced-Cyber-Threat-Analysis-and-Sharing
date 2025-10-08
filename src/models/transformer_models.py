"""
Transformer-based models (BERT, GPT) for advanced threat detection
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ThreatDetectionBERT:
    """
    BERT-based model for threat detection and classification
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', num_classes: int = 5):
        """
        Initialize BERT model for threat detection
        
        Args:
            model_name: Pre-trained BERT model name
            num_classes: Number of threat categories
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained BERT
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.bert.to(self.device)
        self.classifier.to(self.device)
        self.is_trained = False
        
    def encode_text(self, texts: List[str], max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        Encode text using BERT tokenizer
        
        Args:
            texts: List of threat descriptions
            max_length: Maximum sequence length
            
        Returns:
            Encoded inputs for BERT
        """
        encoding = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return encoding
    
    def train(self, train_texts: List[str], train_labels: np.ndarray,
              val_texts: List[str] = None, val_labels: np.ndarray = None,
              epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5) -> Dict[str, Any]:
        """
        Train BERT model for threat classification
        
        Args:
            train_texts: Training threat descriptions
            train_labels: Training labels
            val_texts: Validation threat descriptions
            val_labels: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        optimizer = torch.optim.AdamW(
            list(self.bert.parameters()) + list(self.classifier.parameters()),
            lr=learning_rate
        )
        criterion = nn.CrossEntropyLoss()
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training phase
            self.bert.train()
            self.classifier.train()
            
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for i in range(0, len(train_texts), batch_size):
                batch_texts = train_texts[i:i+batch_size]
                batch_labels = torch.tensor(train_labels[i:i+batch_size]).to(self.device)
                
                # Encode texts
                encoding = self.encode_text(batch_texts)
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
                logits = self.classifier(pooled_output)
                
                loss = criterion(logits, batch_labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            train_acc = train_correct / train_total
            history['train_loss'].append(train_loss / (len(train_texts) / batch_size))
            history['train_acc'].append(train_acc)
            
            # Validation phase
            if val_texts and val_labels is not None:
                val_loss, val_acc = self._validate(val_texts, val_labels, batch_size, criterion)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, "
                          f"Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        self.is_trained = True
        return history
    
    def _validate(self, val_texts: List[str], val_labels: np.ndarray,
                  batch_size: int, criterion) -> Tuple[float, float]:
        """Validate model performance"""
        self.bert.eval()
        self.classifier.eval()
        
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for i in range(0, len(val_texts), batch_size):
                batch_texts = val_texts[i:i+batch_size]
                batch_labels = torch.tensor(val_labels[i:i+batch_size]).to(self.device)
                
                encoding = self.encode_text(batch_texts)
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
                logits = self.classifier(pooled_output)
                
                loss = criterion(logits, batch_labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        return val_loss / (len(val_texts) / batch_size), val_correct / val_total
    
    def predict(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Predict threat categories
        
        Args:
            texts: Threat descriptions
            batch_size: Batch size
            
        Returns:
            Predicted class labels
        """
        self.bert.eval()
        self.classifier.eval()
        
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                encoding = self.encode_text(batch_texts)
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
                logits = self.classifier(pooled_output)
                
                _, predicted = torch.max(logits, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Predict threat category probabilities
        
        Args:
            texts: Threat descriptions
            batch_size: Batch size
            
        Returns:
            Predicted probabilities for each class
        """
        self.bert.eval()
        self.classifier.eval()
        
        probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                encoding = self.encode_text(batch_texts)
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
                logits = self.classifier(pooled_output)
                
                probs = torch.softmax(logits, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def save_model(self, filepath: str) -> None:
        """Save model to disk"""
        torch.save({
            'bert_state_dict': self.bert.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'num_classes': self.num_classes,
            'model_name': self.model_name
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model from disk"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.bert.load_state_dict(checkpoint['bert_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


class ThreatDetectionGPT:
    """
    GPT-based model for threat analysis and text generation
    """
    
    def __init__(self, model_name: str = 'gpt2'):
        """
        Initialize GPT model for threat analysis
        
        Args:
            model_name: Pre-trained GPT model name
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained GPT-2
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2Model.from_pretrained(model_name)
        self.model.to(self.device)
        
    def encode_text(self, texts: List[str], max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        Encode text using GPT tokenizer
        
        Args:
            texts: List of threat descriptions
            max_length: Maximum sequence length
            
        Returns:
            Encoded inputs for GPT
        """
        encoding = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return encoding
    
    def extract_features(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Extract features from text using GPT
        
        Args:
            texts: Threat descriptions
            batch_size: Batch size
            
        Returns:
            Feature vectors
        """
        self.model.eval()
        features = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                encoding = self.encode_text(batch_texts)
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Use mean pooling of last hidden states
                last_hidden = outputs.last_hidden_state
                pooled = torch.mean(last_hidden, dim=1)
                features.extend(pooled.cpu().numpy())
        
        return np.array(features)
    
    def generate_threat_report(self, prompt: str, max_length: int = 200) -> str:
        """
        Generate threat report using GPT
        
        Args:
            prompt: Initial threat description
            max_length: Maximum generation length
            
        Returns:
            Generated threat analysis report
        """
        self.model.eval()
        
        encoding = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                encoding,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
