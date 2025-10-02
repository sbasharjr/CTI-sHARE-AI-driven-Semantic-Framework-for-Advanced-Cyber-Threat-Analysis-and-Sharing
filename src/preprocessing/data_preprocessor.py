"""
Data preprocessing module for cyber threat analysis
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class ThreatDataPreprocessor:
    """
    Preprocesses cyber threat data for machine learning and deep learning models
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
            self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize and lemmatize text
        """
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return tokens
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from raw threat data
        """
        processed_data = data.copy()
        
        # Text features
        if 'summary' in processed_data.columns:
            processed_data['cleaned_summary'] = processed_data['summary'].apply(self.clean_text)
            processed_data['summary_length'] = processed_data['cleaned_summary'].apply(len)
            processed_data['word_count'] = processed_data['cleaned_summary'].apply(
                lambda x: len(x.split()) if x else 0
            )
        
        # Categorical encoding
        categorical_columns = ['event_type', 'motive', 'industry', 'country']
        for col in categorical_columns:
            if col in processed_data.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    processed_data[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        processed_data[col].fillna('unknown')
                    )
                else:
                    processed_data[f'{col}_encoded'] = self.label_encoders[col].transform(
                        processed_data[col].fillna('unknown')
                    )
        
        return processed_data
    
    def normalize_numerical_features(self, data: pd.DataFrame, 
                                     columns: List[str] = None) -> pd.DataFrame:
        """
        Normalize numerical features using StandardScaler
        """
        processed_data = data.copy()
        
        if columns is None:
            columns = processed_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if columns:
            processed_data[columns] = self.scaler.fit_transform(processed_data[columns])
        
        return processed_data
    
    def create_feature_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """
        Create feature matrix for ML models
        """
        feature_columns = [col for col in data.columns if col.endswith('_encoded') or 
                          col in ['summary_length', 'word_count']]
        
        if not feature_columns:
            raise ValueError("No feature columns found. Run extract_features first.")
        
        return data[feature_columns].fillna(0).values
    
    def prepare_text_sequences(self, texts: List[str], max_length: int = 100) -> np.ndarray:
        """
        Prepare text sequences for deep learning models
        """
        tokenized_texts = [self.tokenize_and_lemmatize(text) for text in texts]
        
        # Pad sequences to max_length
        sequences = []
        for tokens in tokenized_texts:
            if len(tokens) > max_length:
                sequences.append(tokens[:max_length])
            else:
                sequences.append(tokens + [''] * (max_length - len(tokens)))
        
        return np.array(sequences)
