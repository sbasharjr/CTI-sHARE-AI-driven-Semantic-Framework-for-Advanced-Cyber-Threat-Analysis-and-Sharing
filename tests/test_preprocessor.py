"""
Tests for data preprocessor
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import pandas as pd
import numpy as np
from src.preprocessing.data_preprocessor import ThreatDataPreprocessor


class TestThreatDataPreprocessor:
    
    def setup_method(self):
        self.preprocessor = ThreatDataPreprocessor()
    
    def test_clean_text(self):
        text = "Check out https://malicious.com for more info! 123"
        cleaned = self.preprocessor.clean_text(text)
        
        assert 'https' not in cleaned
        assert 'malicious.com' not in cleaned
        assert '123' not in cleaned
        assert len(cleaned) > 0
    
    def test_clean_text_empty(self):
        text = None
        cleaned = self.preprocessor.clean_text(text)
        
        assert cleaned == ""
    
    def test_tokenize_and_lemmatize(self):
        text = "The attackers are exploiting vulnerabilities"
        tokens = self.preprocessor.tokenize_and_lemmatize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # Stopwords should be removed
        assert 'the' not in [t.lower() for t in tokens]
    
    def test_extract_features(self):
        data = pd.DataFrame({
            'summary': ['Ransomware attack on hospital', 'Phishing campaign'],
            'event_type': ['malware', 'phishing'],
            'country': ['USA', 'UK']
        })
        
        processed = self.preprocessor.extract_features(data)
        
        assert 'cleaned_summary' in processed.columns
        assert 'summary_length' in processed.columns
        assert 'word_count' in processed.columns
        assert 'event_type_encoded' in processed.columns
        assert 'country_encoded' in processed.columns
    
    def test_normalize_numerical_features(self):
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        normalized = self.preprocessor.normalize_numerical_features(data)
        
        # Check that mean is close to 0 and std is close to 1
        assert abs(normalized['feature1'].mean()) < 1e-10
        assert abs(normalized['feature1'].std() - 1.0) < 0.1
    
    def test_create_feature_matrix(self):
        data = pd.DataFrame({
            'event_type_encoded': [0, 1, 2],
            'country_encoded': [0, 1, 0],
            'summary_length': [100, 200, 150],
            'word_count': [20, 40, 30]
        })
        
        feature_matrix = self.preprocessor.create_feature_matrix(data)
        
        assert isinstance(feature_matrix, np.ndarray)
        assert feature_matrix.shape[0] == 3
        assert feature_matrix.shape[1] == 4
    
    def test_prepare_text_sequences(self):
        texts = [
            "Short text",
            "A much longer text with many more words",
            "Medium length text here"
        ]
        
        sequences = self.preprocessor.prepare_text_sequences(texts, max_length=10)
        
        assert sequences.shape[0] == 3
        assert sequences.shape[1] == 10


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
