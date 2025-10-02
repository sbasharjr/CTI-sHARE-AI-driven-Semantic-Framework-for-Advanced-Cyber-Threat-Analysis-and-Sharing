"""
Data loading utilities
"""

import pandas as pd
import json
import os
from typing import Dict, Any, List, Optional


class ThreatDataLoader:
    """
    Load and manage threat intelligence data
    """
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load threat data from CSV file
        
        Args:
            filename: CSV filename
            
        Returns:
            DataFrame with threat data
        """
        filepath = os.path.join(self.data_dir, filename)
        return pd.read_csv(filepath)
    
    def load_json(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load threat data from JSON file
        
        Args:
            filename: JSON filename
            
        Returns:
            List of threat dictionaries
        """
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def save_csv(self, data: pd.DataFrame, filename: str) -> None:
        """
        Save data to CSV file
        
        Args:
            data: DataFrame to save
            filename: Output filename
        """
        filepath = os.path.join(self.data_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data.to_csv(filepath, index=False)
    
    def save_json(self, data: List[Dict[str, Any]], filename: str) -> None:
        """
        Save data to JSON file
        
        Args:
            data: List of dictionaries to save
            filename: Output filename
        """
        filepath = os.path.join(self.data_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_sample_data(self, num_samples: int = 100) -> pd.DataFrame:
        """
        Create sample threat data for testing
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            DataFrame with sample data
        """
        import random
        from datetime import datetime, timedelta
        
        event_types = ['ransomware', 'phishing', 'ddos', 'data_breach', 'malware']
        countries = ['USA', 'China', 'Russia', 'UK', 'Germany', 'France']
        industries = ['finance', 'healthcare', 'technology', 'retail', 'government']
        motives = ['financial', 'espionage', 'hacktivism', 'ransomware']
        
        sample_texts = [
            "Ransomware attack targeting healthcare organizations with encryption malware",
            "Phishing campaign using fake login pages to steal credentials",
            "DDoS attack disrupting financial services infrastructure",
            "Data breach exposing customer personal information",
            "Advanced persistent threat group targeting government networks",
            "Malware spreading through compromised software updates",
            "Zero-day vulnerability being actively exploited in the wild",
            "Insider threat leading to data exfiltration",
            "Supply chain attack compromising third-party vendors",
            "SQL injection vulnerability allowing unauthorized access"
        ]
        
        data = []
        base_date = datetime.now()
        
        for i in range(num_samples):
            data.append({
                'id': i + 1,
                'event_date': (base_date - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d'),
                'country': random.choice(countries),
                'event_type': random.choice(event_types),
                'motive': random.choice(motives),
                'industry': random.choice(industries),
                'summary': random.choice(sample_texts),
                'severity': random.randint(1, 5)
            })
        
        return pd.DataFrame(data)
