"""
Semantic analysis framework for cyber threat intelligence
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re


class ThreatSemanticAnalyzer:
    """
    Semantic analyzer for cyber threat intelligence
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
        self.threat_taxonomy = self._initialize_threat_taxonomy()
        self.is_fitted = False
    
    def _initialize_threat_taxonomy(self) -> Dict[str, List[str]]:
        """
        Initialize threat taxonomy with common threat categories and keywords
        """
        return {
            'malware': [
                'ransomware', 'trojan', 'virus', 'worm', 'spyware', 'adware',
                'rootkit', 'backdoor', 'keylogger', 'botnet', 'cryptominer'
            ],
            'phishing': [
                'phishing', 'spear phishing', 'whaling', 'vishing', 'smishing',
                'credential harvesting', 'social engineering', 'impersonation'
            ],
            'ddos': [
                'ddos', 'dos', 'distributed denial', 'flood attack', 'amplification',
                'syn flood', 'udp flood', 'http flood'
            ],
            'data_breach': [
                'data breach', 'data leak', 'data exfiltration', 'unauthorized access',
                'data theft', 'database compromise', 'information disclosure'
            ],
            'apt': [
                'apt', 'advanced persistent threat', 'nation state', 'espionage',
                'cyber warfare', 'state sponsored', 'targeted attack'
            ],
            'vulnerability_exploit': [
                'exploit', 'zero day', 'vulnerability', 'cve', 'buffer overflow',
                'sql injection', 'xss', 'remote code execution', 'privilege escalation'
            ],
            'insider_threat': [
                'insider threat', 'insider attack', 'malicious insider', 'privileged abuse',
                'sabotage', 'data theft by employee'
            ],
            'supply_chain': [
                'supply chain', 'third party', 'vendor compromise', 'software supply chain',
                'compromised update', 'dependency attack'
            ]
        }
    
    def categorize_threat(self, text: str) -> Dict[str, float]:
        """
        Categorize threat based on semantic analysis
        
        Args:
            text: Threat description text
            
        Returns:
            Dictionary of threat categories with confidence scores
        """
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in self.threat_taxonomy.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text_lower:
                    # Weight by keyword specificity (inverse of frequency)
                    score += 1.0 / (len(keywords) ** 0.5)
            
            scores[category] = min(score, 1.0)
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}
        
        return scores
    
    def extract_threat_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract threat-related entities from text
        
        Args:
            text: Threat description text
            
        Returns:
            Dictionary of extracted entities by type
        """
        entities = {
            'ips': [],
            'domains': [],
            'hashes': [],
            'emails': [],
            'cves': []
        }
        
        # Extract IP addresses
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        entities['ips'] = re.findall(ip_pattern, text)
        
        # Extract domains
        domain_pattern = r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b'
        entities['domains'] = re.findall(domain_pattern, text)
        
        # Extract file hashes (MD5, SHA1, SHA256)
        hash_patterns = [
            r'\b[a-fA-F0-9]{32}\b',  # MD5
            r'\b[a-fA-F0-9]{40}\b',  # SHA1
            r'\b[a-fA-F0-9]{64}\b'   # SHA256
        ]
        for pattern in hash_patterns:
            entities['hashes'].extend(re.findall(pattern, text))
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['emails'] = re.findall(email_pattern, text)
        
        # Extract CVEs
        cve_pattern = r'CVE-\d{4}-\d{4,7}'
        entities['cves'] = re.findall(cve_pattern, text, re.IGNORECASE)
        
        return entities
    
    def compute_threat_similarity(self, texts: List[str]) -> np.ndarray:
        """
        Compute semantic similarity between threat descriptions
        
        Args:
            texts: List of threat description texts
            
        Returns:
            Similarity matrix
        """
        if not self.is_fitted:
            vectors = self.vectorizer.fit_transform(texts)
            self.is_fitted = True
        else:
            vectors = self.vectorizer.transform(texts)
        
        similarity_matrix = cosine_similarity(vectors)
        return similarity_matrix
    
    def cluster_similar_threats(self, texts: List[str], 
                               threshold: float = 0.7) -> List[List[int]]:
        """
        Cluster similar threats based on semantic similarity
        
        Args:
            texts: List of threat description texts
            threshold: Similarity threshold for clustering
            
        Returns:
            List of clusters (each cluster is a list of indices)
        """
        similarity_matrix = self.compute_threat_similarity(texts)
        
        clusters = []
        visited = set()
        
        for i in range(len(texts)):
            if i in visited:
                continue
            
            cluster = [i]
            visited.add(i)
            
            for j in range(i + 1, len(texts)):
                if j not in visited and similarity_matrix[i, j] >= threshold:
                    cluster.append(j)
                    visited.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def analyze_threat_trends(self, threats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze trends in threat data
        
        Args:
            threats: List of threat dictionaries with 'text' and 'timestamp' keys
            
        Returns:
            Trend analysis results
        """
        trend_data = defaultdict(int)
        
        for threat in threats:
            categories = self.categorize_threat(threat.get('text', ''))
            top_category = max(categories.items(), key=lambda x: x[1])[0] if categories else 'unknown'
            trend_data[top_category] += 1
        
        return {
            'category_distribution': dict(trend_data),
            'total_threats': len(threats),
            'unique_categories': len(trend_data)
        }
    
    def generate_threat_summary(self, text: str, max_sentences: int = 3) -> str:
        """
        Generate a concise summary of threat description
        
        Args:
            text: Threat description text
            max_sentences: Maximum number of sentences in summary
            
        Returns:
            Threat summary
        """
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Simple extractive summarization
        # In production, use more sophisticated methods
        return '. '.join(sentences[:max_sentences]) + '.'
    
    def assess_threat_severity(self, text: str) -> Dict[str, Any]:
        """
        Assess threat severity based on semantic indicators
        
        Args:
            text: Threat description text
            
        Returns:
            Severity assessment
        """
        text_lower = text.lower()
        
        # Severity indicators
        high_severity_keywords = [
            'critical', 'severe', 'widespread', 'zero day', 'actively exploited',
            'ransomware', 'data breach', 'nation state', 'apt'
        ]
        
        medium_severity_keywords = [
            'vulnerability', 'exploit', 'malware', 'phishing', 'ddos'
        ]
        
        high_score = sum(1 for kw in high_severity_keywords if kw in text_lower)
        medium_score = sum(1 for kw in medium_severity_keywords if kw in text_lower)
        
        total_score = high_score * 2 + medium_score
        
        if total_score >= 4:
            severity = 'CRITICAL'
            level = 5
        elif total_score >= 3:
            severity = 'HIGH'
            level = 4
        elif total_score >= 2:
            severity = 'MEDIUM'
            level = 3
        elif total_score >= 1:
            severity = 'LOW'
            level = 2
        else:
            severity = 'INFORMATIONAL'
            level = 1
        
        return {
            'severity': severity,
            'level': level,
            'score': total_score,
            'indicators': {
                'high_severity_matches': high_score,
                'medium_severity_matches': medium_score
            }
        }
