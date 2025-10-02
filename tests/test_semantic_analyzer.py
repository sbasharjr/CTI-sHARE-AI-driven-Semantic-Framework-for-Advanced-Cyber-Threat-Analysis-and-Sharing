"""
Tests for semantic analyzer
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.semantic_analysis.semantic_analyzer import ThreatSemanticAnalyzer


class TestThreatSemanticAnalyzer:
    
    def setup_method(self):
        self.analyzer = ThreatSemanticAnalyzer()
    
    def test_categorize_threat_malware(self):
        text = "Ransomware attack encrypting files on multiple systems"
        categories = self.analyzer.categorize_threat(text)
        
        assert 'malware' in categories
        assert categories['malware'] > 0
    
    def test_categorize_threat_phishing(self):
        text = "Phishing campaign targeting employees with fake login pages"
        categories = self.analyzer.categorize_threat(text)
        
        assert 'phishing' in categories
        assert categories['phishing'] > 0
    
    def test_extract_ip_addresses(self):
        text = "Suspicious traffic from 192.168.1.100 and 10.0.0.1"
        entities = self.analyzer.extract_threat_entities(text)
        
        assert len(entities['ips']) == 2
        assert '192.168.1.100' in entities['ips']
        assert '10.0.0.1' in entities['ips']
    
    def test_extract_cves(self):
        text = "Critical vulnerability CVE-2024-1234 being exploited"
        entities = self.analyzer.extract_threat_entities(text)
        
        assert len(entities['cves']) == 1
        assert 'CVE-2024-1234' in entities['cves']
    
    def test_assess_severity_critical(self):
        text = "Critical zero-day vulnerability actively exploited by nation state actors"
        severity = self.analyzer.assess_threat_severity(text)
        
        assert severity['severity'] in ['HIGH', 'CRITICAL']
        assert severity['level'] >= 4
    
    def test_assess_severity_low(self):
        text = "Minor configuration issue detected"
        severity = self.analyzer.assess_threat_severity(text)
        
        assert severity['level'] <= 2
    
    def test_compute_threat_similarity(self):
        texts = [
            "Ransomware attack on healthcare",
            "Ransomware targeting hospitals",
            "DDoS attack on financial services"
        ]
        
        similarity_matrix = self.analyzer.compute_threat_similarity(texts)
        
        assert similarity_matrix.shape == (3, 3)
        # First two should be more similar than third
        assert similarity_matrix[0, 1] > similarity_matrix[0, 2]
    
    def test_cluster_similar_threats(self):
        texts = [
            "Ransomware attack",
            "Ransomware incident",
            "DDoS attack",
            "DDoS event"
        ]
        
        clusters = self.analyzer.cluster_similar_threats(texts, threshold=0.5)
        
        assert len(clusters) >= 1
        assert len(clusters) <= 4
    
    def test_generate_threat_summary(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        summary = self.analyzer.generate_threat_summary(text, max_sentences=2)
        
        assert len(summary.split('.')) <= 3  # 2 sentences + empty string
    
    def test_analyze_threat_trends(self):
        threats = [
            {'text': 'Ransomware attack', 'timestamp': '2024-01-01'},
            {'text': 'Phishing campaign', 'timestamp': '2024-01-02'},
            {'text': 'Ransomware incident', 'timestamp': '2024-01-03'}
        ]
        
        trends = self.analyzer.analyze_threat_trends(threats)
        
        assert trends['total_threats'] == 3
        assert trends['unique_categories'] > 0
        assert 'category_distribution' in trends


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
