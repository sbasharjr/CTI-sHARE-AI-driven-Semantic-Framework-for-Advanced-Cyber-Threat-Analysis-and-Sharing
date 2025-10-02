"""
RESTful API for threat intelligence sharing
"""

from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
from typing import Dict, Any, List
import json
from datetime import datetime
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThreatSharingAPI:
    """
    API server for threat intelligence sharing
    """
    
    def __init__(self, detector=None, semantic_analyzer=None):
        """
        Initialize threat sharing API
        
        Args:
            detector: Real-time threat detector instance
            semantic_analyzer: Semantic analyzer instance
        """
        self.app = Flask(__name__)
        CORS(self.app)
        self.api = Api(self.app)
        self.detector = detector
        self.semantic_analyzer = semantic_analyzer
        self.threat_database = []
        
        self._register_routes()
    
    def _register_routes(self):
        """
        Register API routes
        """
        self.api.add_resource(
            HealthCheck,
            '/api/health',
            resource_class_kwargs={'api': self}
        )
        self.api.add_resource(
            SubmitThreat,
            '/api/threats/submit',
            resource_class_kwargs={'api': self}
        )
        self.api.add_resource(
            GetThreats,
            '/api/threats',
            resource_class_kwargs={'api': self}
        )
        self.api.add_resource(
            AnalyzeThreat,
            '/api/threats/analyze',
            resource_class_kwargs={'api': self}
        )
        self.api.add_resource(
            GetStatistics,
            '/api/statistics',
            resource_class_kwargs={'api': self}
        )
        self.api.add_resource(
            SearchThreats,
            '/api/threats/search',
            resource_class_kwargs={'api': self}
        )
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """
        Run API server
        
        Args:
            host: Host address
            port: Port number
            debug: Debug mode
        """
        logger.info(f"Starting Threat Sharing API on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


class HealthCheck(Resource):
    """
    Health check endpoint
    """
    
    def __init__(self, api: ThreatSharingAPI):
        self.api = api
    
    def get(self):
        """
        GET /api/health
        """
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }, 200


class SubmitThreat(Resource):
    """
    Submit threat intelligence endpoint
    """
    
    def __init__(self, api: ThreatSharingAPI):
        self.api = api
    
    def post(self):
        """
        POST /api/threats/submit
        
        Body:
            {
                "text": "Threat description",
                "source": "source identifier",
                "metadata": {...}
            }
        """
        try:
            data = request.get_json()
            
            if not data or 'text' not in data:
                return {'error': 'Missing required field: text'}, 400
            
            threat = {
                'id': len(self.api.threat_database) + 1,
                'text': data['text'],
                'source': data.get('source', 'unknown'),
                'metadata': data.get('metadata', {}),
                'timestamp': datetime.now().isoformat(),
                'status': 'submitted'
            }
            
            # Analyze threat if semantic analyzer is available
            if self.api.semantic_analyzer:
                threat['analysis'] = {
                    'categories': self.api.semantic_analyzer.categorize_threat(data['text']),
                    'entities': self.api.semantic_analyzer.extract_threat_entities(data['text']),
                    'severity': self.api.semantic_analyzer.assess_threat_severity(data['text'])
                }
            
            # Add to real-time detector if available
            if self.api.detector:
                self.api.detector.add_threat_data({'text': data['text']})
            
            self.api.threat_database.append(threat)
            
            return {
                'success': True,
                'threat_id': threat['id'],
                'threat': threat
            }, 201
            
        except Exception as e:
            logger.error(f"Error submitting threat: {e}")
            return {'error': str(e)}, 500


class GetThreats(Resource):
    """
    Get threats endpoint
    """
    
    def __init__(self, api: ThreatSharingAPI):
        self.api = api
    
    def get(self):
        """
        GET /api/threats?limit=10&offset=0
        """
        try:
            limit = int(request.args.get('limit', 10))
            offset = int(request.args.get('offset', 0))
            
            threats = self.api.threat_database[offset:offset + limit]
            
            return {
                'success': True,
                'total': len(self.api.threat_database),
                'limit': limit,
                'offset': offset,
                'threats': threats
            }, 200
            
        except Exception as e:
            logger.error(f"Error getting threats: {e}")
            return {'error': str(e)}, 500


class AnalyzeThreat(Resource):
    """
    Analyze threat endpoint
    """
    
    def __init__(self, api: ThreatSharingAPI):
        self.api = api
    
    def post(self):
        """
        POST /api/threats/analyze
        
        Body:
            {
                "text": "Threat description to analyze"
            }
        """
        try:
            data = request.get_json()
            
            if not data or 'text' not in data:
                return {'error': 'Missing required field: text'}, 400
            
            if not self.api.semantic_analyzer:
                return {'error': 'Semantic analyzer not available'}, 503
            
            analysis = {
                'categories': self.api.semantic_analyzer.categorize_threat(data['text']),
                'entities': self.api.semantic_analyzer.extract_threat_entities(data['text']),
                'severity': self.api.semantic_analyzer.assess_threat_severity(data['text']),
                'summary': self.api.semantic_analyzer.generate_threat_summary(data['text'])
            }
            
            return {
                'success': True,
                'analysis': analysis
            }, 200
            
        except Exception as e:
            logger.error(f"Error analyzing threat: {e}")
            return {'error': str(e)}, 500


class GetStatistics(Resource):
    """
    Get statistics endpoint
    """
    
    def __init__(self, api: ThreatSharingAPI):
        self.api = api
    
    def get(self):
        """
        GET /api/statistics
        """
        try:
            stats = {
                'total_threats': len(self.api.threat_database),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add detector statistics if available
            if self.api.detector:
                stats['detector'] = self.api.detector.get_statistics()
            
            # Calculate category distribution
            if self.api.semantic_analyzer and self.api.threat_database:
                category_counts = {}
                for threat in self.api.threat_database:
                    if 'analysis' in threat and 'categories' in threat['analysis']:
                        top_cat = max(
                            threat['analysis']['categories'].items(),
                            key=lambda x: x[1]
                        )[0]
                        category_counts[top_cat] = category_counts.get(top_cat, 0) + 1
                
                stats['category_distribution'] = category_counts
            
            return {
                'success': True,
                'statistics': stats
            }, 200
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}, 500


class SearchThreats(Resource):
    """
    Search threats endpoint
    """
    
    def __init__(self, api: ThreatSharingAPI):
        self.api = api
    
    def get(self):
        """
        GET /api/threats/search?q=keyword&category=malware
        """
        try:
            query = request.args.get('q', '').lower()
            category = request.args.get('category', '').lower()
            
            filtered_threats = []
            
            for threat in self.api.threat_database:
                match = True
                
                # Text search
                if query and query not in threat['text'].lower():
                    match = False
                
                # Category filter
                if category and 'analysis' in threat:
                    categories = threat['analysis'].get('categories', {})
                    if category not in categories or categories[category] < 0.1:
                        match = False
                
                if match:
                    filtered_threats.append(threat)
            
            return {
                'success': True,
                'total': len(filtered_threats),
                'threats': filtered_threats
            }, 200
            
        except Exception as e:
            logger.error(f"Error searching threats: {e}")
            return {'error': str(e)}, 500
