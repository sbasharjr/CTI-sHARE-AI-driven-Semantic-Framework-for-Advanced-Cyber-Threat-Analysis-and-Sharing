"""
Web-based Dashboard for Threat Visualization
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from typing import Dict, Any, List
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ThreatDashboard:
    """
    Web-based dashboard for threat intelligence visualization
    """
    
    def __init__(self, threat_detector=None, semantic_analyzer=None):
        """
        Initialize threat dashboard
        
        Args:
            threat_detector: Real-time threat detector instance
            semantic_analyzer: Semantic analyzer instance
        """
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        CORS(self.app)
        
        self.threat_detector = threat_detector
        self.semantic_analyzer = semantic_analyzer
        self.threat_history = []
        
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/dashboard/stats')
        def get_stats():
            """Get dashboard statistics"""
            stats = self._get_dashboard_stats()
            return jsonify(stats)
        
        @self.app.route('/api/dashboard/threats/recent')
        def get_recent_threats():
            """Get recent threats"""
            limit = request.args.get('limit', 10, type=int)
            threats = self._get_recent_threats(limit)
            return jsonify({'threats': threats})
        
        @self.app.route('/api/dashboard/threats/timeline')
        def get_threat_timeline():
            """Get threat timeline data"""
            hours = request.args.get('hours', 24, type=int)
            timeline = self._get_threat_timeline(hours)
            return jsonify({'timeline': timeline})
        
        @self.app.route('/api/dashboard/threats/categories')
        def get_threat_categories():
            """Get threat distribution by category"""
            categories = self._get_threat_by_category()
            return jsonify({'categories': categories})
        
        @self.app.route('/api/dashboard/threats/severity')
        def get_threat_severity():
            """Get threat distribution by severity"""
            severity = self._get_threat_by_severity()
            return jsonify({'severity': severity})
        
        @self.app.route('/api/dashboard/threats/geo')
        def get_threat_geo():
            """Get geographic threat distribution"""
            geo_data = self._get_threat_geo_distribution()
            return jsonify({'geo': geo_data})
        
        @self.app.route('/api/dashboard/health')
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'detector_running': self.threat_detector.is_running if self.threat_detector else False
            })
    
    def _get_dashboard_stats(self) -> Dict[str, Any]:
        """Get overall dashboard statistics"""
        total_threats = len(self.threat_history)
        
        if total_threats == 0:
            return {
                'total_threats': 0,
                'critical_threats': 0,
                'threats_today': 0,
                'detection_rate': 0,
                'top_category': 'N/A'
            }
        
        # Count threats by severity
        critical_count = sum(1 for t in self.threat_history 
                            if t.get('severity') == 'CRITICAL')
        
        # Count threats today
        today = datetime.now().date()
        threats_today = sum(1 for t in self.threat_history 
                          if datetime.fromisoformat(t.get('timestamp', '')).date() == today)
        
        # Get most common category
        categories = {}
        for t in self.threat_history:
            cat = t.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        top_category = max(categories.items(), key=lambda x: x[1])[0] if categories else 'N/A'
        
        # Detection rate (percentage of confirmed threats)
        confirmed_threats = sum(1 for t in self.threat_history if t.get('is_threat', False))
        detection_rate = (confirmed_threats / total_threats * 100) if total_threats > 0 else 0
        
        return {
            'total_threats': total_threats,
            'critical_threats': critical_count,
            'threats_today': threats_today,
            'detection_rate': round(detection_rate, 2),
            'top_category': top_category
        }
    
    def _get_recent_threats(self, limit: int) -> List[Dict[str, Any]]:
        """Get recent threats"""
        return self.threat_history[-limit:]
    
    def _get_threat_timeline(self, hours: int) -> List[Dict[str, Any]]:
        """Get threat timeline for last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Group threats by hour
        timeline = {}
        for threat in self.threat_history:
            timestamp = datetime.fromisoformat(threat.get('timestamp', ''))
            if timestamp >= cutoff:
                hour_key = timestamp.strftime('%Y-%m-%d %H:00')
                timeline[hour_key] = timeline.get(hour_key, 0) + 1
        
        # Convert to list format
        result = [{'timestamp': k, 'count': v} for k, v in sorted(timeline.items())]
        return result
    
    def _get_threat_by_category(self) -> Dict[str, int]:
        """Get threat distribution by category"""
        categories = {}
        for threat in self.threat_history:
            cat = threat.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        return categories
    
    def _get_threat_by_severity(self) -> Dict[str, int]:
        """Get threat distribution by severity"""
        severity = {}
        for threat in self.threat_history:
            sev = threat.get('severity', 'MEDIUM')
            severity[sev] = severity.get(sev, 0) + 1
        return severity
    
    def _get_threat_geo_distribution(self) -> List[Dict[str, Any]]:
        """Get geographic threat distribution"""
        # Simplified - would use IP geolocation in real implementation
        countries = {}
        for threat in self.threat_history:
            # Extract IPs and map to countries (placeholder logic)
            ips = threat.get('entities', {}).get('ips', [])
            for ip in ips:
                country = 'Unknown'  # Would use geolocation service
                countries[country] = countries.get(country, 0) + 1
        
        return [{'country': k, 'count': v} for k, v in countries.items()]
    
    def add_threat(self, threat_data: Dict[str, Any]) -> None:
        """
        Add threat to dashboard history
        
        Args:
            threat_data: Threat information
        """
        self.threat_history.append(threat_data)
        
        # Keep only last 10000 threats
        if len(self.threat_history) > 10000:
            self.threat_history = self.threat_history[-10000:]
    
    def run(self, host: str = '0.0.0.0', port: int = 5001, debug: bool = False):
        """
        Run dashboard server
        
        Args:
            host: Host to bind to
            port: Port to bind to
            debug: Enable debug mode
        """
        logger.info(f"Starting threat dashboard on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)
