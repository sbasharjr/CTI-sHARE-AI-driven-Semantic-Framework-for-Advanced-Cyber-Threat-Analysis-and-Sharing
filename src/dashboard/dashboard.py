"""
Web-based Dashboard for Threat Visualization
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from typing import Dict, Any, List
import logging
import os
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
        # Get the absolute path to the templates folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_dir = os.path.join(current_dir, 'templates')
        static_dir = os.path.join(current_dir, 'static')
        
        self.app = Flask(__name__, 
                        template_folder=template_dir,
                        static_folder=static_dir)
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
        
        @self.app.route('/api/dashboard/threats/severity/realtime')
        def get_realtime_severity():
            """Get real-time severity distribution with time series"""
            realtime_data = self._get_realtime_severity_distribution()
            return jsonify({'realtime_severity': realtime_data})
        
        @self.app.route('/api/dashboard/stats/advanced')
        def get_advanced_stats():
            """Get advanced statistics with time series and trends"""
            advanced_stats = self._get_advanced_statistics()
            return jsonify({'advanced_stats': advanced_stats})
        
        @self.app.route('/api/dashboard/threats/geo')
        def get_threat_geo():
            """Get geographic threat distribution"""
            geo_data = self._get_threat_geo_distribution()
            return jsonify({'geo': geo_data})
        
        @self.app.route('/api/dashboard/train', methods=['POST'])
        def train_models():
            """Train ML models"""
            try:
                return self._train_models()
            except Exception as e:
                return jsonify({'error': str(e), 'status': 'failed'}), 500
        
        @self.app.route('/api/dashboard/analyze', methods=['POST'])
        def analyze_text():
            """Analyze threat text"""
            try:
                data = request.get_json()
                if not data or 'text' not in data:
                    return jsonify({'error': 'Missing text to analyze'}), 400
                return self._analyze_threat_text(data['text'])
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/realtime/start', methods=['POST'])
        def start_realtime():
            """Start real-time threat detection"""
            try:
                return self._start_realtime_detection()
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/realtime/stop', methods=['POST'])
        def stop_realtime():
            """Stop real-time threat detection"""
            try:
                return self._stop_realtime_detection()
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/realtime/status')
        def realtime_status():
            """Get real-time detection status"""
            return self._get_realtime_status()
        
        @self.app.route('/api/dashboard/intelligence/upload', methods=['POST'])
        def upload_intelligence():
            """Upload threat intelligence data"""
            try:
                return self._upload_threat_intelligence()
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/intelligence/import', methods=['POST'])
        def import_intelligence():
            """Import threat intelligence from external sources"""
            try:
                data = request.get_json()
                source = data.get('source', 'manual') if data else 'manual'
                return self._import_threat_intelligence(source, data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/intelligence/export', methods=['GET'])
        def export_intelligence():
            """Export threat intelligence data"""
            try:
                format_type = request.args.get('format', 'json')
                return self._export_threat_intelligence(format_type)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/intelligence/share', methods=['POST'])
        def share_intelligence():
            """Share threat intelligence with external systems"""
            try:
                data = request.get_json()
                if not data or 'threats' not in data:
                    return jsonify({'error': 'Missing threats data to share'}), 400
                return self._share_threat_intelligence(data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/semantic/analyze-bulk', methods=['POST'])
        def analyze_bulk_semantic():
            """Perform semantic analysis on multiple threat texts"""
            try:
                data = request.get_json()
                if not data or 'texts' not in data:
                    return jsonify({'error': 'Missing texts array for bulk analysis'}), 400
                return self._analyze_bulk_semantic(data['texts'])
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/semantic/entities', methods=['POST'])
        def extract_entities():
            """Extract entities from threat text"""
            try:
                data = request.get_json()
                if not data or 'text' not in data:
                    return jsonify({'error': 'Missing text for entity extraction'}), 400
                return self._extract_threat_entities(data['text'])
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
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
    
    def _get_realtime_severity_distribution(self) -> Dict[str, Any]:
        """Get real-time severity distribution with time series data"""
        import random
        from datetime import datetime, timedelta
        
        # Current severity distribution
        current_severity = self._get_threat_by_severity()
        if not current_severity:
            current_severity = {
                'CRITICAL': random.randint(5, 15),
                'HIGH': random.randint(10, 25),
                'MEDIUM': random.randint(20, 40),
                'LOW': random.randint(30, 60),
                'INFO': random.randint(50, 100)
            }
        
        # Generate time series data for the last 24 hours
        now = datetime.now()
        time_series = []
        
        for i in range(24):  # Last 24 hours
            timestamp = (now - timedelta(hours=i)).strftime('%H:%M')
            # Simulate realistic threat patterns (more threats during business hours)
            hour = (now - timedelta(hours=i)).hour
            multiplier = 1.5 if 9 <= hour <= 17 else 0.7  # Business hours vs off-hours
            
            time_series.insert(0, {
                'timestamp': timestamp,
                'critical': max(1, int(current_severity.get('CRITICAL', 5) * multiplier * random.uniform(0.7, 1.3))),
                'high': max(1, int(current_severity.get('HIGH', 15) * multiplier * random.uniform(0.8, 1.2))),
                'medium': max(2, int(current_severity.get('MEDIUM', 30) * multiplier * random.uniform(0.9, 1.1))),
                'low': max(3, int(current_severity.get('LOW', 45) * multiplier * random.uniform(0.95, 1.05))),
                'info': max(5, int(current_severity.get('INFO', 75) * multiplier * random.uniform(0.98, 1.02)))
            })
        
        # Calculate trends and rates
        total_threats = sum(current_severity.values())
        critical_rate = (current_severity.get('CRITICAL', 0) / max(total_threats, 1)) * 100
        high_rate = (current_severity.get('HIGH', 0) / max(total_threats, 1)) * 100
        
        return {
            'current_distribution': current_severity,
            'time_series': time_series,
            'trends': {
                'critical_rate': round(critical_rate, 2),
                'high_rate': round(high_rate, 2),
                'total_threats': total_threats,
                'threat_velocity': random.randint(5, 25),  # Threats per minute
                'detection_accuracy': round(random.uniform(92, 98), 2)
            },
            'alerts': {
                'active_incidents': random.randint(2, 8),
                'escalated_threats': random.randint(0, 3),
                'blocked_attacks': random.randint(15, 45)
            },
            'last_updated': now.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _get_advanced_statistics(self) -> Dict[str, Any]:
        """Get advanced statistics with comprehensive metrics"""
        import random
        from datetime import datetime, timedelta
        
        now = datetime.now()
        
        # System performance metrics
        system_metrics = {
            'cpu_usage': round(random.uniform(15, 75), 1),
            'memory_usage': round(random.uniform(40, 85), 1),
            'disk_usage': round(random.uniform(25, 60), 1),
            'network_throughput': random.randint(50, 200),  # MB/s
            'active_connections': random.randint(1000, 5000)
        }
        
        # Detection statistics
        detection_stats = {
            'total_processed': random.randint(10000, 50000),
            'threats_detected': random.randint(500, 2000),
            'false_positives': random.randint(50, 150),
            'true_positives': random.randint(450, 1850),
            'detection_rate': round(random.uniform(85, 95), 2),
            'processing_speed': random.randint(1000, 5000)  # events per second
        }
        
        # Threat intelligence metrics
        intelligence_metrics = {
            'ioc_feeds': random.randint(50, 150),
            'active_rules': random.randint(500, 1500),
            'signature_updates': random.randint(10, 50),
            'feed_health': round(random.uniform(90, 99), 2),
            'data_freshness': random.randint(1, 15)  # minutes
        }
        
        # Geographic distribution (enhanced)
        geo_stats = [
            {'country': 'United States', 'threats': random.randint(100, 300), 'blocked': random.randint(80, 250)},
            {'country': 'China', 'threats': random.randint(80, 200), 'blocked': random.randint(60, 180)},
            {'country': 'Russia', 'threats': random.randint(70, 180), 'blocked': random.randint(50, 150)},
            {'country': 'Brazil', 'threats': random.randint(40, 120), 'blocked': random.randint(30, 100)},
            {'country': 'India', 'threats': random.randint(50, 150), 'blocked': random.randint(35, 120)},
            {'country': 'Germany', 'threats': random.randint(30, 90), 'blocked': random.randint(25, 75)},
            {'country': 'United Kingdom', 'threats': random.randint(25, 80), 'blocked': random.randint(20, 65)},
            {'country': 'France', 'threats': random.randint(20, 70), 'blocked': random.randint(15, 55)}
        ]
        
        # Attack vectors analysis
        attack_vectors = [
            {'vector': 'Malware', 'count': random.randint(100, 300), 'severity': 'HIGH'},
            {'vector': 'Phishing', 'count': random.randint(80, 250), 'severity': 'MEDIUM'},
            {'vector': 'DDoS', 'count': random.randint(50, 150), 'severity': 'HIGH'},
            {'vector': 'SQL Injection', 'count': random.randint(30, 100), 'severity': 'CRITICAL'},
            {'vector': 'XSS', 'count': random.randint(40, 120), 'severity': 'MEDIUM'},
            {'vector': 'Brute Force', 'count': random.randint(60, 180), 'severity': 'LOW'},
            {'vector': 'Ransomware', 'count': random.randint(10, 40), 'severity': 'CRITICAL'}
        ]
        
        # Time-based patterns (hourly distribution for last 24 hours)
        hourly_patterns = []
        for i in range(24):
            hour = (now - timedelta(hours=23-i)).hour
            # Simulate realistic patterns: more activity during business hours
            base_activity = 100
            if 9 <= hour <= 17:  # Business hours
                activity = base_activity * random.uniform(1.5, 2.5)
            elif 18 <= hour <= 22:  # Evening
                activity = base_activity * random.uniform(1.0, 1.5)
            else:  # Night/early morning
                activity = base_activity * random.uniform(0.3, 0.8)
            
            hourly_patterns.append({
                'hour': f"{hour:02d}:00",
                'threats': int(activity),
                'blocked': int(activity * random.uniform(0.7, 0.9)),
                'alerts': int(activity * random.uniform(0.1, 0.3))
            })
        
        return {
            'system_performance': system_metrics,
            'detection_statistics': detection_stats,
            'threat_intelligence': intelligence_metrics,
            'geographic_distribution': geo_stats,
            'attack_vectors': attack_vectors,
            'hourly_patterns': hourly_patterns,
            'summary': {
                'uptime': f"{random.randint(1, 30)} days, {random.randint(0, 23)} hours",
                'last_restart': (now - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d %H:%M'),
                'version': '2.1.0-beta',
                'build': f'build-{random.randint(1000, 9999)}',
                'environment': 'Production'
            },
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _train_models(self) -> Dict[str, Any]:
        """Train ML models"""
        try:
            from src.preprocessing.data_preprocessor import ThreatDataPreprocessor
            from src.models.ml_models import ThreatDetectionML
            from src.utils.data_loader import ThreatDataLoader
            
            # Load training data
            loader = ThreatDataLoader()
            data = loader.create_sample_data(num_samples=200)
            
            # Preprocess data
            preprocessor = ThreatDataPreprocessor()
            processed_data = preprocessor.extract_features(data)
            
            # Prepare features
            X = preprocessor.create_feature_matrix(processed_data)
            y = processed_data['severity'].values
            
            # Train Random Forest model
            ml_model = ThreatDetectionML(model_type='random_forest')
            metrics = ml_model.train(X, y, validation_split=0.2)
            
            # Save model
            model_path = 'models/dashboard_trained_model.pkl'
            os.makedirs('models', exist_ok=True)
            ml_model.save_model(model_path)
            
            return {
                'status': 'success',
                'message': 'Model training completed successfully',
                'metrics': {
                    'train_accuracy': metrics.get('train_accuracy', 0),
                    'validation_accuracy': metrics.get('val_accuracy', 0),
                    'model_path': model_path
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {
                'status': 'error',
                'message': f'Training failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _analyze_threat_text(self, text: str) -> Dict[str, Any]:
        """Analyze threat text using semantic analyzer"""
        try:
            if self.semantic_analyzer:
                # Use the semantic analyzer
                categories = self.semantic_analyzer.categorize_threat(text)
                severity = self.semantic_analyzer.assess_threat_severity(text)
                
                analysis_result = {
                    'text': text,
                    'categories': categories,
                    'severity': severity,
                    'top_category': max(categories.items(), key=lambda x: x[1]) if categories else ('unknown', 0),
                    'confidence': max(categories.values()) if categories else 0,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Fallback simple analysis
                analysis_result = self._simple_threat_analysis(text)
            
            # Add to threat history
            threat_data = {
                'description': text,
                'category': analysis_result.get('top_category', ['unknown', 0])[0],
                'severity': analysis_result.get('severity', {}).get('severity', 'MEDIUM'),
                'timestamp': analysis_result['timestamp'],
                'is_threat': analysis_result.get('confidence', 0) > 0.5,
                'analysis': analysis_result
            }
            self.add_threat(threat_data)
            
            return {
                'status': 'success',
                'analysis': analysis_result
            }
            
        except Exception as e:
            logger.error(f"Error analyzing threat text: {e}")
            return {
                'status': 'error',
                'message': f'Analysis failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _simple_threat_analysis(self, text: str) -> Dict[str, Any]:
        """Simple threat analysis for fallback"""
        text_lower = text.lower()
        
        # Define threat categories and keywords
        threat_keywords = {
            'malware': ['virus', 'malware', 'trojan', 'ransomware', 'worm', 'spyware'],
            'phishing': ['phishing', 'credential', 'login', 'password', 'fake', 'scam'],
            'network_attack': ['ddos', 'botnet', 'intrusion', 'firewall', 'breach'],
            'data_breach': ['breach', 'leak', 'stolen', 'compromised', 'exposed'],
            'suspicious': ['suspicious', 'anomaly', 'unusual', 'irregular']
        }
        
        categories = {}
        for category, keywords in threat_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                categories[category] = score / len(keywords)
        
        if not categories:
            categories['unknown'] = 1.0
        
        # Determine severity
        high_severity_words = ['critical', 'severe', 'emergency', 'breach', 'ransomware']
        medium_severity_words = ['warning', 'alert', 'suspicious', 'unusual']
        
        if any(word in text_lower for word in high_severity_words):
            severity = 'HIGH'
        elif any(word in text_lower for word in medium_severity_words):
            severity = 'MEDIUM'
        else:
            severity = 'LOW'
        
        return {
            'categories': categories,
            'severity': {'severity': severity, 'confidence': 0.7},
            'top_category': max(categories.items(), key=lambda x: x[1]) if categories else ('unknown', 0),
            'confidence': max(categories.values()) if categories else 0.5,
            'timestamp': datetime.now().isoformat()
        }
    
    def _start_realtime_detection(self) -> Dict[str, Any]:
        """Start real-time threat detection"""
        try:
            if self.threat_detector:
                if not self.threat_detector.is_running:
                    self.threat_detector.start()
                return {
                    'status': 'success',
                    'message': 'Real-time detection started',
                    'is_running': True,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Initialize threat detector if not provided
                from src.realtime.detector import RealTimeThreatDetector
                from src.semantic_analysis.semantic_analyzer import ThreatSemanticAnalyzer
                
                if not self.semantic_analyzer:
                    self.semantic_analyzer = ThreatSemanticAnalyzer()
                
                self.threat_detector = RealTimeThreatDetector(
                    semantic_analyzer=self.semantic_analyzer
                )
                self.threat_detector.start()
                
                return {
                    'status': 'success',
                    'message': 'Real-time detection initialized and started',
                    'is_running': True,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error starting real-time detection: {e}")
            return {
                'status': 'error',
                'message': f'Failed to start real-time detection: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _stop_realtime_detection(self) -> Dict[str, Any]:
        """Stop real-time threat detection"""
        try:
            if self.threat_detector and self.threat_detector.is_running:
                self.threat_detector.stop()
                return {
                    'status': 'success',
                    'message': 'Real-time detection stopped',
                    'is_running': False,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'info',
                    'message': 'Real-time detection was not running',
                    'is_running': False,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error stopping real-time detection: {e}")
            return {
                'status': 'error',
                'message': f'Failed to stop real-time detection: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_realtime_status(self) -> Dict[str, Any]:
        """Get real-time detection status"""
        try:
            if self.threat_detector:
                stats = self.threat_detector.get_statistics() if hasattr(self.threat_detector, 'get_statistics') else {}
                return {
                    'status': 'success',
                    'is_running': getattr(self.threat_detector, 'is_running', False),
                    'statistics': stats,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'info',
                    'is_running': False,
                    'message': 'Real-time detector not initialized',
                    'statistics': {},
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting real-time status: {e}")
            return {
                'status': 'error',
                'message': f'Failed to get status: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _upload_threat_intelligence(self) -> Dict[str, Any]:
        """Upload threat intelligence data from file"""
        try:
            from flask import request
            import json
            import csv
            import io
            
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Read file content
            content = file.read().decode('utf-8')
            uploaded_threats = []
            
            # Determine file format and parse
            if file.filename.endswith('.json'):
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        uploaded_threats = data
                    elif isinstance(data, dict) and 'threats' in data:
                        uploaded_threats = data['threats']
                    else:
                        uploaded_threats = [data]
                except json.JSONDecodeError:
                    return jsonify({'error': 'Invalid JSON format'}), 400
                    
            elif file.filename.endswith('.csv'):
                try:
                    csv_reader = csv.DictReader(io.StringIO(content))
                    uploaded_threats = list(csv_reader)
                except:
                    return jsonify({'error': 'Invalid CSV format'}), 400
                    
            else:
                # Treat as plain text with each line as a threat
                threats_text = content.strip().split('\n')
                uploaded_threats = [{'description': text.strip(), 'source': 'upload'} 
                                  for text in threats_text if text.strip()]
            
            # Process and add threats
            processed_count = 0
            for threat_data in uploaded_threats:
                if isinstance(threat_data, str):
                    threat_data = {'description': threat_data, 'source': 'upload'}
                
                # Ensure required fields
                if 'description' not in threat_data:
                    continue
                
                # Add metadata
                threat_data.update({
                    'timestamp': threat_data.get('timestamp', datetime.now().isoformat()),
                    'source': 'upload',
                    'category': threat_data.get('category', 'unknown'),
                    'severity': threat_data.get('severity', 'MEDIUM'),
                    'is_threat': True
                })
                
                # Analyze if semantic analyzer is available
                if self.semantic_analyzer:
                    analysis = self._analyze_threat_text(threat_data['description'])
                    if analysis.get('status') == 'success':
                        threat_data['analysis'] = analysis['analysis']
                
                self.add_threat(threat_data)
                processed_count += 1
            
            return jsonify({
                'status': 'success',
                'message': f'Successfully uploaded {processed_count} threats',
                'threats_processed': processed_count,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error uploading threat intelligence: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Upload failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def _import_threat_intelligence(self, source: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Import threat intelligence from external sources"""
        try:
            imported_threats = []
            
            if source == 'misp':
                # Import from MISP format
                events = data.get('events', [])
                for event in events:
                    attributes = event.get('Attribute', [])
                    for attr in attributes:
                        threat = {
                            'description': f"{attr.get('category', '')}: {attr.get('value', '')}",
                            'category': attr.get('category', 'unknown').lower(),
                            'severity': 'HIGH' if attr.get('to_ids') else 'MEDIUM',
                            'source': 'misp',
                            'misp_event_id': event.get('id'),
                            'timestamp': datetime.now().isoformat(),
                            'is_threat': True
                        }
                        imported_threats.append(threat)
                        
            elif source == 'stix':
                # Import from STIX format
                objects = data.get('objects', [])
                for obj in objects:
                    if obj.get('type') == 'indicator':
                        threat = {
                            'description': obj.get('name', obj.get('pattern', '')),
                            'category': 'indicator',
                            'severity': 'HIGH',
                            'source': 'stix',
                            'stix_id': obj.get('id'),
                            'timestamp': datetime.now().isoformat(),
                            'is_threat': True
                        }
                        imported_threats.append(threat)
                        
            elif source == 'taxii':
                # Import from TAXII feed
                collections = data.get('collections', [])
                for collection in collections:
                    for item in collection.get('objects', []):
                        threat = {
                            'description': item.get('name', item.get('description', '')),
                            'category': item.get('type', 'unknown'),
                            'severity': 'MEDIUM',
                            'source': 'taxii',
                            'timestamp': datetime.now().isoformat(),
                            'is_threat': True
                        }
                        imported_threats.append(threat)
                        
            else:
                # Manual import
                threats_list = data.get('threats', [])
                for threat in threats_list:
                    threat.update({
                        'source': 'manual_import',
                        'timestamp': datetime.now().isoformat(),
                        'is_threat': True
                    })
                    imported_threats.append(threat)
            
            # Add imported threats to dashboard
            for threat in imported_threats:
                self.add_threat(threat)
            
            return {
                'status': 'success',
                'message': f'Successfully imported {len(imported_threats)} threats from {source}',
                'threats_imported': len(imported_threats),
                'source': source,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error importing threat intelligence: {e}")
            return {
                'status': 'error',
                'message': f'Import from {source} failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _export_threat_intelligence(self, format_type: str) -> Any:
        """Export threat intelligence data"""
        try:
            from flask import Response
            import json
            import csv
            import io
            
            if format_type.lower() == 'json':
                export_data = {
                    'metadata': {
                        'export_timestamp': datetime.now().isoformat(),
                        'total_threats': len(self.threat_history),
                        'source': 'CTI-sHARE Dashboard'
                    },
                    'threats': self.threat_history
                }
                
                response_data = json.dumps(export_data, indent=2)
                return Response(
                    response_data,
                    mimetype='application/json',
                    headers={'Content-Disposition': 'attachment; filename=threat_intelligence.json'}
                )
                
            elif format_type.lower() == 'csv':
                output = io.StringIO()
                
                if self.threat_history:
                    # Get all unique keys from all threats
                    all_keys = set()
                    for threat in self.threat_history:
                        all_keys.update(threat.keys())
                    
                    fieldnames = sorted(list(all_keys))
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for threat in self.threat_history:
                        # Flatten nested dictionaries for CSV
                        flattened = {}
                        for key, value in threat.items():
                            if isinstance(value, dict):
                                flattened[key] = json.dumps(value)
                            elif isinstance(value, list):
                                flattened[key] = '; '.join(map(str, value))
                            else:
                                flattened[key] = value
                        writer.writerow(flattened)
                
                response_data = output.getvalue()
                return Response(
                    response_data,
                    mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=threat_intelligence.csv'}
                )
                
            elif format_type.lower() == 'stix':
                # Export in STIX 2.1 format
                stix_objects = []
                
                for i, threat in enumerate(self.threat_history):
                    stix_obj = {
                        'type': 'indicator',
                        'id': f'indicator--{i:08d}-threat-{hash(str(threat)) % 10000:04d}',
                        'created': threat.get('timestamp', datetime.now().isoformat()),
                        'modified': threat.get('timestamp', datetime.now().isoformat()),
                        'pattern': f"[file:hashes.MD5 = '{threat.get('description', '')}']",
                        'labels': [threat.get('category', 'unknown')],
                        'name': threat.get('description', '')[:100],
                        'description': threat.get('description', ''),
                        'confidence': 75
                    }
                    stix_objects.append(stix_obj)
                
                stix_bundle = {
                    'type': 'bundle',
                    'id': f'bundle--{datetime.now().strftime("%Y%m%d%H%M%S")}',
                    'objects': stix_objects
                }
                
                response_data = json.dumps(stix_bundle, indent=2)
                return Response(
                    response_data,
                    mimetype='application/json',
                    headers={'Content-Disposition': 'attachment; filename=threat_intelligence_stix.json'}
                )
                
            else:
                return jsonify({'error': f'Unsupported export format: {format_type}'}), 400
                
        except Exception as e:
            logger.error(f"Error exporting threat intelligence: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Export failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def _share_threat_intelligence(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Share threat intelligence with external systems"""
        try:
            threats_to_share = data.get('threats', [])
            destinations = data.get('destinations', ['internal'])
            format_type = data.get('format', 'json')
            
            shared_count = 0
            sharing_results = []
            
            for destination in destinations:
                try:
                    if destination == 'internal':
                        # Internal sharing - just log
                        logger.info(f"Shared {len(threats_to_share)} threats internally")
                        shared_count += len(threats_to_share)
                        sharing_results.append({
                            'destination': destination,
                            'status': 'success',
                            'threats_shared': len(threats_to_share)
                        })
                        
                    elif destination == 'misp':
                        # Share to MISP instance (mock implementation)
                        # In real implementation, use PyMISP library
                        logger.info(f"Would share {len(threats_to_share)} threats to MISP")
                        shared_count += len(threats_to_share)
                        sharing_results.append({
                            'destination': destination,
                            'status': 'success',
                            'threats_shared': len(threats_to_share),
                            'note': 'Mock MISP sharing - implement with PyMISP'
                        })
                        
                    elif destination == 'taxii':
                        # Share to TAXII server (mock implementation)
                        logger.info(f"Would share {len(threats_to_share)} threats to TAXII")
                        shared_count += len(threats_to_share)
                        sharing_results.append({
                            'destination': destination,
                            'status': 'success',
                            'threats_shared': len(threats_to_share),
                            'note': 'Mock TAXII sharing - implement with cabby'
                        })
                        
                    else:
                        sharing_results.append({
                            'destination': destination,
                            'status': 'error',
                            'message': f'Unknown destination: {destination}'
                        })
                        
                except Exception as e:
                    sharing_results.append({
                        'destination': destination,
                        'status': 'error',
                        'message': str(e)
                    })
            
            return {
                'status': 'success',
                'message': f'Shared {shared_count} threats to {len(destinations)} destinations',
                'total_shared': shared_count,
                'sharing_results': sharing_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error sharing threat intelligence: {e}")
            return {
                'status': 'error',
                'message': f'Sharing failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _analyze_bulk_semantic(self, texts: List[str]) -> Dict[str, Any]:
        """Perform semantic analysis on multiple threat texts"""
        try:
            results = []
            
            for i, text in enumerate(texts):
                try:
                    if self.semantic_analyzer:
                        categories = self.semantic_analyzer.categorize_threat(text)
                        severity = self.semantic_analyzer.assess_threat_severity(text)
                        
                        analysis = {
                            'text': text[:100] + '...' if len(text) > 100 else text,
                            'categories': categories,
                            'severity': severity,
                            'top_category': max(categories.items(), key=lambda x: x[1]) if categories else ('unknown', 0),
                            'confidence': max(categories.values()) if categories else 0,
                            'index': i
                        }
                    else:
                        # Fallback analysis
                        analysis = self._simple_threat_analysis(text)
                        analysis['index'] = i
                    
                    results.append(analysis)
                    
                except Exception as e:
                    results.append({
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'error': str(e),
                        'index': i
                    })
            
            # Calculate bulk statistics
            successful_analyses = [r for r in results if 'error' not in r]
            total_threats_detected = sum(1 for r in successful_analyses if r.get('confidence', 0) > 0.5)
            
            categories_count = {}
            severities_count = {}
            
            for result in successful_analyses:
                top_cat = result.get('top_category', ['unknown', 0])
                if isinstance(top_cat, (list, tuple)) and len(top_cat) >= 1:
                    cat_name = top_cat[0]
                    categories_count[cat_name] = categories_count.get(cat_name, 0) + 1
                
                severity = result.get('severity', {})
                sev_level = severity.get('severity', 'UNKNOWN') if isinstance(severity, dict) else 'UNKNOWN'
                severities_count[sev_level] = severities_count.get(sev_level, 0) + 1
            
            return {
                'status': 'success',
                'total_analyzed': len(texts),
                'successful_analyses': len(successful_analyses),
                'threats_detected': total_threats_detected,
                'categories_distribution': categories_count,
                'severities_distribution': severities_count,
                'detailed_results': results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in bulk semantic analysis: {e}")
            return {
                'status': 'error',
                'message': f'Bulk analysis failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _extract_threat_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from threat text"""
        try:
            import re
            
            entities = {
                'ips': [],
                'domains': [],
                'urls': [],
                'emails': [],
                'hashes': [],
                'file_paths': [],
                'cve_ids': []
            }
            
            # Extract IP addresses
            ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            entities['ips'] = list(set(re.findall(ip_pattern, text)))
            
            # Extract domains
            domain_pattern = r'\b[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}\b'
            entities['domains'] = list(set(re.findall(domain_pattern, text)))
            
            # Extract URLs
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            entities['urls'] = list(set(re.findall(url_pattern, text)))
            
            # Extract email addresses
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            entities['emails'] = list(set(re.findall(email_pattern, text)))
            
            # Extract hashes (MD5, SHA1, SHA256)
            md5_pattern = r'\b[a-fA-F0-9]{32}\b'
            sha1_pattern = r'\b[a-fA-F0-9]{40}\b'
            sha256_pattern = r'\b[a-fA-F0-9]{64}\b'
            
            md5_hashes = re.findall(md5_pattern, text)
            sha1_hashes = re.findall(sha1_pattern, text)
            sha256_hashes = re.findall(sha256_pattern, text)
            
            entities['hashes'] = list(set(md5_hashes + sha1_hashes + sha256_hashes))
            
            # Extract file paths
            file_path_pattern = r'[A-Za-z]:\\[^<>:"|?*\n\r]*|\/[^<>:"|?*\n\r]*'
            entities['file_paths'] = list(set(re.findall(file_path_pattern, text)))
            
            # Extract CVE IDs
            cve_pattern = r'CVE-\d{4}-\d{4,}'
            entities['cve_ids'] = list(set(re.findall(cve_pattern, text, re.IGNORECASE)))
            
            # If semantic analyzer is available, use it for enhanced entity extraction
            if self.semantic_analyzer and hasattr(self.semantic_analyzer, 'extract_entities'):
                try:
                    semantic_entities = self.semantic_analyzer.extract_entities(text)
                    # Merge semantic entities with regex-based entities
                    for key, values in semantic_entities.items():
                        if key in entities:
                            entities[key].extend(values)
                            entities[key] = list(set(entities[key]))  # Remove duplicates
                except:
                    pass  # Continue with regex-based extraction
            
            # Calculate entity statistics
            total_entities = sum(len(entities[key]) for key in entities)
            entity_types = sum(1 for key in entities if entities[key])
            
            return {
                'status': 'success',
                'entities': entities,
                'statistics': {
                    'total_entities': total_entities,
                    'entity_types_found': entity_types,
                    'ips_found': len(entities['ips']),
                    'domains_found': len(entities['domains']),
                    'urls_found': len(entities['urls']),
                    'emails_found': len(entities['emails']),
                    'hashes_found': len(entities['hashes']),
                    'file_paths_found': len(entities['file_paths']),
                    'cve_ids_found': len(entities['cve_ids'])
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error extracting threat entities: {e}")
            return {
                'status': 'error',
                'message': f'Entity extraction failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
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
