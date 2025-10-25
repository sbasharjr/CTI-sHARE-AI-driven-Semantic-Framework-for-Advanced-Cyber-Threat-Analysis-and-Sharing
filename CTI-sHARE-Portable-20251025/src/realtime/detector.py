"""
Real-time threat detection system
"""

import numpy as np
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import threading
import queue
import time
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeThreatDetector:
    """
    Real-time threat detection and monitoring system
    """
    
    def __init__(self, ml_model=None, dl_model=None, semantic_analyzer=None):
        """
        Initialize real-time detector
        
        Args:
            ml_model: Machine learning model for threat detection
            dl_model: Deep learning model for threat detection
            semantic_analyzer: Semantic analyzer for threat categorization
        """
        self.ml_model = ml_model
        self.dl_model = dl_model
        self.semantic_analyzer = semantic_analyzer
        self.threat_queue = queue.Queue()
        self.is_running = False
        self.detection_thread = None
        self.callbacks = []
        self.detected_threats = []
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register callback function to be called when threat is detected
        
        Args:
            callback: Callback function that takes threat dict as input
        """
        self.callbacks.append(callback)
    
    def add_threat_data(self, data: Dict[str, Any]) -> None:
        """
        Add threat data to processing queue
        
        Args:
            data: Threat data dictionary
        """
        self.threat_queue.put(data)
    
    def _process_threat(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process single threat data point
        
        Args:
            data: Threat data dictionary
            
        Returns:
            Processed threat with detection results
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'original_data': data,
            'predictions': {},
            'semantic_analysis': {},
            'is_threat': False,
            'confidence': 0.0
        }
        
        # Extract features for ML/DL models
        if 'features' in data:
            features = np.array(data['features']).reshape(1, -1)
            
            # ML model prediction
            if self.ml_model and hasattr(self.ml_model, 'predict_proba'):
                try:
                    ml_proba = self.ml_model.predict_proba(features)[0]
                    ml_pred = self.ml_model.predict(features)[0]
                    result['predictions']['ml_model'] = {
                        'prediction': int(ml_pred),
                        'confidence': float(max(ml_proba))
                    }
                except Exception as e:
                    logger.error(f"ML model prediction error: {e}")
            
            # DL model prediction
            if self.dl_model and hasattr(self.dl_model, 'predict_proba'):
                try:
                    dl_proba = self.dl_model.predict_proba(features)[0]
                    dl_pred = self.dl_model.predict(features)[0]
                    result['predictions']['dl_model'] = {
                        'prediction': int(dl_pred),
                        'confidence': float(max(dl_proba))
                    }
                except Exception as e:
                    logger.error(f"DL model prediction error: {e}")
        
        # Semantic analysis
        if self.semantic_analyzer and 'text' in data:
            try:
                categories = self.semantic_analyzer.categorize_threat(data['text'])
                entities = self.semantic_analyzer.extract_threat_entities(data['text'])
                severity = self.semantic_analyzer.assess_threat_severity(data['text'])
                
                result['semantic_analysis'] = {
                    'categories': categories,
                    'entities': entities,
                    'severity': severity
                }
            except Exception as e:
                logger.error(f"Semantic analysis error: {e}")
        
        # Determine if threat is detected
        all_predictions = [
            p.get('prediction', 0) 
            for p in result['predictions'].values()
        ]
        
        if all_predictions:
            # Consider it a threat if majority of models predict threat
            result['is_threat'] = sum(all_predictions) > len(all_predictions) / 2
            
            # Average confidence from all models
            confidences = [
                p.get('confidence', 0.0) 
                for p in result['predictions'].values()
            ]
            result['confidence'] = sum(confidences) / len(confidences)
        
        return result
    
    def _detection_loop(self) -> None:
        """
        Main detection loop running in separate thread
        """
        logger.info("Real-time threat detection started")
        
        while self.is_running:
            try:
                # Get data from queue with timeout
                data = self.threat_queue.get(timeout=1.0)
                
                # Process threat
                result = self._process_threat(data)
                
                # Store detected threat
                self.detected_threats.append(result)
                
                # Call registered callbacks
                for callback in self.callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                # Log detection
                if result['is_threat']:
                    logger.warning(f"THREAT DETECTED: {result['confidence']:.2%} confidence")
                else:
                    logger.info("No threat detected")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Detection loop error: {e}")
        
        logger.info("Real-time threat detection stopped")
    
    def start(self) -> None:
        """
        Start real-time threat detection
        """
        if self.is_running:
            logger.warning("Detection already running")
            return
        
        self.is_running = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        logger.info("Real-time detection started")
    
    def stop(self) -> None:
        """
        Stop real-time threat detection
        """
        if not self.is_running:
            logger.warning("Detection not running")
            return
        
        self.is_running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=5.0)
        logger.info("Real-time detection stopped")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detection statistics
        
        Returns:
            Statistics dictionary
        """
        total_processed = len(self.detected_threats)
        threats_detected = sum(1 for t in self.detected_threats if t['is_threat'])
        
        avg_confidence = 0.0
        if self.detected_threats:
            avg_confidence = sum(
                t['confidence'] for t in self.detected_threats
            ) / total_processed
        
        return {
            'total_processed': total_processed,
            'threats_detected': threats_detected,
            'detection_rate': threats_detected / total_processed if total_processed > 0 else 0.0,
            'average_confidence': avg_confidence,
            'queue_size': self.threat_queue.qsize()
        }
    
    def get_recent_threats(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent detected threats
        
        Args:
            limit: Maximum number of threats to return
            
        Returns:
            List of recent threats
        """
        threats = [t for t in self.detected_threats if t['is_threat']]
        return threats[-limit:]


class ThreatMonitor:
    """
    Monitor and aggregate threat detection events
    """
    
    def __init__(self, detector: RealTimeThreatDetector):
        """
        Initialize threat monitor
        
        Args:
            detector: Real-time threat detector instance
        """
        self.detector = detector
        self.alerts = []
        self.detector.register_callback(self._on_threat_detected)
    
    def _on_threat_detected(self, threat: Dict[str, Any]) -> None:
        """
        Callback for threat detection events
        
        Args:
            threat: Detected threat dictionary
        """
        if threat['is_threat']:
            alert = {
                'timestamp': threat['timestamp'],
                'confidence': threat['confidence'],
                'severity': threat.get('semantic_analysis', {}).get('severity', {}).get('severity', 'UNKNOWN'),
                'summary': self._generate_alert_summary(threat)
            }
            self.alerts.append(alert)
            logger.warning(f"ALERT: {alert['summary']}")
    
    def _generate_alert_summary(self, threat: Dict[str, Any]) -> str:
        """
        Generate human-readable alert summary
        
        Args:
            threat: Threat dictionary
            
        Returns:
            Alert summary string
        """
        severity = threat.get('semantic_analysis', {}).get('severity', {}).get('severity', 'UNKNOWN')
        confidence = threat['confidence']
        
        categories = threat.get('semantic_analysis', {}).get('categories', {})
        top_category = max(categories.items(), key=lambda x: x[1])[0] if categories else 'unknown'
        
        return f"{severity} severity {top_category} threat detected (confidence: {confidence:.2%})"
    
    def get_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent alerts
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of alerts
        """
        return self.alerts[-limit:]
    
    def clear_alerts(self) -> None:
        """
        Clear all stored alerts
        """
        self.alerts.clear()
