#!/usr/bin/env python3
"""
CTI-sHARE Production Server - Lite Version
==========================================

Simplified production server for creating executables without heavy ML dependencies.
This version focuses on the core dashboard functionality and MISP integration.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS

# Add src to path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir / 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, 
           template_folder='src/dashboard/templates',
           static_folder='src/dashboard/static')
CORS(app)

# In-memory storage for demo
threats_db = []
notifications_db = []

@app.route('/')
def dashboard():
    """Serve the main dashboard"""
    try:
        return render_template('dashboard.html')
    except Exception as e:
        logger.error(f"Error serving dashboard: {e}")
        return f"Error loading dashboard: {e}", 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': 'v1.0.0-lite'
    })

@app.route('/api/dashboard/stats')
def get_stats():
    """Get dashboard statistics"""
    return jsonify({
        'total_threats': len(threats_db),
        'high_severity': len([t for t in threats_db if t.get('severity') == 'high']),
        'medium_severity': len([t for t in threats_db if t.get('severity') == 'medium']),
        'low_severity': len([t for t in threats_db if t.get('severity') == 'low']),
        'system_status': 'active',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/dashboard/threats', methods=['GET', 'POST'])
def handle_threats():
    """Handle threat data"""
    if request.method == 'POST':
        try:
            data = request.get_json()
            threat = {
                'id': len(threats_db) + 1,
                'description': data.get('description', ''),
                'severity': data.get('severity', 'medium'),
                'category': data.get('category', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'source': 'manual'
            }
            threats_db.append(threat)
            
            logger.info(f"New threat added: {threat['id']}")
            return jsonify({'status': 'success', 'threat': threat})
        except Exception as e:
            logger.error(f"Error adding threat: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    else:
        # GET request
        limit = request.args.get('limit', 10, type=int)
        recent_threats = threats_db[-limit:] if threats_db else []
        return jsonify({'threats': recent_threats})

@app.route('/api/dashboard/analyze', methods=['POST'])
def analyze_text():
    """Simple text analysis (without ML dependencies)"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        # Simple keyword-based analysis
        threat_keywords = {
            'malware': ['virus', 'trojan', 'ransomware', 'malware', 'backdoor'],
            'network': ['ddos', 'attack', 'intrusion', 'breach', 'vulnerability'],
            'phishing': ['phishing', 'spoofing', 'email', 'fake', 'credential']
        }
        
        detected_threats = []
        confidence = 0.5
        
        text_lower = text.lower()
        for category, keywords in threat_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_threats.append({
                        'type': category,
                        'keyword': keyword,
                        'confidence': min(0.9, confidence + 0.1)
                    })
                    confidence += 0.1
        
        result = {
            'status': 'success',
            'analysis': {
                'text': text,
                'threats_detected': len(detected_threats),
                'confidence': min(1.0, confidence),
                'threats': detected_threats,
                'summary': f"Detected {len(detected_threats)} potential threat indicators"
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in text analysis: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/dashboard/train', methods=['POST'])
def train_models():
    """Simulate model training"""
    try:
        # Simulate training process
        import time
        time.sleep(2)  # Simulate training time
        
        return jsonify({
            'status': 'success',
            'message': 'Model training completed (simulated)',
            'metrics': {
                'train_accuracy': 0.95,
                'validation_accuracy': 0.92,
                'training_time': '2.0s'
            }
        })
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/dashboard/realtime/start', methods=['POST'])
def start_realtime():
    """Start real-time monitoring (simulated)"""
    return jsonify({
        'status': 'success',
        'message': 'Real-time monitoring started (simulated)'
    })

@app.route('/api/dashboard/realtime/stop', methods=['POST'])
def stop_realtime():
    """Stop real-time monitoring (simulated)"""
    return jsonify({
        'status': 'success',
        'message': 'Real-time monitoring stopped (simulated)'
    })

@app.route('/api/dashboard/realtime/status')
def realtime_status():
    """Get real-time status"""
    return jsonify({
        'active': False,
        'threats_processed': 0,
        'accuracy': 0.0,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/dashboard/threats/categories')
def get_threat_categories():
    """Get threat categories"""
    categories = {}
    for threat in threats_db:
        cat = threat.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    return jsonify({
        'categories': categories,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/dashboard/threats/recent')
def get_recent_threats():
    """Get recent threats"""
    limit = request.args.get('limit', 5, type=int)
    recent = threats_db[-limit:] if threats_db else []
    return jsonify({'threats': recent})

# MISP Integration endpoints (simplified)
@app.route('/api/dashboard/misp/test-connection', methods=['POST'])
def test_misp_connection():
    """Test MISP connection (simulated)"""
    try:
        data = request.get_json()
        server_url = data.get('server_url', '')
        api_key = data.get('api_key', '')
        
        if not server_url or not api_key:
            return jsonify({
                'status': 'error',
                'message': 'Server URL and API key are required'
            })
        
        # Simulate connection test
        return jsonify({
            'status': 'success',
            'message': 'MISP connection successful (simulated)',
            'server_info': {'version': 'MISP 2.4.x'},
            'organization': 'Test Organization',
            'stats': {
                'events': 100,
                'attributes': 1500,
                'iocs': 500,
                'events_today': 5,
                'attributes_today': 75,
                'iocs_synced': 50
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/dashboard/misp/import-events', methods=['POST'])
def import_misp_events():
    """Import MISP events (simulated)"""
    return jsonify({
        'status': 'success',
        'events_imported': 25,
        'updated_stats': {
            'events': 125,
            'attributes': 1625,
            'iocs': 550
        }
    })

@app.route('/api/dashboard/misp/import-attributes', methods=['POST'])
def import_misp_attributes():
    """Import MISP attributes (simulated)"""
    return jsonify({
        'status': 'success',
        'attributes_imported': 150,
        'updated_stats': {
            'events': 125,
            'attributes': 1775,
            'iocs': 550
        }
    })

@app.route('/api/dashboard/misp/import-iocs', methods=['POST'])
def import_misp_iocs():
    """Import MISP IOCs (simulated)"""
    return jsonify({
        'status': 'success',
        'iocs_imported': 75,
        'ioc_breakdown': {
            'ip': 25,
            'domain': 20,
            'hash': 15,
            'url': 10,
            'email': 5
        },
        'updated_stats': {
            'events': 125,
            'attributes': 1775,
            'iocs': 625
        }
    })

@app.route('/api/dashboard/misp/export', methods=['POST'])
def export_to_misp():
    """Export to MISP (simulated)"""
    return jsonify({
        'status': 'success',
        'event_id': 'test-event-123',
        'items_exported': 10
    })

@app.route('/api/dashboard/misp/sync', methods=['POST'])
def sync_misp_data():
    """Sync MISP data (simulated)"""
    return jsonify({
        'status': 'success',
        'imported': 15,
        'exported': 8,
        'conflicts': 0,
        'updated_stats': {
            'events': 140,
            'attributes': 1850,
            'iocs': 675
        }
    })

def load_sample_data():
    """Load sample threat data"""
    sample_threats = [
        {
            'id': 1,
            'description': 'Malware detected in email attachment',
            'severity': 'high',
            'category': 'malware',
            'timestamp': datetime.now().isoformat(),
            'source': 'email_scanner'
        },
        {
            'id': 2,
            'description': 'Suspicious network traffic detected',
            'severity': 'medium',
            'category': 'network',
            'timestamp': datetime.now().isoformat(),
            'source': 'network_monitor'
        },
        {
            'id': 3,
            'description': 'Phishing attempt blocked',
            'severity': 'high',
            'category': 'phishing',
            'timestamp': datetime.now().isoformat(),
            'source': 'web_filter'
        }
    ]
    
    threats_db.extend(sample_threats)
    logger.info(f"Loaded {len(sample_threats)} sample threats")

def run_production_server():
    """Run the CTI-sHARE dashboard using Waitress production server"""
    try:
        from waitress import serve
        
        print("=" * 80)
        print("🛡️  CTI-sHARE Dashboard - Production Server (Lite)")
        print("=" * 80)
        print("🚀 Starting production WSGI server...")
        print("📡 Server URL: http://localhost:5001")
        print("🔧 WSGI Server: Waitress")
        print("⚡ Environment: Production (Lite)")
        print("📝 Features: Core Dashboard + MISP Integration")
        print("=" * 80)
        
        # Load sample data
        load_sample_data()
        
        # Production server configuration
        serve(
            app,
            host='0.0.0.0',
            port=5001,
            threads=4,
            connection_limit=500,
            cleanup_interval=30,
            channel_timeout=120,
            log_socket_errors=True,
            expose_tracebacks=False,
        )
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Install with: pip install waitress flask flask-cors")
        return False
    except Exception as e:
        print(f"❌ Error starting production server: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run_production_server()