#!/usr/bin/env python3
"""
Simple Flask application for CTI-sHARE with MISP Integration
"""

from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import json
from datetime import datetime
import os
import sys

# Import MISP integration
try:
    from misp_integration import MISPIntegration, create_misp_api_endpoints
    MISP_AVAILABLE = True
except ImportError:
    print("⚠️  MISP integration not available - misp_integration.py not found")
    MISP_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Simple in-memory storage for threats
threats_db = []

# HTML template for simple dashboard
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>CTI-sHARE Threat Intelligence Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; }
        .card { background: white; padding: 20px; border-radius: 10px; 
               box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
        .stat-box { text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; }
        .stat-number { font-size: 2em; font-weight: bold; color: #667eea; }
        .endpoint { background: #e8f4fd; padding: 10px; margin: 5px 0; border-radius: 5px; font-family: monospace; }
        .threat-item { background: #fff3cd; padding: 10px; margin: 5px 0; border-left: 4px solid #ffc107; }
        button { background: #667eea; color: white; border: none; padding: 10px 20px; 
                border-radius: 5px; cursor: pointer; margin: 5px; }
        button:hover { background: #5a67d8; }
        input[type="text"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin: 5px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ CTI-sHARE Threat Intelligence Dashboard</h1>
        <p>AI-driven Semantic Framework for Advanced Cyber Threat Analysis and Sharing</p>
    </div>

    <div class="stats">
        <div class="stat-box">
            <div class="stat-number" id="threatCount">0</div>
            <div>Total Threats</div>
        </div>
        <div class="stat-box">
            <div class="stat-number" id="apiCalls">0</div>
            <div>API Calls</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">Active</div>
            <div>System Status</div>
        </div>
    </div>

    <div class="card">
        <h2>🔗 Available API Endpoints</h2>
        <div class="endpoint">GET /api/health - Health check</div>
        <div class="endpoint">POST /api/threats - Submit new threat</div>
        <div class="endpoint">GET /api/threats - Get all threats</div>
        <div class="endpoint">GET /api/stats - Get statistics</div>
        <div class="endpoint">POST /api/analyze - Analyze threat text</div>
    </div>

    <div class="card">
        <h2>➕ Submit New Threat</h2>
        <input type="text" id="threatText" placeholder="Enter threat description...">
        <button onclick="submitThreat()">Submit Threat</button>
    </div>

    <div class="card">
        <h2>📊 Recent Threats</h2>
        <div id="threatsList">
            <p>No threats submitted yet.</p>
        </div>
        <button onclick="loadThreats()">Refresh Threats</button>
    </div>

    <script>
        let apiCallCount = 0;

        function updateStats() {
            document.getElementById('apiCalls').textContent = apiCallCount;
        }

        function submitThreat() {
            const threatText = document.getElementById('threatText').value;
            if (!threatText.trim()) {
                alert('Please enter a threat description');
                return;
            }

            fetch('/api/threats', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    description: threatText,
                    timestamp: new Date().toISOString(),
                    source: 'Dashboard'
                })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('threatText').value = '';
                loadThreats();
                apiCallCount++;
                updateStats();
            })
            .catch(error => console.error('Error:', error));
        }

        function loadThreats() {
            fetch('/api/threats')
            .then(response => response.json())
            .then(data => {
                const threatsList = document.getElementById('threatsList');
                document.getElementById('threatCount').textContent = data.threats.length;
                
                if (data.threats.length === 0) {
                    threatsList.innerHTML = '<p>No threats submitted yet.</p>';
                } else {
                    threatsList.innerHTML = data.threats.map((threat, index) => 
                        `<div class="threat-item">
                            <strong>Threat ${index + 1}:</strong> ${threat.description}<br>
                            <small>Time: ${threat.timestamp} | Source: ${threat.source}</small>
                        </div>`
                    ).join('');
                }
                apiCallCount++;
                updateStats();
            })
            .catch(error => console.error('Error:', error));
        }

        // Load threats on page load
        loadThreats();
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'CTI-sHARE Threat Intelligence API',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/threats', methods=['GET'])
def get_threats():
    """Get all threats"""
    return jsonify({
        'threats': threats_db,
        'count': len(threats_db),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/threats', methods=['POST'])
def submit_threat():
    """Submit a new threat"""
    try:
        data = request.get_json()
        
        if not data or 'description' not in data:
            return jsonify({'error': 'Missing threat description'}), 400
        
        threat = {
            'id': len(threats_db) + 1,
            'description': data['description'],
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'source': data.get('source', 'API'),
            'severity': data.get('severity', 'medium'),
            'category': data.get('category', 'unknown')
        }
        
        threats_db.append(threat)
        
        return jsonify({
            'message': 'Threat submitted successfully',
            'threat_id': threat['id'],
            'timestamp': threat['timestamp']
        }), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_threat():
    """Analyze threat text (simplified version)"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text to analyze'}), 400
        
        text = data['text'].lower()
        
        # Simple threat analysis based on keywords
        malware_keywords = ['virus', 'malware', 'trojan', 'ransomware', 'worm']
        phishing_keywords = ['phishing', 'email', 'credential', 'login', 'password']
        network_keywords = ['ddos', 'botnet', 'intrusion', 'network', 'firewall']
        
        category = 'unknown'
        severity = 'low'
        
        if any(keyword in text for keyword in malware_keywords):
            category = 'malware'
            severity = 'high'
        elif any(keyword in text for keyword in phishing_keywords):
            category = 'phishing'
            severity = 'medium'
        elif any(keyword in text for keyword in network_keywords):
            category = 'network_attack'
            severity = 'high'
        
        confidence = 0.75 if category != 'unknown' else 0.25
        
        return jsonify({
            'analysis': {
                'category': category,
                'severity': severity,
                'confidence': confidence,
                'keywords_found': [kw for kw in malware_keywords + phishing_keywords + network_keywords if kw in text]
            },
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    total_threats = len(threats_db)
    
    # Count by category
    categories = {}
    severities = {}
    
    for threat in threats_db:
        cat = threat.get('category', 'unknown')
        sev = threat.get('severity', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
        severities[sev] = severities.get(sev, 0) + 1
    
    return jsonify({
        'total_threats': total_threats,
        'categories': categories,
        'severities': severities,
        'system_status': 'active',
        'timestamp': datetime.now().isoformat()
    })

# ===================================
# MISP INTEGRATION API ENDPOINTS
# ===================================

@app.route('/api/dashboard/misp/test-connection', methods=['POST'])
def test_misp_connection():
    """Test MISP server connection"""
    if not MISP_AVAILABLE:
        return jsonify({
            'status': 'error',
            'message': 'MISP integration not available'
        }), 503
    
    try:
        data = request.get_json()
        server_url = data.get('server_url')
        api_key = data.get('api_key')
        organization = data.get('organization', '')
        
        misp = MISPIntegration(server_url, api_key, organization)
        result = misp.test_connection()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/api/dashboard/misp/import-events', methods=['POST'])
def import_misp_events():
    """Import events from MISP"""
    if not MISP_AVAILABLE:
        return jsonify({
            'status': 'error',
            'message': 'MISP integration not available'
        }), 503
    
    try:
        data = request.get_json()
        server_url = data.get('serverUrl')
        api_key = data.get('apiKey')
        organization = data.get('organization', '')
        
        misp = MISPIntegration(server_url, api_key, organization)
        result = misp.import_events(
            days_back=data.get('days_back', 30),
            limit=data.get('limit', 100)
        )
        
        # Add imported events to local threats database
        if result['status'] == 'success':
            for event in result.get('events', []):
                threat = {
                    'id': f"misp_{event['id']}",
                    'type': 'event',
                    'category': 'malware',
                    'severity': 'high' if event.get('threat_level', 3) <= 2 else 'medium',
                    'description': event.get('info', 'MISP Event'),
                    'source': 'MISP',
                    'timestamp': datetime.now().isoformat(),
                    'misp_data': event
                }
                threats_db.append(threat)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/api/dashboard/misp/import-attributes', methods=['POST'])
def import_misp_attributes():
    """Import attributes from MISP"""
    if not MISP_AVAILABLE:
        return jsonify({
            'status': 'error',
            'message': 'MISP integration not available'
        }), 503
    
    try:
        data = request.get_json()
        server_url = data.get('serverUrl')
        api_key = data.get('apiKey')
        organization = data.get('organization', '')
        
        misp = MISPIntegration(server_url, api_key, organization)
        
        filter_data = data.get('filter', {})
        result = misp.import_attributes(
            categories=filter_data.get('category', None)
        )
        
        # Add imported attributes to local threats database
        if result['status'] == 'success':
            for attr in result.get('attributes', []):
                threat = {
                    'id': f"misp_attr_{attr['id']}",
                    'type': 'attribute',
                    'category': attr.get('category', 'other'),
                    'severity': 'high' if attr.get('to_ids') else 'medium',
                    'description': f"{attr.get('type', '')}: {attr.get('value', '')}",
                    'source': 'MISP',
                    'timestamp': datetime.now().isoformat(),
                    'misp_data': attr
                }
                threats_db.append(threat)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/api/dashboard/misp/import-iocs', methods=['POST'])
def import_misp_iocs():
    """Import IOCs from MISP"""
    if not MISP_AVAILABLE:
        return jsonify({
            'status': 'error',
            'message': 'MISP integration not available'
        }), 503
    
    try:
        data = request.get_json()
        server_url = data.get('serverUrl')
        api_key = data.get('apiKey')
        organization = data.get('organization', '')
        
        misp = MISPIntegration(server_url, api_key, organization)
        result = misp.import_iocs(
            ioc_types=data.get('ioc_types', None),
            confidence_threshold=data.get('confidence_threshold', 70)
        )
        
        # Add imported IOCs to local threats database
        if result['status'] == 'success':
            for ioc in result.get('iocs', []):
                threat = {
                    'id': f"misp_ioc_{ioc['id']}",
                    'type': 'ioc',
                    'category': ioc.get('category', 'indicator'),
                    'severity': 'critical' if ioc.get('confidence', 0) > 90 else 'high',
                    'description': f"{ioc.get('type', '')}: {ioc.get('value', '')}",
                    'source': 'MISP',
                    'timestamp': datetime.now().isoformat(),
                    'misp_data': ioc
                }
                threats_db.append(threat)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/api/dashboard/misp/export', methods=['POST'])
def export_to_misp():
    """Export data to MISP"""
    if not MISP_AVAILABLE:
        return jsonify({
            'status': 'error',
            'message': 'MISP integration not available'
        }), 503
    
    try:
        data = request.get_json()
        server_url = data.get('serverUrl')
        api_key = data.get('apiKey')
        organization = data.get('organization', '')
        
        misp = MISPIntegration(server_url, api_key, organization)
        
        # Prepare export data from local threats
        export_options = data.get('export_options', {})
        export_data = {
            'threats': [],
            'iocs': [],
            'analysis': {},
            'threat_level': export_options.get('threat_level', 3),
            'distribution': export_options.get('distribution', 1)
        }
        
        # Convert local threats to MISP format
        for threat in threats_db:
            if threat.get('source') != 'MISP':  # Don't re-export MISP data
                export_item = {
                    'type': 'text',
                    'category': 'Other',
                    'value': threat.get('description', ''),
                    'comment': f"Exported from CTI-sHARE: {threat.get('type', '')}",
                    'to_ids': False
                }
                export_data['threats'].append(export_item)
        
        result = misp.export_to_misp(
            export_data,
            create_event=export_options.get('create_event', True)
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/api/dashboard/misp/sync', methods=['POST'])
def sync_misp_data():
    """Synchronize data with MISP"""
    if not MISP_AVAILABLE:
        return jsonify({
            'status': 'error',
            'message': 'MISP integration not available'
        }), 503
    
    try:
        data = request.get_json()
        server_url = data.get('serverUrl')
        api_key = data.get('apiKey')
        organization = data.get('organization', '')
        
        misp = MISPIntegration(server_url, api_key, organization)
        
        sync_options = data.get('sync_options', {})
        result = misp.sync_data(
            bidirectional=sync_options.get('bidirectional', True)
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🛡️  CTI-sHARE Threat Intelligence API with MISP Integration")
    print("=" * 60)
    print(f"🌐 Dashboard: http://localhost:5000")
    print(f"🔗 API Base URL: http://localhost:5000/api")
    print("=" * 60)
    print("Available endpoints:")
    print("  GET  / - Dashboard")
    print("  GET  /api/health - Health check")
    print("  POST /api/threats - Submit threat")
    print("  GET  /api/threats - Get threats")
    print("  POST /api/analyze - Analyze threat")
    print("  GET  /api/stats - Get statistics")
    print("=" * 60)
    if MISP_AVAILABLE:
        print("🔗 MISP Integration endpoints:")
        print("  POST /api/dashboard/misp/test-connection - Test MISP connection")
        print("  POST /api/dashboard/misp/import-events - Import MISP events")
        print("  POST /api/dashboard/misp/import-attributes - Import MISP attributes")
        print("  POST /api/dashboard/misp/import-iocs - Import MISP IOCs")
        print("  POST /api/dashboard/misp/export - Export to MISP")
        print("  POST /api/dashboard/misp/sync - Sync with MISP")
        print("=" * 60)
        print("✅ MISP Framework Integration: ENABLED")
    else:
        print("⚠️  MISP Framework Integration: DISABLED")
        print("   (Install PyMISP: pip install pymisp)")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)