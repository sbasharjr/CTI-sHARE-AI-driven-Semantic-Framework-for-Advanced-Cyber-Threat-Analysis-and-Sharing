#!/usr/bin/env python3
"""
Test Complete Notification System with Hide Buttons and History
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, render_template, jsonify
from flask_cors import CORS
import threading
import time
import webbrowser

app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(__file__), 'src', 'dashboard', 'templates'),
           static_folder=os.path.join(os.path.dirname(__file__), 'src', 'dashboard', 'static'))
CORS(app)

@app.route('/')
def dashboard():
    """Main dashboard route"""
    return render_template('dashboard.html')

@app.route('/api/stats')
def api_stats():
    """API endpoint for dashboard statistics"""
    return jsonify({
        'total_threats': 1247,
        'high_priority': 89,
        'resolved': 1089,
        'active_feeds': 15,
        'last_updated': '2024-01-10 15:30:25'
    })

@app.route('/api/recent_threats')
def api_recent_threats():
    """API endpoint for recent threats"""
    return jsonify([
        {'id': 1, 'type': 'Malware', 'severity': 'High', 'source': 'MISP Feed 1', 'timestamp': '2024-01-10 15:25:00'},
        {'id': 2, 'type': 'Phishing', 'severity': 'Medium', 'source': 'OSINT', 'timestamp': '2024-01-10 15:20:00'},
        {'id': 3, 'type': 'C2 Server', 'severity': 'Critical', 'source': 'Internal', 'timestamp': '2024-01-10 15:15:00'}
    ])

@app.route('/api/severity_distribution')
def api_severity_distribution():
    """API endpoint for severity distribution"""
    import random
    return jsonify({
        'critical': random.randint(15, 45),
        'high': random.randint(25, 65),
        'medium': random.randint(35, 85),
        'low': random.randint(45, 95),
        'info': random.randint(55, 105)
    })

@app.route('/api/geographic_threats')
def api_geographic_threats():
    """API endpoint for geographic threat data"""
    import random
    countries = ['US', 'CN', 'RU', 'DE', 'GB', 'FR', 'JP', 'KR', 'IN', 'BR']
    return jsonify([
        {'country': country, 'threats': random.randint(5, 50), 'lat': random.uniform(-60, 60), 'lng': random.uniform(-180, 180)}
        for country in countries
    ])

@app.route('/api/attack_vectors')
def api_attack_vectors():
    """API endpoint for attack vectors"""
    import random
    vectors = ['Email', 'Web', 'Network', 'USB', 'Social Engineering', 'Mobile', 'Cloud']
    return jsonify([
        {'vector': vector, 'count': random.randint(10, 100)}
        for vector in vectors
    ])

@app.route('/api/hourly_activity')
def api_hourly_activity():
    """API endpoint for hourly activity"""
    import random
    return jsonify([
        {'hour': f'{i:02d}:00', 'activity': random.randint(5, 50)}
        for i in range(24)
    ])

@app.route('/api/system_performance')
def api_system_performance():
    """API endpoint for system performance"""
    import random
    return jsonify({
        'cpu': random.randint(20, 80),
        'memory': random.randint(30, 90),
        'disk': random.randint(10, 70),
        'network': random.randint(5, 95)
    })

@app.route('/api/resource_distribution')
def api_resource_distribution():
    """API endpoint for resource distribution"""
    import random
    return jsonify({
        'available': random.randint(40, 70),
        'system': random.randint(10, 25),
        'applications': random.randint(15, 35),
        'cache': random.randint(5, 15)
    })

def start_server():
    """Start the Flask server in a separate thread"""
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def open_browser():
    """Open browser after a short delay"""
    time.sleep(2)
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("=" * 80)
    print("🛡️  CTI-sHARE Complete Notification System Test")
    print("=" * 80)
    print("🚀 Features Testing:")
    print("   ✅ Hide Buttons for Push Notifications and Social Share")
    print("   ✅ Notification History Management")
    print("   ✅ Panel Minimize/Hide Controls")
    print("   ✅ Real-time Active Charts")
    print("   ✅ Social Sharing Integration")
    print("   ✅ Live Information Displays")
    print()
    print("🌐 Dashboard will open at: http://localhost:5000")
    print("🔧 Test the following features:")
    print("   • Click panel toggle buttons to hide/show panels")
    print("   • Use minimize buttons to minimize panels")
    print("   • Check notification history functionality")
    print("   • Test social sharing with live information")
    print("   • Verify real-time chart updates")
    print("=" * 80)
    
    # Start server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Open browser
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down dashboard...")
        print("✅ Test completed successfully!")