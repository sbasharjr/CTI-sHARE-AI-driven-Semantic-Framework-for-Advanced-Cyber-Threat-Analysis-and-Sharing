#!/usr/bin/env python3
"""
Direct Flask Dashboard Test
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.dashboard.dashboard import ThreatDashboard

if __name__ == '__main__':
    print("=" * 60)
    print("🛡️  CTI-sHARE Dashboard Starting...")
    print("=" * 60)
    
    try:
        # Create dashboard
        dashboard = ThreatDashboard()
        
        # Add some sample data
        dashboard.add_threat({
            'description': 'Malware detected in network traffic',
            'category': 'malware',
            'severity': 'HIGH',
            'timestamp': '2025-10-22T10:00:00',
            'is_threat': True,
            'entities': {'ips': ['192.168.1.100']}
        })
        
        dashboard.add_threat({
            'description': 'Suspicious phishing email intercepted',
            'category': 'phishing',
            'severity': 'MEDIUM',
            'timestamp': '2025-10-22T11:30:00',
            'is_threat': True,
            'entities': {'ips': ['10.0.0.50']}
        })
        
        dashboard.add_threat({
            'description': 'DDoS attack attempt blocked',
            'category': 'network_attack',
            'severity': 'CRITICAL',
            'timestamp': '2025-10-22T12:15:00',
            'is_threat': True,
            'entities': {'ips': ['203.0.113.1']}
        })
        
        print(f"🌐 Dashboard will be available at: http://localhost:5001")
        print(f"📊 API endpoints available at: http://localhost:5001/api/dashboard/*")
        print("=" * 60)
        print("Sample data loaded:")
        stats = dashboard._get_dashboard_stats()
        print(f"  Total Threats: {stats['total_threats']}")
        print(f"  Critical Threats: {stats['critical_threats']}")
        print(f"  Detection Rate: {stats['detection_rate']}%")
        print("=" * 60)
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Start the dashboard
        dashboard.run(host='127.0.0.1', port=5001, debug=True)
        
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        import traceback
        traceback.print_exc()