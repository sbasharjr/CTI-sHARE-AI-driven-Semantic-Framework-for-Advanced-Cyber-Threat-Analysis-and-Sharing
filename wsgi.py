#!/usr/bin/env python3
"""
Production WSGI Application for CTI-sHARE Dashboard
"""

import os
import sys
import json
from pathlib import Path

# Get the directory where this file is located
current_dir = Path(__file__).resolve().parent

# Add project root to path
sys.path.insert(0, str(current_dir))

# Set environment variables for production
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('FLASK_DEBUG', '0')

def create_app():
    """Create and configure the Flask application for production"""
    try:
        from src.dashboard.dashboard import ThreatDashboard
        
        # Create dashboard instance
        dashboard = ThreatDashboard()
        
        # Add some sample threat data for demonstration
        sample_threats = [
            {
                'description': 'Advanced Persistent Threat (APT) activity detected in network traffic',
                'category': 'apt',
                'severity': 'CRITICAL',
                'timestamp': '2025-10-23T17:00:00',
                'is_threat': True,
                'entities': {
                    'ips': ['192.168.1.100', '10.0.0.15'],
                    'domains': ['malicious-domain.com'],
                    'urls': ['http://suspicious-site.net/payload']
                }
            },
            {
                'description': 'Malware signature detected in email attachments with ransomware characteristics',
                'category': 'malware',
                'severity': 'HIGH',
                'timestamp': '2025-10-23T16:45:00',
                'is_threat': True,
                'entities': {
                    'file_hashes': ['a1b2c3d4e5f6789012345678901234567890abcd'],
                    'emails': ['attacker@evil-domain.com']
                }
            },
            {
                'description': 'Phishing campaign targeting financial institutions detected',
                'category': 'phishing',
                'severity': 'HIGH',
                'timestamp': '2025-10-23T16:30:00',
                'is_threat': True,
                'entities': {
                    'domains': ['fake-bank-site.net', 'phishing-portal.org'],
                    'urls': ['https://fake-login.com/banking']
                }
            },
            {
                'description': 'Brute force attack detected on SSH services',
                'category': 'brute_force',
                'severity': 'MEDIUM',
                'timestamp': '2025-10-23T16:15:00',
                'is_threat': True,
                'entities': {
                    'ips': ['203.0.113.50', '198.51.100.75', '192.0.2.100']
                }
            },
            {
                'description': 'SQL injection attempt blocked on web application',
                'category': 'sql_injection',
                'severity': 'MEDIUM',
                'timestamp': '2025-10-23T16:00:00',
                'is_threat': True,
                'entities': {
                    'ips': ['203.0.113.25'],
                    'urls': ['http://webapp.example.com/login.php']
                }
            }
        ]
        
        # Load live threat data if available
        live_data_file = current_dir / 'live_threat_data.json'
        if live_data_file.exists():
            try:
                with open(live_data_file, 'r') as f:
                    live_threats = json.load(f)
                
                print(f"📥 Loading {len(live_threats)} live threats from file...")
                for threat in live_threats:
                    dashboard.add_threat(threat)
                
                print(f"✅ CTI-sHARE Dashboard initialized with {len(live_threats)} live threats")
            except Exception as e:
                print(f"⚠️ Error loading live threat data: {e}")
                # Fallback to sample data
                for threat in sample_threats:
                    dashboard.add_threat(threat)
                print("✅ CTI-sHARE Dashboard initialized with sample threat data")
        else:
            # Generate live threat data
            try:
                from initialize_live_threat_data import initialize_dashboard_with_live_data
                print("🔄 Generating fresh live threat data...")
                live_threats = initialize_dashboard_with_live_data()
                
                for threat in live_threats:
                    dashboard.add_threat(threat)
                
                print(f"✅ CTI-sHARE Dashboard initialized with {len(live_threats)} fresh live threats")
            except Exception as e:
                print(f"⚠️ Error generating live threat data: {e}")
                # Fallback to sample data
                for threat in sample_threats:
                    dashboard.add_threat(threat)
                print("✅ CTI-sHARE Dashboard initialized with sample threat data")
        
        return dashboard.app
        
    except Exception as e:
        print(f"❌ Error creating dashboard application: {e}")
        import traceback
        traceback.print_exc()
        raise

# Create the WSGI application
application = create_app()
app = application

if __name__ == "__main__":
    print("=" * 80)
    print("🛡️  CTI-sHARE Production Dashboard")
    print("=" * 80)
    print("⚠️  This is the WSGI application module.")
    print("💡 For development, use: python run_dashboard.py")
    print("🚀 For production, use: gunicorn wsgi:application")
    print("=" * 80)
    
    # Run in development mode if executed directly
    application.run(debug=False, host='0.0.0.0', port=5001)