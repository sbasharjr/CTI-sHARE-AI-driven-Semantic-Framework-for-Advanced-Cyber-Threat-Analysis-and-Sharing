#!/usr/bin/env python3
"""
Test script for CTI-sHARE Dashboard
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.dashboard.dashboard import ThreatDashboard
    print("✅ Successfully imported ThreatDashboard")
    
    # Create dashboard instance
    dashboard = ThreatDashboard()
    print("✅ Successfully created ThreatDashboard instance")
    
    # Add some test data
    dashboard.add_threat({
        'description': 'Test malware detected',
        'category': 'malware',
        'severity': 'HIGH',
        'timestamp': '2025-10-22T10:00:00',
        'is_threat': True
    })
    
    dashboard.add_threat({
        'description': 'Phishing email intercepted',
        'category': 'phishing',
        'severity': 'MEDIUM',
        'timestamp': '2025-10-22T11:30:00',
        'is_threat': True
    })
    
    print("✅ Added test threat data")
    
    # Test dashboard methods
    stats = dashboard._get_dashboard_stats()
    print(f"✅ Dashboard stats: {stats}")
    
    recent = dashboard._get_recent_threats(5)
    print(f"✅ Recent threats: {len(recent)} items")
    
    categories = dashboard._get_threat_by_category()
    print(f"✅ Categories: {categories}")
    
    print("\n🎉 Dashboard is ready to run!")
    print("To start the dashboard, run: python main.py dashboard --port 5001")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()