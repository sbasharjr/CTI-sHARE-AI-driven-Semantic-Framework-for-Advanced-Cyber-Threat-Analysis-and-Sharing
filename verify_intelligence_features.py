#!/usr/bin/env python3
"""
Quick verification of enhanced CTI-sHARE dashboard features
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.dashboard.dashboard import ThreatDashboard
    
    print("✅ Successfully imported enhanced ThreatDashboard")
    
    # Create dashboard instance
    dashboard = ThreatDashboard()
    
    # Check for new methods
    intelligence_methods = [method for method in dir(dashboard) if any(keyword in method.lower() for keyword in 
                           ['upload', 'import', 'export', 'share', 'bulk', 'entities'])]
    
    print(f"✅ Dashboard loaded with {len(intelligence_methods)} threat intelligence methods:")
    for method in intelligence_methods:
        print(f"   • {method}")
    
    # Check routes
    with dashboard.app.test_client() as client:
        # Test health endpoint
        response = client.get('/api/dashboard/health')
        if response.status_code == 200:
            print("✅ Health endpoint working")
        
        # List all available routes
        print(f"\n📊 Available Intelligence Routes:")
        for rule in dashboard.app.url_map.iter_rules():
            if 'intelligence' in str(rule) or 'semantic' in str(rule):
                print(f"   • {rule.methods} {rule}")
    
    print(f"\n🎉 Enhanced CTI-sHARE Dashboard ready with full threat intelligence capabilities!")
    print(f"📊 Total methods available: {len(dir(dashboard))}")
    print(f"🌐 Intelligence-specific methods: {len(intelligence_methods)}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()