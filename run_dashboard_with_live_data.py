#!/usr/bin/env python3
"""
Enhanced CTI-sHARE Dashboard with Live Threat Data Initialization
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add src to path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir / 'src'))

def run_dashboard_with_live_data():
    """Run dashboard with current live threat data"""
    print("=" * 80)
    print("🛡️  CTI-sHARE Dashboard with Live Threat Intelligence")
    print("=" * 80)
    
    try:
        from src.dashboard.dashboard import ThreatDashboard
        from initialize_live_threat_data import initialize_dashboard_with_live_data
        
        print("📊 Initializing live threat data...")
        threats = initialize_dashboard_with_live_data()
        
        print("\n🚀 Starting CTI-sHARE Dashboard...")
        
        # Create dashboard instance
        dashboard = ThreatDashboard()
        
        # Load live threat data into dashboard
        print(f"📥 Loading {len(threats)} live threats into dashboard...")
        for threat in threats:
            dashboard.add_threat(threat)
        
        # Display dashboard statistics
        stats = dashboard.get_stats()
        print(f"\n📈 Dashboard Statistics:")
        print(f"   Total Threats: {stats.get('total_threats', 0)}")
        print(f"   Critical Threats: {len([t for t in threats if t.get('severity') == 'CRITICAL'])}")
        print(f"   High Threats: {len([t for t in threats if t.get('severity') == 'HIGH'])}")
        print(f"   Active Categories: {len(set(t.get('category') for t in threats))}")
        
        # Get recent threat activity
        recent_threats = sorted(threats, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
        
        print(f"\n🚨 Recent Threat Activity:")
        for threat in recent_threats[:5]:
            timestamp = datetime.fromisoformat(threat['timestamp'])
            hours_ago = int((datetime.now() - timestamp).total_seconds() / 3600)
            severity_emoji = "🔴" if threat['severity'] == 'CRITICAL' else "🟡" if threat['severity'] == 'HIGH' else "🟢"
            print(f"   {severity_emoji} {threat['title']} ({hours_ago}h ago)")
        
        print("\n" + "=" * 80)
        print("🌐 Dashboard Server Starting...")
        print("📡 URL: http://localhost:5001")
        print("⚡ Live Data: Active")
        print("🔄 Real-time Updates: Enabled")
        print("=" * 80)
        
        # Start the Flask application
        dashboard.app.run(debug=False, host='0.0.0.0', port=5001)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    run_dashboard_with_live_data()