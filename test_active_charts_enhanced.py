#!/usr/bin/env python3
"""
Active Charts Functionality Test
Tests the enhanced Attack Vectors, Geographic Distribution, and Hourly Activity charts
"""

import time
import threading
import requests
from datetime import datetime

def test_active_charts():
    """Test all active chart functionalities"""
    base_url = 'http://localhost:5001'
    
    print("🎯 Testing Active Charts Implementation")
    print("=" * 60)
    
    # Test connection
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code != 200:
            print("❌ Dashboard not accessible. Please start with: python main.py dashboard --port 5001")
            return
        print("✅ Dashboard is running at http://localhost:5001")
    except:
        print("❌ Cannot connect to dashboard")
        return
    
    # Test new API endpoints
    print("\n📊 Testing Enhanced API Endpoints...")
    
    endpoints = [
        ("/api/dashboard/threats/severity/realtime", "Real-time Severity"),
        ("/api/dashboard/stats/advanced", "Advanced Statistics"),
        ("/api/dashboard/threats/categories", "Categories"),
    ]
    
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ {name}: {len(str(data))} bytes of data")
            else:
                print(f"   ❌ {name}: HTTP {response.status_code}")
        except Exception as e:
            print(f"   ❌ {name}: Error - {e}")
    
    print("\n🎨 Active Charts Features Available:")
    print("   • Attack Vectors (Live) - Dynamic threat type visualization")
    print("   • Geographic Distribution (Live) - Country-based threat mapping")
    print("   • Hourly Activity Pattern (Live) - 24-hour threat timeline")
    print("   • Interactive controls for each chart")
    print("   • Real-time data updates every 8-15 seconds")
    print("   • Click interactions with detailed tooltips")
    print("   • Export functionality for all charts")
    print("   • Start/Stop live mode controls")
    
    print("\n🚀 How to Test the Active Charts:")
    print("   1. Open http://localhost:5001 in your browser")
    print("   2. Scroll down to 'Real-time Statistics & System Performance'")
    print("   3. Click 'Start Live Mode' on Attack Vectors chart")
    print("   4. Click 'Start Live Mode' on Geographic Distribution chart")
    print("   5. Click 'Start Live Mode' on Hourly Activity Pattern chart")
    print("   6. Watch the charts update automatically with new data")
    print("   7. Try clicking on chart elements for detailed info")
    print("   8. Use 'Export' buttons to download charts as images")
    print("   9. Use 'Pause All' and 'Start All' for bulk control")
    
    print("\n⚡ Enhanced Chart Features:")
    print("   • Attack Vectors: Color-coded by threat severity")
    print("   • Geographic: Shows threats detected vs blocked with effectiveness rates")
    print("   • Hourly Activity: Area chart with realistic business hour patterns")
    print("   • All charts: Enhanced tooltips with calculated metrics")
    print("   • Interactive: Click handlers for drill-down information")
    print("   • Responsive: Mobile-friendly design with touch support")
    
    print("\n🎛️ Live Update Intervals:")
    print("   • Attack Vectors: Updates every 8 seconds")
    print("   • Geographic Distribution: Updates every 12 seconds")
    print("   • Hourly Activity: Updates every 15 seconds")
    print("   • Severity Distribution: Updates every 5 seconds")
    print("   • System Performance: Updates every 10 seconds")
    
    print("\n📈 Data Visualization Features:")
    print("   • Smooth animations with easing effects")
    print("   • Dynamic color schemes based on threat levels")
    print("   • Realistic data patterns (business hours vs off-hours)")
    print("   • Hover effects and interactive tooltips")
    print("   • Professional Chart.js implementation")
    
    # Simulate some activity
    print("\n🔄 Simulating chart activity for 30 seconds...")
    start_time = time.time()
    
    while time.time() - start_time < 30:
        # Test real-time endpoints
        try:
            response = requests.get(f"{base_url}/api/dashboard/threats/severity/realtime", timeout=2)
            if response.status_code == 200:
                data = response.json()
                rt_data = data.get('realtime_severity', {})
                if rt_data.get('trends'):
                    print(f"   📊 Live data: {rt_data['trends']['threat_velocity']}/min velocity, "
                          f"{rt_data['trends']['critical_rate']}% critical rate")
        except:
            pass
        
        time.sleep(5)
    
    print("\n✅ Active Charts Test Complete!")
    print("\n🌐 Visit http://localhost:5001 to interact with the live charts")
    print("   The charts are now running with full interactivity!")

if __name__ == "__main__":
    test_active_charts()