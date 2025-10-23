#!/usr/bin/env python3
"""
Test script for new Live Data API endpoints
"""

import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_live_apis():
    """Test the new live data API endpoints"""
    print("=" * 60)
    print("🔴 Testing Live Data API Endpoints")
    print("=" * 60)
    
    try:
        from src.dashboard.dashboard import ThreatDashboard
        
        # Create dashboard instance
        dashboard = ThreatDashboard()
        app = dashboard.app
        
        with app.test_client() as client:
            # Test System Performance API
            print("\n🖥️ Testing System Performance API...")
            response = client.get('/api/dashboard/live/system-performance')
            if response.status_code == 200:
                data = response.get_json()
                print(f"✅ System Performance: CPU:{data.get('cpu_usage')}% MEM:{data.get('memory_usage')}%")
            else:
                print(f"❌ System Performance: {response.status_code}")
            
            # Test Resource Distribution API
            print("\n📊 Testing Resource Distribution API...")
            response = client.get('/api/dashboard/live/resource-distribution')
            if response.status_code == 200:
                data = response.get_json()
                print(f"✅ Resource Distribution: Available:{data.get('available')}% System:{data.get('system')}%")
            else:
                print(f"❌ Resource Distribution: {response.status_code}")
            
            # Test Attack Vectors API
            print("\n🎯 Testing Attack Vectors API...")
            response = client.get('/api/dashboard/live/attack-vectors')
            if response.status_code == 200:
                data = response.get_json()
                vectors = data.get('vectors', {})
                print(f"✅ Attack Vectors: Total:{data.get('total_attacks')} Malware:{vectors.get('malware')}")
            else:
                print(f"❌ Attack Vectors: {response.status_code}")
            
            # Test Geographic Distribution API
            print("\n🌍 Testing Geographic Distribution API...")
            response = client.get('/api/dashboard/live/geographic-distribution')
            if response.status_code == 200:
                data = response.get_json()
                print(f"✅ Geographic Distribution: Total:{data.get('total_threats')} Top:{data.get('top_region')}")
            else:
                print(f"❌ Geographic Distribution: {response.status_code}")
            
            # Test Hourly Activity API
            print("\n⏰ Testing Hourly Activity API...")
            response = client.get('/api/dashboard/live/hourly-activity')
            if response.status_code == 200:
                data = response.get_json()
                print(f"✅ Hourly Activity: Current:{data.get('current_hour')} Peak:{data.get('peak_hour')} Total:{data.get('total_today')}")
            else:
                print(f"❌ Hourly Activity: {response.status_code}")
            
            # Test Feed Status API
            print("\n📡 Testing Feed Status API...")
            response = client.get('/api/dashboard/live/feed-status')
            if response.status_code == 200:
                data = response.get_json()
                print(f"✅ Feed Status: Health:{data.get('overall_health')}% Active:{data.get('active_feeds')}/{data.get('total_feeds')}")
            else:
                print(f"❌ Feed Status: {response.status_code}")
            
            # Test IOC Types API
            print("\n🔍 Testing IOC Types API...")
            response = client.get('/api/dashboard/live/ioc-types')
            if response.status_code == 200:
                data = response.get_json()
                print(f"✅ IOC Types: Total:{data.get('total_iocs')} Most Common:{data.get('most_common')}")
            else:
                print(f"❌ IOC Types: {response.status_code}")
        
        print("\n" + "=" * 60)
        print("✅ All Live Data API Tests Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_live_apis()