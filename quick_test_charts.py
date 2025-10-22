#!/usr/bin/env python3
"""
Quick test for active charts API endpoints
"""

try:
    import requests
    import json
    
    base_url = 'http://localhost:5001'
    
    print("Testing Active Charts API Endpoints...")
    print("=" * 50)
    
    # Test real-time severity endpoint
    print("\n1. Testing /api/dashboard/threats/severity/realtime")
    try:
        response = requests.get(f"{base_url}/api/dashboard/threats/severity/realtime", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Response keys: {list(data.keys())}")
            if 'realtime_severity' in data:
                rt_data = data['realtime_severity']
                print(f"   Real-time data keys: {list(rt_data.keys())}")
                print("   ✅ Real-time severity endpoint working!")
            else:
                print("   ❌ No realtime_severity key in response")
        else:
            print(f"   ❌ HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test advanced stats endpoint
    print("\n2. Testing /api/dashboard/stats/advanced")
    try:
        response = requests.get(f"{base_url}/api/dashboard/stats/advanced", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Response keys: {list(data.keys())}")
            if 'advanced_stats' in data:
                adv_data = data['advanced_stats']
                print(f"   Advanced stats keys: {list(adv_data.keys())}")
                print("   ✅ Advanced statistics endpoint working!")
            else:
                print("   ❌ No advanced_stats key in response")
        else:
            print(f"   ❌ HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test basic dashboard health
    print("\n3. Testing basic dashboard health")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"   Dashboard status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Dashboard is running!")
        else:
            print("   ❌ Dashboard not responding properly")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Test completed!")
    print("🌐 Open http://localhost:5001 to see the active charts!")
    print("\nNew Features Added:")
    print("• Active Severity Distribution with real-time updates")
    print("• Real-time Statistics with system performance metrics")
    print("• Interactive charts using Chart.js")
    print("• Live data updates every 5-10 seconds")
    print("• Export functionality for all charts")
    print("• Geographic distribution visualization")
    print("• Attack vectors analysis")
    print("• Hourly activity patterns")

except ImportError:
    print("❌ requests module not available")
except Exception as e:
    print(f"❌ Unexpected error: {e}")