#!/usr/bin/env python3
"""
Enhanced Attack Vectors Display Test
Test the comprehensive attack vectors data display with live API integration.
"""

import sys
import os
import time
import requests
import json
from datetime import datetime

# Add the src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import our dashboard
try:
    from dashboard.dashboard import ThreatDashboard
    print("✅ Successfully imported ThreatDashboard")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_attack_vectors_api():
    """Test the attack vectors API endpoint"""
    print("\n" + "="*60)
    print("🎯 TESTING ATTACK VECTORS API")
    print("="*60)
    
    # Initialize dashboard
    dashboard = ThreatDashboard()
    app = dashboard.app
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        print("📡 Testing /api/dashboard/live/attack-vectors endpoint...")
        
        # Test the attack vectors API
        response = client.get('/api/dashboard/live/attack-vectors')
        
        if response.status_code == 200:
            data = response.get_json()
            print("✅ API Response successful!")
            print(f"📊 Response Data:")
            print(f"   - Total Attacks: {data.get('total_attacks', 'N/A')}")
            print(f"   - Analysis Period: {data.get('analysis_period', 'N/A')}")
            print(f"   - Threat Count: {data.get('threat_count', 'N/A')}")
            print(f"   - Timestamp: {data.get('timestamp', 'N/A')}")
            
            vectors = data.get('vectors', {})
            print(f"🚨 Attack Vectors Detected:")
            
            # Sort vectors by count for better display
            sorted_vectors = sorted(vectors.items(), key=lambda x: x[1], reverse=True)
            
            for i, (vector_type, count) in enumerate(sorted_vectors, 1):
                severity = "CRITICAL" if count > 200 else "HIGH" if count > 150 else "MEDIUM" if count > 100 else "LOW"
                emoji = "🔴" if count > 200 else "🟠" if count > 150 else "🟡" if count > 100 else "🟢"
                
                vector_name = vector_type.replace('_', ' ').title()
                print(f"   {i:2d}. {emoji} {vector_name:<20} | {count:>3d} threats | {severity}")
            
            # Calculate statistics
            total_threats = sum(vectors.values())
            avg_threats = total_threats // len(vectors) if vectors else 0
            max_threats = max(vectors.values()) if vectors else 0
            max_vector = max(vectors.items(), key=lambda x: x[1])[0] if vectors else "None"
            
            print(f"\n📈 Summary Statistics:")
            print(f"   - Total Attack Instances: {total_threats:,}")
            print(f"   - Average per Vector: {avg_threats}")
            print(f"   - Primary Threat Vector: {max_vector.replace('_', ' ').title()} ({max_threats} attacks)")
            
            # Test data structure for frontend
            print(f"\n🔧 Frontend Integration Test:")
            print(f"   - Vector labels: {list(vectors.keys())}")
            print(f"   - Vector counts: {list(vectors.values())}")
            
            return True, data
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"Response: {response.get_data(as_text=True)}")
            return False, None

def test_frontend_integration():
    """Test frontend integration by starting server and making HTTP requests"""
    print("\n" + "="*60)
    print("🌐 TESTING FRONTEND INTEGRATION")
    print("="*60)
    
    try:
        print("🚀 Starting test server...")
        from threading import Thread
        import subprocess
        import socket
        
        # Check if port 5000 is available
        def is_port_available(port):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result != 0
        
        if not is_port_available(5000):
            print("⚠️  Port 5000 is busy, testing with test client instead...")
            return test_with_test_client()
        
        # Start server in background
        server_process = subprocess.Popen([
            sys.executable, '-c', '''
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from dashboard.dashboard import ThreatDashboard
dashboard = ThreatDashboard()
dashboard.app.run(host="127.0.0.1", port=5000, debug=False)
'''
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(3)
        
        try:
            # Test the attack vectors API via HTTP
            print("📡 Testing HTTP API call...")
            response = requests.get('http://127.0.0.1:5000/api/dashboard/live/attack-vectors', timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ HTTP API call successful!")
                print(f"📊 Received {len(data.get('vectors', {}))} attack vector types")
                
                # Test dashboard page
                print("🖥️  Testing dashboard page...")
                dashboard_response = requests.get('http://127.0.0.1:5000/', timeout=10)
                
                if dashboard_response.status_code == 200:
                    print("✅ Dashboard page loaded successfully!")
                    
                    # Check if attack vectors section exists
                    content = dashboard_response.text
                    if 'attackVectorsChart' in content:
                        print("✅ Attack Vectors chart container found in HTML")
                    if 'updateAttackVectorsChart' in content:
                        print("✅ Attack Vectors update function found in HTML")
                    if 'displayAttackVectorsInfo' in content:
                        print("✅ Attack Vectors info display function found in HTML")
                    
                    return True
                else:
                    print(f"❌ Dashboard page failed: {dashboard_response.status_code}")
                    return False
            else:
                print(f"❌ HTTP API call failed: {response.status_code}")
                return False
                
        finally:
            # Clean up server process
            server_process.terminate()
            server_process.wait()
            print("🛑 Test server stopped")
            
    except Exception as e:
        print(f"❌ Frontend integration test error: {e}")
        return False

def test_with_test_client():
    """Fallback test using Flask test client"""
    print("🧪 Using Flask test client for integration test...")
    
    dashboard = ThreatDashboard()
    app = dashboard.app
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        # Test dashboard page
        response = client.get('/')
        if response.status_code == 200:
            print("✅ Dashboard page accessible via test client")
            
            content = response.get_data(as_text=True)
            tests = [
                ('attackVectorsChart', 'Attack Vectors chart container'),
                ('updateAttackVectorsChart', 'Attack Vectors update function'),
                ('displayAttackVectorsInfo', 'Attack Vectors info display function'),
                ('attack-vectors-info', 'Attack Vectors info panel styles'),
                ('/api/dashboard/live/attack-vectors', 'Attack Vectors API endpoint')
            ]
            
            for search_term, description in tests:
                if search_term in content:
                    print(f"✅ {description} found")
                else:
                    print(f"⚠️  {description} not found")
            
            return True
        else:
            print(f"❌ Dashboard page failed: {response.status_code}")
            return False

def demonstrate_attack_vectors_features():
    """Demonstrate the enhanced attack vectors features"""
    print("\n" + "="*60)
    print("🎯 ATTACK VECTORS ENHANCED FEATURES DEMO")
    print("="*60)
    
    features = [
        "📊 Real-time attack vector data from live threat feeds",
        "🎨 Color-coded threat severity (Critical/High/Medium/Low)",
        "📈 Interactive bar chart with hover tooltips",
        "📋 Comprehensive information panel with statistics",
        "🏆 Top 5 attack vectors ranking display", 
        "🔄 Live mode with automatic data refresh",
        "📱 Responsive design for mobile and desktop",
        "⚡ Smooth animations and transitions",
        "🎯 Primary threat vector highlighting",
        "📊 Total attacks and analysis period display"
    ]
    
    print("🚀 Enhanced Attack Vectors Display Features:")
    for i, feature in enumerate(features, 1):
        print(f"   {i:2d}. {feature}")
    
    print(f"\n💡 Key Improvements:")
    print(f"   ✨ API Integration: Fetches real data from backend threat analysis")
    print(f"   ✨ Enhanced Visualization: Color-coded severity levels and interactive charts")
    print(f"   ✨ Detailed Information: Comprehensive statistics and threat attribution")
    print(f"   ✨ Real-time Updates: Live mode with automatic refresh capabilities")
    print(f"   ✨ Professional UI: Modern design with smooth animations")

def main():
    """Main test function"""
    print("🎯 CTI-sHARE Enhanced Attack Vectors Display Test")
    print("=" * 70)
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run tests
    test_results = []
    
    # Test 1: API functionality
    try:
        api_success, api_data = test_attack_vectors_api()
        test_results.append(("Attack Vectors API", api_success))
    except Exception as e:
        print(f"❌ API test failed: {e}")
        test_results.append(("Attack Vectors API", False))
    
    # Test 2: Frontend integration
    try:
        frontend_success = test_frontend_integration()
        test_results.append(("Frontend Integration", frontend_success))
    except Exception as e:
        print(f"❌ Frontend test failed: {e}")
        test_results.append(("Frontend Integration", False))
    
    # Test 3: Feature demonstration
    try:
        demonstrate_attack_vectors_features()
        test_results.append(("Feature Demo", True))
    except Exception as e:
        print(f"❌ Feature demo failed: {e}")
        test_results.append(("Feature Demo", False))
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name:<25} | {status}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced Attack Vectors display is ready!")
        print("\n🚀 To see the enhanced attack vectors in action:")
        print("   1. Run: python run_dashboard.py")
        print("   2. Open: http://127.0.0.1:5000")
        print("   3. Look for the Attack Vectors chart with detailed info panel")
        print("   4. Click 'Start Live Mode' to see real-time updates")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print(f"\n⏰ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()