#!/usr/bin/env python3
"""
Test script for Enhanced CTI-sHARE Dashboard with Train, Analyze, and Realtime features
"""

import sys
import os
import time
import requests
import threading

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_dashboard_api():
    """Test all dashboard API endpoints"""
    base_url = "http://localhost:5001"
    
    print("🧪 Testing Dashboard API Endpoints...")
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/api/dashboard/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check: PASS")
        else:
            print(f"❌ Health check: FAIL ({response.status_code})")
    except:
        print("❌ Health check: Server not running")
        return False
    
    # Test stats
    try:
        response = requests.get(f"{base_url}/api/dashboard/stats", timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard stats: PASS")
        else:
            print(f"❌ Dashboard stats: FAIL ({response.status_code})")
    except:
        print("❌ Dashboard stats: FAIL")
    
    # Test text analysis
    try:
        test_text = "Suspicious malware detected in network traffic with ransomware characteristics"
        response = requests.post(f"{base_url}/api/dashboard/analyze", 
                               json={"text": test_text}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                print("✅ Text analysis: PASS")
                analysis = data.get('analysis', {})
                print(f"   Category: {analysis.get('top_category', ['unknown', 0])[0]}")
                print(f"   Severity: {analysis.get('severity', {}).get('severity', 'unknown')}")
            else:
                print(f"❌ Text analysis: FAIL - {data.get('message', 'Unknown error')}")
        else:
            print(f"❌ Text analysis: FAIL ({response.status_code})")
    except Exception as e:
        print(f"❌ Text analysis: FAIL - {str(e)}")
    
    # Test model training
    try:
        print("🚀 Testing model training (this may take a moment)...")
        response = requests.post(f"{base_url}/api/dashboard/train", timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                print("✅ Model training: PASS")
                metrics = data.get('metrics', {})
                print(f"   Training accuracy: {metrics.get('train_accuracy', 0):.1%}")
            else:
                print(f"❌ Model training: FAIL - {data.get('message', 'Unknown error')}")
        else:
            print(f"❌ Model training: FAIL ({response.status_code})")
    except Exception as e:
        print(f"❌ Model training: FAIL - {str(e)}")
    
    # Test real-time detection
    try:
        # Start real-time
        response = requests.post(f"{base_url}/api/dashboard/realtime/start", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                print("✅ Real-time start: PASS")
                
                # Check status
                time.sleep(1)
                response = requests.get(f"{base_url}/api/dashboard/realtime/status", timeout=5)
                if response.status_code == 200:
                    status_data = response.json()
                    if status_data.get('is_running'):
                        print("✅ Real-time status check: PASS")
                    else:
                        print("❌ Real-time status check: Not running")
                
                # Stop real-time
                response = requests.post(f"{base_url}/api/dashboard/realtime/stop", timeout=10)
                if response.status_code == 200:
                    stop_data = response.json()
                    if stop_data.get('status') == 'success':
                        print("✅ Real-time stop: PASS")
                    else:
                        print(f"❌ Real-time stop: FAIL - {stop_data.get('message', 'Unknown error')}")
                else:
                    print(f"❌ Real-time stop: FAIL ({response.status_code})")
            else:
                print(f"❌ Real-time start: FAIL - {data.get('message', 'Unknown error')}")
        else:
            print(f"❌ Real-time start: FAIL ({response.status_code})")
    except Exception as e:
        print(f"❌ Real-time detection test: FAIL - {str(e)}")
    
    return True

def run_dashboard_server():
    """Run the enhanced dashboard server"""
    try:
        from src.dashboard.dashboard import ThreatDashboard
        from src.semantic_analysis.semantic_analyzer import ThreatSemanticAnalyzer
        from src.realtime.detector import RealTimeThreatDetector
        
        print("🛡️ Starting Enhanced CTI-sHARE Dashboard...")
        print("=" * 60)
        
        # Initialize components
        analyzer = ThreatSemanticAnalyzer()
        detector = RealTimeThreatDetector(semantic_analyzer=analyzer)
        
        # Create enhanced dashboard
        dashboard = ThreatDashboard(threat_detector=detector, semantic_analyzer=analyzer)
        
        # Add sample data
        sample_threats = [
            {
                'description': 'Advanced persistent threat detected in network',
                'category': 'apt',
                'severity': 'CRITICAL',
                'timestamp': '2025-10-22T10:00:00',
                'is_threat': True,
                'entities': {'ips': ['192.168.1.100']}
            },
            {
                'description': 'Phishing email with credential harvesting attempt',
                'category': 'phishing',
                'severity': 'HIGH',
                'timestamp': '2025-10-22T11:30:00',
                'is_threat': True,
                'entities': {'ips': ['10.0.0.50']}
            },
            {
                'description': 'Suspicious network traffic to known C&C server',
                'category': 'network_attack',
                'severity': 'HIGH',
                'timestamp': '2025-10-22T12:15:00',
                'is_threat': True,
                'entities': {'ips': ['203.0.113.1']}
            },
            {
                'description': 'Malware signature detected in file upload',
                'category': 'malware',
                'severity': 'MEDIUM',
                'timestamp': '2025-10-22T13:45:00',
                'is_threat': True,
                'entities': {'ips': ['172.16.0.25']}
            }
        ]
        
        for threat in sample_threats:
            dashboard.add_threat(threat)
        
        print(f"🌐 Enhanced Dashboard: http://localhost:5001")
        print(f"📊 Dashboard Features:")
        print(f"   • Model Training: /api/dashboard/train")
        print(f"   • Text Analysis: /api/dashboard/analyze")
        print(f"   • Real-time Control: /api/dashboard/realtime/*")
        print(f"   • Statistics: /api/dashboard/stats")
        print("=" * 60)
        print(f"Sample data loaded: {len(sample_threats)} threats")
        stats = dashboard._get_dashboard_stats()
        print(f"Total threats: {stats['total_threats']}")
        print(f"Critical threats: {stats['critical_threats']}")
        print(f"Detection rate: {stats['detection_rate']}%")
        print("=" * 60)
        print("🎯 New Features Available:")
        print("   🚀 Train ML models with real data")
        print("   🔍 Analyze text for threat indicators")
        print("   ⚡ Start/stop real-time threat detection")
        print("   📊 Monitor real-time statistics")
        print("=" * 60)
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Start the enhanced dashboard
        dashboard.run(host='127.0.0.1', port=5001, debug=True)
        
    except Exception as e:
        print(f"❌ Error starting enhanced dashboard: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Test mode - assumes server is already running
        test_dashboard_api()
    else:
        # Run the enhanced dashboard server
        run_dashboard_server()