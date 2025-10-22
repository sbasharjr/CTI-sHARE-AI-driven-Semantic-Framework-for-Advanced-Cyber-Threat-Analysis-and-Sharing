#!/usr/bin/env python3
"""
Active Charts Testing Script for CTI-sHARE Dashboard
Tests and demonstrates the active severity distribution and real-time statistics charts
"""

import requests
import json
import time
import threading
from datetime import datetime

class ActiveChartsTest:
    def __init__(self, base_url='http://localhost:5001'):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_connection(self):
        """Test if dashboard is running"""
        try:
            response = self.session.get(f"{self.base_url}/api/dashboard/health")
            if response.status_code == 200:
                print("✅ Dashboard is running")
                return True
            else:
                print(f"❌ Dashboard returned status code: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to dashboard. Please start it with: python main.py dashboard --port 5001")
            return False
        except Exception as e:
            print(f"❌ Error connecting to dashboard: {e}")
            return False
    
    def test_severity_endpoints(self):
        """Test the new severity distribution endpoints"""
        print("\n🔍 Testing Severity Distribution Endpoints...")
        
        # Test standard severity endpoint
        try:
            response = self.session.get(f"{self.base_url}/api/dashboard/threats/severity")
            if response.status_code == 200:
                data = response.json()
                print("✅ Standard severity endpoint working")
                print(f"   Data: {data}")
            else:
                print(f"❌ Standard severity endpoint failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Error testing standard severity endpoint: {e}")
        
        # Test real-time severity endpoint
        try:
            response = self.session.get(f"{self.base_url}/api/dashboard/threats/severity/realtime")
            if response.status_code == 200:
                data = response.json()
                print("✅ Real-time severity endpoint working")
                print(f"   Current distribution: {data.get('realtime_severity', {}).get('current_distribution', {})}")
                print(f"   Time series entries: {len(data.get('realtime_severity', {}).get('time_series', []))}")
                print(f"   Trends: {data.get('realtime_severity', {}).get('trends', {})}")
            else:
                print(f"❌ Real-time severity endpoint failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Error testing real-time severity endpoint: {e}")
    
    def test_advanced_stats_endpoint(self):
        """Test the advanced statistics endpoint"""
        print("\n📊 Testing Advanced Statistics Endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/api/dashboard/stats/advanced")
            if response.status_code == 200:
                data = response.json()
                advanced_stats = data.get('advanced_stats', {})
                print("✅ Advanced statistics endpoint working")
                
                # Check data structure
                if 'system_performance' in advanced_stats:
                    print(f"   System performance: CPU {advanced_stats['system_performance']['cpu_usage']}%, "
                          f"Memory {advanced_stats['system_performance']['memory_usage']}%")
                
                if 'geographic_distribution' in advanced_stats:
                    geo_count = len(advanced_stats['geographic_distribution'])
                    print(f"   Geographic data: {geo_count} countries")
                
                if 'attack_vectors' in advanced_stats:
                    vector_count = len(advanced_stats['attack_vectors'])
                    print(f"   Attack vectors: {vector_count} types")
                
                if 'hourly_patterns' in advanced_stats:
                    hourly_count = len(advanced_stats['hourly_patterns'])
                    print(f"   Hourly patterns: {hourly_count} data points")
                
            else:
                print(f"❌ Advanced statistics endpoint failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Error testing advanced statistics endpoint: {e}")
    
    def simulate_threat_data_uploads(self, count=5):
        """Simulate uploading threat intelligence to generate chart data"""
        print(f"\n📤 Simulating {count} threat intelligence uploads...")
        
        sample_threats = [
            {
                "format": "json",
                "data": json.dumps({
                    "threat_type": "malware",
                    "severity": "CRITICAL",
                    "description": "Advanced persistent threat detected",
                    "iocs": ["192.168.1.100", "malicious.example.com"],
                    "source": "automated_analysis"
                })
            },
            {
                "format": "json", 
                "data": json.dumps({
                    "threat_type": "phishing",
                    "severity": "HIGH",
                    "description": "Phishing campaign targeting financial institutions",
                    "iocs": ["phishing.badsite.com", "fake-bank-login.evil.com"],
                    "source": "threat_feed"
                })
            },
            {
                "format": "json",
                "data": json.dumps({
                    "threat_type": "ransomware",
                    "severity": "CRITICAL",
                    "description": "Ransomware family with new encryption method",
                    "iocs": ["ransom-c2.darkweb.com", "pay-bitcoin-here.onion"],
                    "source": "incident_response"
                })
            },
            {
                "format": "text",
                "data": "DDoS attack from botnet detected targeting e-commerce sites during peak hours"
            },
            {
                "format": "text", 
                "data": "SQL injection attempts detected against multiple web applications using automated tools"
            }
        ]
        
        uploaded = 0
        for i, threat in enumerate(sample_threats[:count]):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/dashboard/intelligence/upload",
                    json=threat
                )
                if response.status_code == 200:
                    uploaded += 1
                    print(f"✅ Uploaded threat {i+1}/{count}")
                else:
                    print(f"❌ Failed to upload threat {i+1}: {response.status_code}")
                time.sleep(1)  # Space out uploads
            except Exception as e:
                print(f"❌ Error uploading threat {i+1}: {e}")
        
        print(f"📊 Successfully uploaded {uploaded}/{count} threat intelligence entries")
        return uploaded > 0
    
    def monitor_realtime_updates(self, duration=60):
        """Monitor real-time chart updates for a specified duration"""
        print(f"\n🔄 Monitoring real-time chart updates for {duration} seconds...")
        print("   (This simulates what the dashboard charts will show)")
        
        start_time = time.time()
        update_count = 0
        
        while time.time() - start_time < duration:
            try:
                # Test real-time severity updates
                response = self.session.get(f"{self.base_url}/api/dashboard/threats/severity/realtime")
                if response.status_code == 200:
                    data = response.json()
                    realtime_data = data.get('realtime_severity', {})
                    
                    if realtime_data:
                        trends = realtime_data.get('trends', {})
                        alerts = realtime_data.get('alerts', {})
                        
                        print(f"⏱️  Update {update_count + 1} at {datetime.now().strftime('%H:%M:%S')}")
                        print(f"   Critical Rate: {trends.get('critical_rate', 0)}%")
                        print(f"   Threat Velocity: {trends.get('threat_velocity', 0)}/min")
                        print(f"   Active Incidents: {alerts.get('active_incidents', 0)}")
                        print(f"   Blocked Attacks: {alerts.get('blocked_attacks', 0)}")
                        
                        update_count += 1
                
                # Test advanced stats updates
                response = self.session.get(f"{self.base_url}/api/dashboard/stats/advanced")
                if response.status_code == 200:
                    data = response.json()
                    system_perf = data.get('advanced_stats', {}).get('system_performance', {})
                    
                    if system_perf:
                        print(f"   System CPU: {system_perf.get('cpu_usage', 0)}%")
                        print(f"   Memory: {system_perf.get('memory_usage', 0)}%")
                        print("   " + "-" * 40)
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                print(f"❌ Error during monitoring: {e}")
                break
        
        print(f"✅ Monitoring completed. Total updates: {update_count}")
    
    def test_chart_data_quality(self):
        """Test the quality and structure of chart data"""
        print("\n🎯 Testing Chart Data Quality...")
        
        endpoints = [
            ("/api/dashboard/threats/categories", "categories"),
            ("/api/dashboard/threats/severity/realtime", "realtime_severity"),
            ("/api/dashboard/stats/advanced", "advanced_stats")
        ]
        
        for endpoint, data_key in endpoints:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}")
                if response.status_code == 200:
                    data = response.json()
                    chart_data = data.get(data_key, {})
                    
                    print(f"✅ {endpoint}")
                    
                    # Validate data structure based on endpoint
                    if endpoint == "/api/dashboard/threats/categories":
                        if isinstance(chart_data, dict) and len(chart_data) > 0:
                            print(f"   Categories count: {len(chart_data)}")
                        else:
                            print("   ⚠️  No category data available")
                    
                    elif endpoint == "/api/dashboard/threats/severity/realtime":
                        required_keys = ['current_distribution', 'time_series', 'trends', 'alerts']
                        missing_keys = [key for key in required_keys if key not in chart_data]
                        if not missing_keys:
                            print("   All required keys present")
                            time_series_count = len(chart_data.get('time_series', []))
                            print(f"   Time series points: {time_series_count}")
                        else:
                            print(f"   ⚠️  Missing keys: {missing_keys}")
                    
                    elif endpoint == "/api/dashboard/stats/advanced":
                        required_sections = ['system_performance', 'geographic_distribution', 'attack_vectors', 'hourly_patterns']
                        available_sections = [sec for sec in required_sections if sec in chart_data]
                        print(f"   Available sections: {len(available_sections)}/{len(required_sections)}")
                        
                else:
                    print(f"❌ {endpoint} failed: {response.status_code}")
            except Exception as e:
                print(f"❌ Error testing {endpoint}: {e}")
    
    def run_comprehensive_test(self):
        """Run all tests"""
        print("🚀 CTI-sHARE Active Charts Comprehensive Test")
        print("=" * 60)
        
        # Test connection first
        if not self.test_connection():
            return False
        
        # Run all tests
        self.test_severity_endpoints()
        self.test_advanced_stats_endpoint()
        self.test_chart_data_quality()
        
        # Simulate data uploads to populate charts
        if self.simulate_threat_data_uploads(3):
            print("\n⏳ Waiting 5 seconds for data to be processed...")
            time.sleep(5)
            
            # Re-test to show updated data
            print("\n🔄 Re-testing endpoints with uploaded data...")
            self.test_severity_endpoints()
        
        # Monitor real-time updates
        monitor_choice = input("\n🤖 Would you like to monitor real-time chart updates for 60 seconds? (y/n): ")
        if monitor_choice.lower() == 'y':
            self.monitor_realtime_updates(60)
        
        print("\n✅ All tests completed!")
        print("\n🌐 Open your browser to http://localhost:5001 to see the active charts in action!")
        print("\nFeatures to test in the browser:")
        print("   • Click 'Start Real-time' on Severity Distribution")
        print("   • Click 'Start Monitoring' on Real-time Statistics")
        print("   • Watch charts update automatically")
        print("   • Export charts as PNG images")
        print("   • Upload threat intelligence to see live data")
        
        return True

if __name__ == "__main__":
    tester = ActiveChartsTest()
    tester.run_comprehensive_test()