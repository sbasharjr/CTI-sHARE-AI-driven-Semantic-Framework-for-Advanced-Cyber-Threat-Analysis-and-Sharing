#!/usr/bin/env python3
"""
Comprehensive Test for Enhanced CTI-sHARE Dashboard with Threat Intelligence Sharing
"""

import sys
import os
import time
import requests
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_threat_intelligence_features():
    """Test all threat intelligence sharing features"""
    base_url = "http://localhost:5001"
    
    print("🧪 Testing Enhanced Threat Intelligence Features...")
    print("=" * 60)
    
    # Test text analysis with entity extraction
    try:
        test_text = "Malicious IP 192.168.1.100 contacted C&C server malware.example.com via HTTP. CVE-2023-1234 exploit detected in file C:\\temp\\malware.exe with hash d41d8cd98f00b204e9800998ecf8427e"
        
        print("🔍 Testing semantic entity extraction...")
        response = requests.post(f"{base_url}/api/dashboard/semantic/entities", 
                               json={"text": test_text}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                entities = data.get('entities', {})
                stats = data.get('statistics', {})
                print("✅ Entity extraction: PASS")
                print(f"   IPs found: {stats.get('ips_found', 0)}")
                print(f"   Domains found: {stats.get('domains_found', 0)}")
                print(f"   Hashes found: {stats.get('hashes_found', 0)}")
                print(f"   CVE IDs found: {stats.get('cve_ids_found', 0)}")
            else:
                print(f"❌ Entity extraction: FAIL - {data.get('message', 'Unknown error')}")
        else:
            print(f"❌ Entity extraction: FAIL ({response.status_code})")
    except Exception as e:
        print(f"❌ Entity extraction: FAIL - {str(e)}")
    
    # Test bulk semantic analysis
    try:
        print("\n📊 Testing bulk semantic analysis...")
        bulk_texts = [
            "Ransomware detected in network traffic",
            "Phishing email with credential harvesting",
            "DDoS attack from botnet detected",
            "Suspicious file with malware signature"
        ]
        
        response = requests.post(f"{base_url}/api/dashboard/semantic/analyze-bulk", 
                               json={"texts": bulk_texts}, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                print("✅ Bulk semantic analysis: PASS")
                print(f"   Total analyzed: {data.get('total_analyzed', 0)}")
                print(f"   Threats detected: {data.get('threats_detected', 0)}")
                print(f"   Categories found: {len(data.get('categories_distribution', {}))}")
            else:
                print(f"❌ Bulk semantic analysis: FAIL - {data.get('message', 'Unknown error')}")
        else:
            print(f"❌ Bulk semantic analysis: FAIL ({response.status_code})")
    except Exception as e:
        print(f"❌ Bulk semantic analysis: FAIL - {str(e)}")
    
    # Test threat intelligence import
    try:
        print("\n📥 Testing threat intelligence import...")
        import_data = {
            "source": "manual",
            "threats": [
                {
                    "description": "Test threat from API",
                    "category": "test",
                    "severity": "HIGH"
                },
                {
                    "description": "Another test threat",
                    "category": "malware", 
                    "severity": "MEDIUM"
                }
            ]
        }
        
        response = requests.post(f"{base_url}/api/dashboard/intelligence/import", 
                               json=import_data, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                print("✅ Threat intelligence import: PASS")
                print(f"   Threats imported: {data.get('threats_imported', 0)}")
            else:
                print(f"❌ Threat intelligence import: FAIL - {data.get('message', 'Unknown error')}")
        else:
            print(f"❌ Threat intelligence import: FAIL ({response.status_code})")
    except Exception as e:
        print(f"❌ Threat intelligence import: FAIL - {str(e)}")
    
    # Test threat intelligence sharing
    try:
        print("\n🌐 Testing threat intelligence sharing...")
        share_data = {
            "threats": [
                {
                    "description": "Shared test threat",
                    "category": "test",
                    "severity": "LOW"
                }
            ],
            "destinations": ["internal", "misp"],
            "format": "json"
        }
        
        response = requests.post(f"{base_url}/api/dashboard/intelligence/share", 
                               json=share_data, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                print("✅ Threat intelligence sharing: PASS")
                print(f"   Total shared: {data.get('total_shared', 0)}")
                print(f"   Destinations: {len(data.get('sharing_results', []))}")
            else:
                print(f"❌ Threat intelligence sharing: FAIL - {data.get('message', 'Unknown error')}")
        else:
            print(f"❌ Threat intelligence sharing: FAIL ({response.status_code})")
    except Exception as e:
        print(f"❌ Threat intelligence sharing: FAIL - {str(e)}")
    
    # Test export functionality
    try:
        print("\n📤 Testing threat intelligence export...")
        formats = ['json', 'csv', 'stix']
        
        for fmt in formats:
            response = requests.get(f"{base_url}/api/dashboard/intelligence/export?format={fmt}", timeout=10)
            if response.status_code == 200:
                print(f"✅ Export ({fmt.upper()}): PASS - {len(response.content)} bytes")
            else:
                print(f"❌ Export ({fmt.upper()}): FAIL ({response.status_code})")
                
    except Exception as e:
        print(f"❌ Export testing: FAIL - {str(e)}")
    
    print("\n" + "=" * 60)
    print("🎯 All threat intelligence features tested!")
    return True

def run_enhanced_dashboard_with_intelligence():
    """Run the enhanced dashboard with full threat intelligence capabilities"""
    try:
        from src.dashboard.dashboard import ThreatDashboard
        from src.semantic_analysis.semantic_analyzer import ThreatSemanticAnalyzer
        from src.realtime.detector import RealTimeThreatDetector
        
        print("🛡️ Starting CTI-sHARE Enhanced Dashboard with Threat Intelligence Sharing...")
        print("=" * 80)
        
        # Initialize components
        analyzer = ThreatSemanticAnalyzer()
        detector = RealTimeThreatDetector(semantic_analyzer=analyzer)
        
        # Create enhanced dashboard
        dashboard = ThreatDashboard(threat_detector=detector, semantic_analyzer=analyzer)
        
        # Add comprehensive sample data
        sample_threats = [
            {
                'description': 'Advanced persistent threat with C&C communication to malware.example.com',
                'category': 'apt',
                'severity': 'CRITICAL',
                'timestamp': '2025-10-22T10:00:00',
                'is_threat': True,
                'entities': {
                    'ips': ['192.168.1.100', '203.0.113.1'],
                    'domains': ['malware.example.com'],
                    'hashes': ['d41d8cd98f00b204e9800998ecf8427e']
                }
            },
            {
                'description': 'Phishing campaign targeting credentials via fake login page',
                'category': 'phishing',
                'severity': 'HIGH',
                'timestamp': '2025-10-22T11:30:00',
                'is_threat': True,
                'entities': {
                    'urls': ['http://fake-bank.evil.com/login'],
                    'emails': ['admin@evil.com']
                }
            },
            {
                'description': 'Ransomware deployment detected with file encryption behavior',
                'category': 'malware',
                'severity': 'CRITICAL',
                'timestamp': '2025-10-22T12:15:00',
                'is_threat': True,
                'entities': {
                    'file_paths': ['C:\\temp\\malware.exe', '/tmp/ransomware'],
                    'hashes': ['5d41402abc4b2a76b9719d911017c592']
                }
            },
            {
                'description': 'DDoS attack from distributed botnet infrastructure',
                'category': 'network_attack',
                'severity': 'HIGH',
                'timestamp': '2025-10-22T13:45:00',
                'is_threat': True,
                'entities': {
                    'ips': ['10.0.0.50', '172.16.0.25', '198.51.100.1'],
                    'domains': ['botnet.command.net']
                }
            },
            {
                'description': 'CVE-2023-1234 exploitation attempt detected in web traffic',
                'category': 'vulnerability_exploit',
                'severity': 'HIGH',
                'timestamp': '2025-10-22T14:20:00',
                'is_threat': True,
                'entities': {
                    'cve_ids': ['CVE-2023-1234'],
                    'ips': ['172.17.0.1']
                }
            }
        ]
        
        for threat in sample_threats:
            dashboard.add_threat(threat)
        
        print(f"🌐 Enhanced Dashboard: http://localhost:5001")
        print(f"📊 New Threat Intelligence Features:")
        print(f"   • File Upload: Upload JSON/CSV/TXT threat files")
        print(f"   • Import/Export: MISP, STIX, TAXII format support")
        print(f"   • Entity Extraction: IPs, domains, hashes, CVEs, etc.")
        print(f"   • Bulk Analysis: Analyze multiple threats at once")
        print(f"   • Intelligence Sharing: Share with external platforms")
        print("=" * 80)
        print(f"🎯 New API Endpoints:")
        print(f"   • POST /api/dashboard/intelligence/upload")
        print(f"   • POST /api/dashboard/intelligence/import")
        print(f"   • GET  /api/dashboard/intelligence/export")
        print(f"   • POST /api/dashboard/intelligence/share") 
        print(f"   • POST /api/dashboard/semantic/analyze-bulk")
        print(f"   • POST /api/dashboard/semantic/entities")
        print("=" * 80)
        print(f"Sample data loaded: {len(sample_threats)} comprehensive threats")
        stats = dashboard._get_dashboard_stats()
        print(f"Total threats: {stats['total_threats']}")
        print(f"Critical threats: {stats['critical_threats']}")
        print(f"Detection rate: {stats['detection_rate']}%")
        print("=" * 80)
        print("🚀 Enhanced Features Available:")
        print("   📤 Upload threat intelligence files (JSON/CSV/TXT)")
        print("   📥 Import from MISP, STIX, TAXII formats")
        print("   📊 Bulk semantic analysis of multiple texts")
        print("   🔍 Advanced entity extraction (IPs, domains, hashes, CVEs)")
        print("   🌐 Share intelligence with external platforms")
        print("   📄 Export in multiple formats (JSON, CSV, STIX)")
        print("=" * 80)
        print("Press Ctrl+C to stop the server")
        print("=" * 80)
        
        # Start the enhanced dashboard
        dashboard.run(host='127.0.0.1', port=5001, debug=True)
        
    except Exception as e:
        print(f"❌ Error starting enhanced dashboard: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Test mode - assumes server is already running
        test_threat_intelligence_features()
    else:
        # Run the enhanced dashboard server with full threat intelligence
        run_enhanced_dashboard_with_intelligence()