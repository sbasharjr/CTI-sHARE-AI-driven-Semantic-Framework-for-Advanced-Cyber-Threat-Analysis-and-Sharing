#!/usr/bin/env python3
"""
Simple Dashboard Runner with Live Threat Data
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir / 'src'))

def create_sample_live_threats():
    """Create sample live threat data"""
    current_time = datetime.now()
    
    threats = [
        {
            'id': 'LIVE-001',
            'title': 'Emotet Botnet Resurgence October 2025',
            'description': 'Major Emotet botnet campaign targeting global financial institutions with advanced evasion techniques.',
            'category': 'botnet',
            'severity': 'CRITICAL',
            'confidence': 95,
            'is_threat': True,
            'tlp': 'RED',
            'source': 'Botnet Tracker',
            'timestamp': (current_time - timedelta(hours=2)).isoformat(),
            'tags': ['emotet', 'botnet', 'financial-targeting', 'active-campaign'],
            'entities': {
                'ips': ['185.220.100.241', '185.220.101.182', '91.219.236.166'],
                'domains': ['emotet-c2-2025.tk', 'banking-update-portal.ml'],
                'file_hashes': ['a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456']
            },
            'geographic_origin': 'Eastern Europe',
            'attack_vector': 'Email phishing'
        },
        {
            'id': 'LIVE-002', 
            'title': 'RedLine Stealer Campaign via Malvertising',
            'description': 'Active RedLine Stealer distribution through malicious advertisements targeting cryptocurrency users.',
            'category': 'malware',
            'severity': 'HIGH',
            'confidence': 88,
            'is_threat': True,
            'tlp': 'AMBER',
            'source': 'VirusTotal',
            'timestamp': (current_time - timedelta(hours=1)).isoformat(),
            'tags': ['redline-stealer', 'malvertising', 'cryptocurrency', 'credential-theft'],
            'entities': {
                'ips': ['203.0.113.150', '198.51.100.200'],
                'domains': ['crypto-exchange-update.ga', 'secure-wallet-app.tk'],
                'urls': ['https://fake-binance-update.com/wallet.exe']
            },
            'geographic_origin': 'Russia',
            'attack_vector': 'Malvertising'
        },
        {
            'id': 'LIVE-003',
            'title': 'APT29 Infrastructure Updates Detected',
            'description': 'APT29 (Cozy Bear) showing new command and control infrastructure with advanced persistence techniques.',
            'category': 'apt',
            'severity': 'CRITICAL', 
            'confidence': 97,
            'is_threat': True,
            'tlp': 'RED',
            'source': 'National Intelligence',
            'timestamp': (current_time - timedelta(minutes=30)).isoformat(),
            'tags': ['apt29', 'cozy-bear', 'nation-state', 'government-targeting'],
            'entities': {
                'ips': ['185.220.102.50', '91.219.237.100'],
                'domains': ['microsoft-update-service.org', 'windows-security-center.net'],
                'certificates': ['CN=*.microsoft-update-portal.com']
            },
            'geographic_origin': 'Russia',
            'attribution': 'APT29 / SVR',
            'attack_vector': 'Spear phishing'
        },
        {
            'id': 'LIVE-004',
            'title': 'LockBit Ransomware Double Extortion Campaign',
            'description': 'LockBit ransomware group targeting healthcare organizations with double extortion tactics.',
            'category': 'ransomware',
            'severity': 'CRITICAL',
            'confidence': 92,
            'is_threat': True,
            'tlp': 'RED',
            'source': 'Ransomware Tracker',
            'timestamp': (current_time - timedelta(hours=4)).isoformat(),
            'tags': ['lockbit', 'ransomware', 'healthcare', 'double-extortion'],
            'entities': {
                'ips': ['37.221.113.150', '5.189.140.200'],
                'domains': ['lockbit-payment-portal.onion'],
                'bitcoin_addresses': ['bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh']
            },
            'geographic_origin': 'Unknown',
            'ransom_amount': '$2,500,000',
            'attack_vector': 'RDP brute force'
        },
        {
            'id': 'LIVE-005',
            'title': 'Widespread Exploitation of CVE-2025-4966',
            'description': 'Active exploitation of critical vulnerability CVE-2025-4966 in Citrix NetScaler appliances.',
            'category': 'exploit',
            'severity': 'CRITICAL',
            'confidence': 94,
            'is_threat': True,
            'tlp': 'WHITE',
            'source': 'CISA KEV',
            'timestamp': (current_time - timedelta(minutes=45)).isoformat(),
            'tags': ['cve-2025-4966', 'citrix', 'exploit', 'active-exploitation'],
            'entities': {
                'ips': ['203.0.113.75', '198.51.100.125', '192.0.2.200'],
                'urls': ['http://exploit-toolkit.tk/citrix-poc.php'],
                'cves': ['CVE-2025-4966']
            },
            'cvss_score': 9.8,
            'affected_products': 'Citrix NetScaler ADC and Gateway'
        }
    ]
    
    return threats

def run_dashboard_simple():
    """Run dashboard with simple live threat data"""
    print("=" * 80)
    print("🛡️  CTI-sHARE Dashboard - Live Threat Intelligence")
    print("=" * 80)
    
    try:
        # Generate live threat data
        threats = create_sample_live_threats()
        print(f"📊 Generated {len(threats)} live threat indicators")
        
        # Save to file
        with open('live_threat_data.json', 'w') as f:
            json.dump(threats, f, indent=2)
        print("💾 Live threat data saved to: live_threat_data.json")
        
        # Display threat summary
        print("\n🚨 Current Live Threats:")
        for threat in threats:
            hours_ago = int((datetime.now() - datetime.fromisoformat(threat['timestamp'])).total_seconds() / 3600)
            severity_emoji = "🔴" if threat['severity'] == 'CRITICAL' else "🟡" if threat['severity'] == 'HIGH' else "🟢"
            print(f"   {severity_emoji} {threat['title']} ({hours_ago}h ago)")
        
        print("\n📈 Threat Statistics:")
        categories = {}
        severities = {}
        for threat in threats:
            categories[threat['category']] = categories.get(threat['category'], 0) + 1
            severities[threat['severity']] = severities.get(threat['severity'], 0) + 1
        
        print(f"   Categories: {categories}")
        print(f"   Severities: {severities}")
        
        print("\n" + "=" * 80)
        print("✅ Live Threat Data Initialization Complete!")
        print("🚀 Ready to start CTI-sHARE Dashboard")
        print("=" * 80)
        
        return threats
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == '__main__':
    run_dashboard_simple()