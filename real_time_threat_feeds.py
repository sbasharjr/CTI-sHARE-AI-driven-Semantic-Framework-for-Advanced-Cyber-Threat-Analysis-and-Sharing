#!/usr/bin/env python3
"""
Real-time Threat Feed Simulator for CTI-sHARE Dashboard
Simulates live threat intelligence feeds with realistic data patterns
"""

import json
import time
import random
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any

class RealTimeThreatFeedSimulator:
    """Simulate real-time threat intelligence feeds"""
    
    def __init__(self):
        self.is_running = False
        self.feed_thread = None
        self.threat_queue = []
        
        # Real threat feed sources
        self.threat_feeds = {
            "MISP Community": {
                "status": "active",
                "reliability": 0.95,
                "update_frequency": 300,  # seconds
                "threat_types": ["malware", "phishing", "apt"]
            },
            "AlienVault OTX": {
                "status": "active", 
                "reliability": 0.92,
                "update_frequency": 180,
                "threat_types": ["exploits", "malware", "botnet"]
            },
            "VirusTotal": {
                "status": "active",
                "reliability": 0.98,
                "update_frequency": 120,
                "threat_types": ["malware", "suspicious_files"]
            },
            "Emerging Threats": {
                "status": "active",
                "reliability": 0.90,
                "update_frequency": 240,
                "threat_types": ["network_intrusion", "malware", "botnet"]
            },
            "ThreatFox": {
                "status": "active",
                "reliability": 0.88,
                "update_frequency": 360,
                "threat_types": ["iocs", "malware", "c2_infrastructure"]
            },
            "Abuse.ch": {
                "status": "active",
                "reliability": 0.93,
                "update_frequency": 200,
                "threat_types": ["malware", "botnet", "urls"]
            }
        }
        
        # Current global threat landscape
        self.active_campaigns = [
            {
                "name": "RedLine Stealer Campaign Q4 2025",
                "active_since": datetime.now() - timedelta(days=15),
                "intensity": "high",
                "geographic_focus": ["North America", "Europe"],
                "target_sectors": ["Financial Services", "Healthcare", "Retail"]
            },
            {
                "name": "Emotet Resurgence October 2025", 
                "active_since": datetime.now() - timedelta(days=8),
                "intensity": "critical",
                "geographic_focus": ["Global"],
                "target_sectors": ["Government", "Critical Infrastructure"]
            },
            {
                "name": "APT29 Infrastructure Updates",
                "active_since": datetime.now() - timedelta(days=3),
                "intensity": "medium",
                "geographic_focus": ["Europe", "Asia"],
                "target_sectors": ["Government", "Defense", "Technology"]
            }
        ]

    def generate_real_time_threat(self) -> Dict[str, Any]:
        """Generate a realistic real-time threat based on current landscape"""
        
        # Select random active campaign
        campaign = random.choice(self.active_campaigns)
        
        # Generate threat based on campaign characteristics
        if "RedLine" in campaign["name"]:
            return self._generate_redline_threat(campaign)
        elif "Emotet" in campaign["name"]:
            return self._generate_emotet_threat(campaign)
        elif "APT29" in campaign["name"]:
            return self._generate_apt29_threat(campaign)
        else:
            return self._generate_generic_threat(campaign)

    def _generate_redline_threat(self, campaign: Dict) -> Dict[str, Any]:
        """Generate RedLine Stealer campaign threat"""
        return {
            'id': f"RT-REDLINE-{random.randint(100000, 999999)}",
            'title': "RedLine Stealer Distribution via Malicious Advertisements",
            'description': f"Active RedLine Stealer campaign targeting {random.choice(campaign['target_sectors'])} "
                         f"via malvertising. Campaign observed in {random.choice(campaign['geographic_focus'])}.",
            'category': 'malware',
            'severity': 'HIGH',
            'confidence': random.randint(88, 95),
            'is_threat': True,
            'tlp': 'AMBER',
            'source': random.choice(list(self.threat_feeds.keys())),
            'timestamp': datetime.now().isoformat(),
            'tags': ['redline-stealer', 'malvertising', 'credential-theft', 'active-campaign'],
            'campaign_id': 'redline-q4-2025',
            'entities': {
                'ips': [self._generate_malicious_ip() for _ in range(random.randint(2, 4))],
                'domains': [f'ad-network-{random.randint(1000, 9999)}.{random.choice(["tk", "ml", "ga"])}'],
                'file_hashes': [self._generate_hash() for _ in range(random.randint(1, 2))],
                'urls': [f'https://fake-download-{random.randint(100, 999)}.com/update.exe']
            },
            'geographic_origin': random.choice(['Russia', 'Eastern Europe']),
            'real_time_indicators': {
                'active_c2_servers': random.randint(15, 45),
                'infected_hosts_24h': random.randint(500, 2000),
                'credential_harvests_24h': random.randint(100, 800)
            }
        }

    def _generate_emotet_threat(self, campaign: Dict) -> Dict[str, Any]:
        """Generate Emotet resurgence threat"""
        return {
            'id': f"RT-EMOTET-{random.randint(100000, 999999)}",
            'title': "Emotet Botnet Infrastructure Expansion",
            'description': f"Emotet botnet showing significant infrastructure expansion. "
                         f"New command and control servers detected with advanced evasion techniques.",
            'category': 'botnet',
            'severity': 'CRITICAL',
            'confidence': random.randint(92, 98),
            'is_threat': True,
            'tlp': 'RED',
            'source': random.choice(['Botnet Tracker', 'Abuse.ch', 'Shadowserver']),
            'timestamp': datetime.now().isoformat(),
            'tags': ['emotet', 'botnet', 'c2-expansion', 'critical-infrastructure'],
            'campaign_id': 'emotet-resurgence-2025',
            'entities': {
                'ips': [self._generate_malicious_ip() for _ in range(random.randint(5, 10))],
                'domains': [f'emotet-c2-{random.randint(100, 999)}.{random.choice(["com", "net", "org"])}'],
                'file_hashes': [self._generate_hash() for _ in range(random.randint(3, 6))]
            },
            'geographic_origin': random.choice(['Global Infrastructure']),
            'real_time_indicators': {
                'active_c2_servers': random.randint(50, 150),
                'bot_population': random.randint(50000, 200000),
                'spam_volume_24h': random.randint(1000000, 5000000)
            }
        }

    def _generate_apt29_threat(self, campaign: Dict) -> Dict[str, Any]:
        """Generate APT29 infrastructure threat"""
        return {
            'id': f"RT-APT29-{random.randint(100000, 999999)}",
            'title': "APT29 Cozy Bear Infrastructure Updates Detected",
            'description': f"APT29 (Cozy Bear) infrastructure updates observed. "
                         f"New command and control domains registered with advanced evasion techniques.",
            'category': 'apt',
            'severity': 'CRITICAL',
            'confidence': random.randint(95, 99),
            'is_threat': True,
            'tlp': 'RED',
            'source': 'National Threat Intelligence',
            'timestamp': datetime.now().isoformat(),
            'tags': ['apt29', 'cozy-bear', 'nation-state', 'advanced-persistent-threat'],
            'campaign_id': 'apt29-infrastructure-2025',
            'entities': {
                'ips': [self._generate_malicious_ip() for _ in range(random.randint(3, 7))],
                'domains': [f'legitimate-sounding-{random.randint(1, 99)}.{random.choice(["org", "net", "info"])}'],
                'certificates': [f'CN=*.microsoft-update-{random.randint(1, 20)}.com']
            },
            'geographic_origin': 'Russia',
            'attribution': 'APT29 / Cozy Bear / SVR',
            'real_time_indicators': {
                'new_infrastructure_24h': random.randint(5, 15),
                'target_reconnaissance': random.randint(20, 100),
                'phishing_attempts_24h': random.randint(50, 200)
            }
        }

    def _generate_generic_threat(self, campaign: Dict) -> Dict[str, Any]:
        """Generate generic real-time threat"""
        threat_types = ['phishing', 'malware', 'exploit', 'ransomware']
        threat_type = random.choice(threat_types)
        
        return {
            'id': f"RT-GENERIC-{random.randint(100000, 999999)}",
            'title': f"Real-time {threat_type.title()} Activity Detected",
            'description': f"Active {threat_type} campaign detected via real-time threat feeds. "
                         f"Campaign shows characteristics consistent with {campaign['name']}.",
            'category': threat_type,
            'severity': random.choice(['HIGH', 'MEDIUM', 'CRITICAL']),
            'confidence': random.randint(80, 92),
            'is_threat': True,
            'tlp': random.choice(['WHITE', 'GREEN', 'AMBER']),
            'source': random.choice(list(self.threat_feeds.keys())),
            'timestamp': datetime.now().isoformat(),
            'tags': [threat_type, 'real-time-detection', 'active-campaign'],
            'entities': {
                'ips': [self._generate_malicious_ip() for _ in range(random.randint(1, 4))],
                'domains': [f'{threat_type}-{random.randint(100, 999)}.{random.choice(["tk", "ml", "ga"])}']
            },
            'geographic_origin': random.choice(campaign['geographic_focus']),
            'real_time_indicators': {
                'detection_sources': random.randint(3, 8),
                'activity_volume_1h': random.randint(10, 100)
            }
        }

    def _generate_malicious_ip(self) -> str:
        """Generate realistic malicious IP addresses"""
        # Known malicious IP ranges and suspicious regions
        suspicious_ranges = [
            (185, random.randint(1, 255), random.randint(1, 255)),  # RIPE region
            (91, random.randint(1, 255), random.randint(1, 255)),   # Eastern Europe
            (5, random.randint(1, 255), random.randint(1, 255)),    # RIPE region
            (37, random.randint(1, 255), random.randint(1, 255)),   # Africa/Middle East
        ]
        range_prefix = random.choice(suspicious_ranges)
        return f"{range_prefix[0]}.{range_prefix[1]}.{range_prefix[2]}.{random.randint(1, 254)}"

    def _generate_hash(self) -> str:
        """Generate realistic file hash"""
        return ''.join(random.choices('0123456789abcdef', k=64))  # SHA256

    def start_real_time_feed(self, callback_func=None):
        """Start real-time threat feed simulation"""
        if self.is_running:
            print("⚠️ Real-time feed already running")
            return
        
        self.is_running = True
        
        def feed_worker():
            print("🔄 Real-time threat feed started")
            
            while self.is_running:
                try:
                    # Generate new threat every 30-180 seconds
                    wait_time = random.randint(30, 180)
                    time.sleep(wait_time)
                    
                    if not self.is_running:
                        break
                    
                    # Generate new threat
                    threat = self.generate_real_time_threat()
                    self.threat_queue.append(threat)
                    
                    # Limit queue size
                    if len(self.threat_queue) > 100:
                        self.threat_queue.pop(0)
                    
                    print(f"🚨 New real-time threat: {threat['title']}")
                    
                    # Call callback if provided
                    if callback_func:
                        callback_func(threat)
                        
                except Exception as e:
                    print(f"❌ Error in real-time feed: {e}")
                    time.sleep(60)  # Wait before retrying
        
        self.feed_thread = threading.Thread(target=feed_worker, daemon=True)
        self.feed_thread.start()

    def stop_real_time_feed(self):
        """Stop real-time threat feed simulation"""
        if self.is_running:
            self.is_running = False
            print("🛑 Real-time threat feed stopped")

    def get_feed_status(self) -> Dict[str, Any]:
        """Get current status of all threat feeds"""
        feed_status = {}
        
        for feed_name, config in self.threat_feeds.items():
            # Simulate occasional feed issues
            if random.random() < 0.05:  # 5% chance of temporary issue
                status = "warning"
                health = random.randint(70, 85)
            else:
                status = "active"
                health = random.randint(85, 100)
            
            feed_status[feed_name] = {
                "status": status,
                "health": health,
                "last_update": (datetime.now() - timedelta(minutes=random.randint(1, 30))).isoformat(),
                "reliability": config["reliability"],
                "threats_24h": random.randint(50, 500),
                "iocs_24h": random.randint(100, 2000)
            }
        
        return feed_status

    def get_recent_threats(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent threats from the queue"""
        return sorted(self.threat_queue, key=lambda x: x['timestamp'], reverse=True)[:limit]

def create_dashboard_with_live_feeds():
    """Create and run dashboard with live threat feeds"""
    print("=" * 80)
    print("🌐 CTI-sHARE Dashboard with Real-Time Threat Feeds")
    print("=" * 80)
    
    # Initialize threat feed simulator
    feed_simulator = RealTimeThreatFeedSimulator()
    
    # Start real-time feed
    def threat_handler(threat):
        print(f"📥 Processing real-time threat: {threat['category'].upper()} - {threat['severity']}")
    
    feed_simulator.start_real_time_feed(callback_func=threat_handler)
    
    print("✅ Real-time threat feeds active")
    print("🔄 Dashboard will receive live threat intelligence")
    print("📡 Feed sources: MISP, OTX, VirusTotal, Emerging Threats, ThreatFox, Abuse.ch")
    
    return feed_simulator

if __name__ == '__main__':
    simulator = create_dashboard_with_live_feeds()
    
    try:
        # Keep running and display feed status
        while True:
            time.sleep(30)
            status = simulator.get_feed_status()
            active_feeds = sum(1 for feed in status.values() if feed['status'] == 'active')
            total_threats = sum(feed['threats_24h'] for feed in status.values())
            
            print(f"📊 Feed Status: {active_feeds}/{len(status)} active | Total threats (24h): {total_threats:,}")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping real-time feeds...")
        simulator.stop_real_time_feed()
        print("✅ Shutdown complete")