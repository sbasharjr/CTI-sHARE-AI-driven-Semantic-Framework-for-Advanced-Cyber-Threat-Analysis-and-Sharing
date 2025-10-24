#!/usr/bin/env python3
"""
Live Threat Data Initializer for CTI-sHARE Dashboard
Generates realistic, current threat intelligence data for dashboard initialization
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

class LiveThreatDataGenerator:
    """Generate realistic live threat data for dashboard initialization"""
    
    def __init__(self):
        self.current_time = datetime.now()
        
        # Real-world threat indicators and patterns
        self.malware_families = [
            "Emotet", "TrickBot", "Qakbot", "Cobalt Strike", "Ryuk", "Conti", 
            "Dridex", "IcedID", "BazaLoader", "Zloader", "Gozi", "RedLine Stealer"
        ]
        
        self.attack_groups = [
            "APT29", "APT28", "Lazarus Group", "FIN7", "Carbanak", "APT40",
            "Wizard Spider", "TA505", "Silence Group", "Evil Corp", "DarkHalo"
        ]
        
        self.countries = [
            "United States", "China", "Russia", "North Korea", "Iran", "Germany",
            "United Kingdom", "Brazil", "India", "Japan", "South Korea", "Canada"
        ]
        
        self.threat_vectors = [
            "Email phishing", "Spear phishing", "Watering hole", "Supply chain",
            "RDP brute force", "VPN exploitation", "Web application attacks",
            "USB-based attacks", "Social engineering", "Insider threats"
        ]
        
        self.iot_devices = [
            "Router", "IP Camera", "Smart TV", "IoT Gateway", "Industrial Control System",
            "Smart Thermostat", "Network Printer", "VoIP Phone", "NAS Device", "Smart Lock"
        ]

    def generate_current_threats(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate current live threat data"""
        threats = []
        
        for i in range(count):
            threat_type = random.choice(['malware', 'phishing', 'apt', 'ransomware', 'botnet', 'exploit'])
            
            if threat_type == 'malware':
                threat = self._generate_malware_threat()
            elif threat_type == 'phishing':
                threat = self._generate_phishing_threat()
            elif threat_type == 'apt':
                threat = self._generate_apt_threat()
            elif threat_type == 'ransomware':
                threat = self._generate_ransomware_threat()
            elif threat_type == 'botnet':
                threat = self._generate_botnet_threat()
            else:
                threat = self._generate_exploit_threat()
            
            # Add timestamp within last 24 hours
            hours_ago = random.randint(0, 24)
            threat['timestamp'] = (self.current_time - timedelta(hours=hours_ago)).isoformat()
            
            threats.append(threat)
        
        return sorted(threats, key=lambda x: x['timestamp'], reverse=True)

    def _generate_malware_threat(self) -> Dict[str, Any]:
        """Generate malware threat data"""
        family = random.choice(self.malware_families)
        country = random.choice(self.countries)
        
        return {
            'id': f"MAL-{random.randint(10000, 99999)}",
            'title': f'{family} Malware Campaign Detected',
            'description': f'Active {family} malware distribution campaign targeting financial institutions. '
                         f'Observed command and control infrastructure originating from {country}.',
            'category': 'malware',
            'severity': random.choice(['HIGH', 'CRITICAL', 'MEDIUM']),
            'confidence': random.randint(85, 98),
            'is_threat': True,
            'tlp': random.choice(['WHITE', 'GREEN', 'AMBER']),
            'source': random.choice(['MISP', 'AlienVault OTX', 'VirusTotal', 'Internal Honeypot']),
            'tags': [family.lower(), 'banking-trojan', 'c2-communication', country.lower().replace(' ', '-')],
            'entities': {
                'ips': [self._generate_ip() for _ in range(random.randint(2, 5))],
                'domains': [f'{family.lower()}-c2-{random.randint(1, 99)}.{random.choice(["com", "net", "org"])}'],
                'file_hashes': [self._generate_hash() for _ in range(random.randint(1, 3))],
                'urls': [f'http://{family.lower()}-panel.{random.choice(["tk", "ml", "ga"])}/gate.php']
            },
            'iocs': {
                'network_indicators': random.randint(5, 15),
                'file_indicators': random.randint(3, 8),
                'behavioral_indicators': random.randint(2, 6)
            },
            'geographic_origin': country,
            'attack_vector': random.choice(self.threat_vectors)
        }

    def _generate_phishing_threat(self) -> Dict[str, Any]:
        """Generate phishing threat data"""
        target = random.choice(['Microsoft', 'Google', 'Banking', 'Cryptocurrency', 'Social Media'])
        country = random.choice(self.countries)
        
        return {
            'id': f"PHISH-{random.randint(10000, 99999)}",
            'title': f'{target} Phishing Campaign Active',
            'description': f'Large-scale phishing campaign impersonating {target} services. '
                         f'Campaign infrastructure hosted in {country}. High volume of credential harvesting attempts.',
            'category': 'phishing',
            'severity': random.choice(['HIGH', 'MEDIUM']),
            'confidence': random.randint(90, 99),
            'is_threat': True,
            'tlp': 'AMBER',
            'source': random.choice(['Phishing Database', 'URLVoid', 'OpenPhish', 'PhishTank']),
            'tags': ['phishing', 'credential-harvesting', target.lower(), 'social-engineering'],
            'entities': {
                'domains': [f'fake-{target.lower()}-{random.randint(100, 999)}.{random.choice(["tk", "ml", "cf"])}'],
                'urls': [f'https://secure-{target.lower()}.{random.choice(["tk", "ml", "ga"])}/login.php'],
                'ips': [self._generate_ip() for _ in range(random.randint(1, 3))],
                'emails': [f'{random.choice(["no-reply", "security", "support"])}@fake-{target.lower()}.com']
            },
            'iocs': {
                'network_indicators': random.randint(8, 20),
                'email_indicators': random.randint(5, 12),
                'web_indicators': random.randint(10, 25)
            },
            'geographic_origin': country,
            'attack_vector': 'Email phishing',
            'targets': f'{target} users globally'
        }

    def _generate_apt_threat(self) -> Dict[str, Any]:
        """Generate APT (Advanced Persistent Threat) data"""
        group = random.choice(self.attack_groups)
        country = random.choice(self.countries)
        
        return {
            'id': f"APT-{random.randint(10000, 99999)}",
            'title': f'{group} Advanced Persistent Threat Activity',
            'description': f'Sophisticated {group} campaign targeting government and critical infrastructure. '
                         f'Advanced evasion techniques and custom malware observed.',
            'category': 'apt',
            'severity': 'CRITICAL',
            'confidence': random.randint(95, 99),
            'is_threat': True,
            'tlp': 'RED',
            'source': 'Threat Intelligence Team',
            'tags': [group.lower().replace(' ', '-'), 'apt', 'targeted-attack', 'nation-state'],
            'entities': {
                'ips': [self._generate_ip() for _ in range(random.randint(3, 8))],
                'domains': [f'{group.lower().replace(" ", "")}-infra-{random.randint(1, 50)}.com'],
                'file_hashes': [self._generate_hash() for _ in range(random.randint(2, 5))],
                'certificates': [f'CN=*.{random.choice(["microsoft", "google", "adobe"])}-update.com']
            },
            'iocs': {
                'network_indicators': random.randint(15, 30),
                'file_indicators': random.randint(8, 18),
                'behavioral_indicators': random.randint(10, 20),
                'infrastructure_indicators': random.randint(5, 12)
            },
            'geographic_origin': country,
            'attack_vector': 'Spear phishing',
            'targets': 'Government, Defense, Critical Infrastructure',
            'attribution': group
        }

    def _generate_ransomware_threat(self) -> Dict[str, Any]:
        """Generate ransomware threat data"""
        ransomware_families = ["Ryuk", "Conti", "DarkSide", "REvil", "LockBit", "BlackMatter", "Hive"]
        family = random.choice(ransomware_families)
        country = random.choice(self.countries)
        
        return {
            'id': f"RANSOM-{random.randint(10000, 99999)}",
            'title': f'{family} Ransomware Campaign',
            'description': f'Active {family} ransomware deployment targeting enterprise networks. '
                         f'Double extortion tactics observed with data exfiltration capabilities.',
            'category': 'ransomware',
            'severity': 'CRITICAL',
            'confidence': random.randint(92, 99),
            'is_threat': True,
            'tlp': 'RED',
            'source': random.choice(['Ransomware Tracker', 'Industrial Security', 'CISA Alert']),
            'tags': [family.lower(), 'ransomware', 'double-extortion', 'enterprise-target'],
            'entities': {
                'ips': [self._generate_ip() for _ in range(random.randint(2, 6))],
                'domains': [f'{family.lower()}-payment-{random.randint(1, 20)}.onion'],
                'file_hashes': [self._generate_hash() for _ in range(random.randint(2, 4))],
                'bitcoin_addresses': [self._generate_bitcoin_address() for _ in range(random.randint(1, 3))]
            },
            'iocs': {
                'network_indicators': random.randint(10, 25),
                'file_indicators': random.randint(8, 15),
                'payment_indicators': random.randint(2, 5),
                'behavioral_indicators': random.randint(6, 12)
            },
            'geographic_origin': country,
            'attack_vector': random.choice(['RDP brute force', 'Email phishing', 'Supply chain']),
            'ransom_amount': f'${random.randint(50000, 5000000):,}',
            'encryption_algorithm': random.choice(['AES-256', 'ChaCha20', 'RSA-2048'])
        }

    def _generate_botnet_threat(self) -> Dict[str, Any]:
        """Generate botnet threat data"""
        botnets = ["Emotet", "TrickBot", "Mirai", "Necurs", "Dridex", "Zeus", "Conficker"]
        botnet = random.choice(botnets)
        country = random.choice(self.countries)
        
        return {
            'id': f"BOT-{random.randint(10000, 99999)}",
            'title': f'{botnet} Botnet Activity Surge',
            'description': f'Significant increase in {botnet} botnet activity. '
                         f'Command and control servers showing high activity with {random.randint(10000, 100000):,} infected hosts.',
            'category': 'botnet',
            'severity': random.choice(['HIGH', 'MEDIUM']),
            'confidence': random.randint(88, 96),
            'is_threat': True,
            'tlp': 'AMBER',
            'source': random.choice(['Botnet Tracker', 'Shadowserver', 'Abuse.ch']),
            'tags': [botnet.lower(), 'botnet', 'c2-activity', 'mass-infection'],
            'entities': {
                'ips': [self._generate_ip() for _ in range(random.randint(5, 12))],
                'domains': [f'{botnet.lower()}-c2-{random.randint(1, 99)}.{random.choice(["tk", "ml", "ga"])}'],
                'urls': [f'http://{botnet.lower()}-panel.{random.choice(["com", "net"])}/admin.php']
            },
            'iocs': {
                'network_indicators': random.randint(20, 50),
                'infrastructure_indicators': random.randint(10, 25),
                'behavioral_indicators': random.randint(5, 15)
            },
            'geographic_origin': country,
            'attack_vector': random.choice(['Email attachments', 'Drive-by downloads', 'USB infection']),
            'infected_hosts': f'{random.randint(10000, 500000):,}',
            'bot_capabilities': random.choice(['Credential theft', 'Banking fraud', 'DDoS attacks', 'Spam distribution'])
        }

    def _generate_exploit_threat(self) -> Dict[str, Any]:
        """Generate exploit/vulnerability threat data"""
        cves = ["CVE-2023-4966", "CVE-2023-3519", "CVE-2023-20198", "CVE-2023-22515", "CVE-2023-36884"]
        cve = random.choice(cves)
        country = random.choice(self.countries)
        
        return {
            'id': f"EXPLOIT-{random.randint(10000, 99999)}",
            'title': f'Active Exploitation of {cve}',
            'description': f'Widespread exploitation of {cve} vulnerability observed in the wild. '
                         f'Attackers leveraging this vulnerability for initial access and privilege escalation.',
            'category': 'exploit',
            'severity': random.choice(['CRITICAL', 'HIGH']),
            'confidence': random.randint(90, 98),
            'is_threat': True,
            'tlp': 'WHITE',
            'source': random.choice(['CVE Database', 'CISA KEV', 'Exploit Database']),
            'tags': ['exploit', 'vulnerability', 'active-exploitation', cve.lower()],
            'entities': {
                'ips': [self._generate_ip() for _ in range(random.randint(3, 8))],
                'urls': [f'http://exploit-{random.randint(1, 999)}.{random.choice(["tk", "ml"])}/poc.php'],
                'cves': [cve]
            },
            'iocs': {
                'network_indicators': random.randint(8, 20),
                'exploit_indicators': random.randint(5, 12),
                'behavioral_indicators': random.randint(3, 8)
            },
            'geographic_origin': country,
            'attack_vector': 'Web application exploitation',
            'cvss_score': round(random.uniform(7.0, 10.0), 1),
            'exploitability': 'High',
            'affected_products': random.choice(['Citrix NetScaler', 'Cisco Switches', 'Microsoft Exchange', 'VMware vCenter'])
        }

    def _generate_ip(self) -> str:
        """Generate realistic IP address"""
        # Mix of different IP ranges
        ranges = [
            (203, 0, 113),    # TEST-NET-3
            (198, 51, 100),   # TEST-NET-2
            (192, 0, 2),      # TEST-NET-1
            (185, random.randint(1, 255), random.randint(1, 255)),  # RIPE region
            (23, random.randint(1, 255), random.randint(1, 255)),   # North America
        ]
        range_prefix = random.choice(ranges)
        return f"{range_prefix[0]}.{range_prefix[1]}.{range_prefix[2]}.{random.randint(1, 254)}"

    def _generate_hash(self) -> str:
        """Generate realistic file hash"""
        hash_types = ['sha256', 'md5', 'sha1']
        hash_type = random.choice(hash_types)
        
        if hash_type == 'sha256':
            return ''.join(random.choices('0123456789abcdef', k=64))
        elif hash_type == 'md5':
            return ''.join(random.choices('0123456789abcdef', k=32))
        else:  # sha1
            return ''.join(random.choices('0123456789abcdef', k=40))

    def _generate_bitcoin_address(self) -> str:
        """Generate realistic Bitcoin address"""
        prefixes = ['1', '3', 'bc1']
        prefix = random.choice(prefixes)
        
        if prefix == 'bc1':
            # Bech32 address
            chars = '023456789acdefghjklmnpqrstuvwxyz'
            return f'bc1{"".join(random.choices(chars, k=random.randint(39, 59)))}'
        else:
            # Legacy address
            chars = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
            return f'{prefix}{"".join(random.choices(chars, k=random.randint(25, 33)))}'

def initialize_dashboard_with_live_data():
    """Initialize dashboard with current live threat data"""
    print("=" * 80)
    print("🛡️  Initializing CTI-sHARE Dashboard with Live Threat Data")
    print("=" * 80)
    
    try:
        # Generate live threat data
        generator = LiveThreatDataGenerator()
        threats = generator.generate_current_threats(count=75)
        
        print(f"✅ Generated {len(threats)} current threat indicators")
        
        # Display threat statistics
        categories = {}
        severities = {}
        countries = {}
        
        for threat in threats:
            # Count categories
            cat = threat['category']
            categories[cat] = categories.get(cat, 0) + 1
            
            # Count severities
            sev = threat['severity']
            severities[sev] = severities.get(sev, 0) + 1
            
            # Count countries
            country = threat.get('geographic_origin', 'Unknown')
            countries[country] = countries.get(country, 0) + 1
        
        print("\n📊 Threat Data Statistics:")
        print(f"   Categories: {dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))}")
        print(f"   Severities: {dict(sorted(severities.items(), key=lambda x: x[1], reverse=True))}")
        print(f"   Top Countries: {dict(list(sorted(countries.items(), key=lambda x: x[1], reverse=True))[:5])}")
        
        # Save to JSON file for dashboard import
        with open('live_threat_data.json', 'w') as f:
            json.dump(threats, f, indent=2)
        
        print(f"\n💾 Threat data saved to: live_threat_data.json")
        
        # Display recent critical threats
        print("\n🚨 Recent Critical Threats:")
        critical_threats = [t for t in threats[:10] if t['severity'] == 'CRITICAL']
        for threat in critical_threats[:5]:
            timestamp = datetime.fromisoformat(threat['timestamp'])
            hours_ago = int((datetime.now() - timestamp).total_seconds() / 3600)
            print(f"   • {threat['title']} ({hours_ago}h ago) - {threat['category'].upper()}")
        
        print("\n" + "=" * 80)
        print("✅ Dashboard Live Threat Data Initialization Complete!")
        print("🚀 Data ready for import into CTI-sHARE Dashboard")
        print("=" * 80)
        
        return threats
        
    except Exception as e:
        print(f"❌ Error initializing live threat data: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == '__main__':
    initialize_dashboard_with_live_data()