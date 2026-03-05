#!/usr/bin/env python3
"""
Generate 10,000 threat intelligence datasets for production testing
"""

import json
import random
from datetime import datetime, timedelta

def generate_large_threat_dataset(count=""):
    """Generate a large dataset of realistic threat intelligence data"""
    
    categories = [
        'malware', 'phishing', 'ransomware', 'ddos', 'brute_force',
        'sql_injection', 'xss', 'apt', 'data_breach', 'insider_threat',
        'zero_day', 'botnet', 'cryptojacking', 'social_engineering',
        'privilege_escalation', 'lateral_movement', 'command_control',
        'exfiltration', 'persistence', 'reconnaissance'
    ]
    
    severities = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    severity_weights = [30, 40, 20, 10]  # Distribution percentages
    
    attack_vectors = [
        'Email attachment', 'Malicious URL', 'Compromised credentials',
        'Network vulnerability', 'Social engineering', 'Zero-day exploit',
        'Supply chain attack', 'Watering hole', 'Man-in-the-middle',
        'DNS spoofing', 'Remote code execution', 'Buffer overflow'
    ]
    
    threat_actors = [
        'APT28', 'APT29', 'Lazarus Group', 'Carbanak', 'FIN7',
        'Sandworm', 'OceanLotus', 'Turla', 'Equation Group', 'DarkHotel',
        'Unknown Actor', 'Script Kiddie', 'Organized Crime', 'Nation State',
        'Hacktivist Group', 'Insider Threat'
    ]
    
    ip_ranges = [
        '192.168.{}.{}', '10.0.{}.{}', '172.16.{}.{}',
        '203.0.113.{}', '198.51.100.{}', '45.{}.{}.{}',
        '185.{}.{}.{}', '91.{}.{}.{}'
    ]
    
    domains = [
        'malicious-site.com', 'phishing-portal.net', 'fake-bank.org',
        'evil-domain.ru', 'suspicious-cdn.cn', 'malware-host.xyz',
        'fake-update.com', 'scam-alert.net', 'trojan-download.info'
    ]
    
    threats = []
    start_date = datetime.now() - timedelta(days=30)
    
    for i in range(count):
        # Random timestamp within last 30 days
        random_seconds = random.randint(0, 30 * 24 * 60 * 60)
        timestamp = start_date + timedelta(seconds=random_seconds)
        
        category = random.choice(categories)
        severity = random.choices(severities, weights=severity_weights)[0]
        actor = random.choice(threat_actors)
        vector = random.choice(attack_vectors)
        
        # Generate random IPs
        num_ips = random.randint(1, 5)
        ips = []
        for _ in range(num_ips):
            ip_template = random.choice(ip_ranges)
            if ip_template.count('{}') == 2:
                ip = ip_template.format(random.randint(0, 255), random.randint(1, 254))
            elif ip_template.count('{}') == 3:
                ip = ip_template.format(random.randint(0, 255), random.randint(0, 255), random.randint(1, 254))
            else:
                ip = ip_template.format(random.randint(0, 255))
            ips.append(ip)
        
        # Generate random domains
        num_domains = random.randint(0, 3)
        threat_domains = random.sample(domains, min(num_domains, len(domains)))
        
        # Generate description
        descriptions = [
            f"{category.upper()} activity detected from {actor} using {vector}",
            f"Suspicious {category} behavior identified in network traffic",
            f"{severity} severity {category} attack blocked by security controls",
            f"Potential {category} threat associated with {actor}",
            f"Anomalous {category} pattern detected via {vector}",
            f"Security alert: {category} indicators matched known {actor} TTPs",
            f"Automated detection: {category} threat with {severity} severity",
            f"Threat intelligence: {category} campaign attributed to {actor}"
        ]
        
        threat = {
            'id': f'THREAT-{i+1:06d}',
            'description': random.choice(descriptions),
            'category': category,
            'severity': severity,
            'timestamp': timestamp.isoformat(),
            'is_threat': True,
            'confidence': round(random.uniform(0.6, 0.99), 2),
            'threat_actor': actor,
            'attack_vector': vector,
            'entities': {
                'ips': ips,
                'domains': threat_domains if threat_domains else [],
                'urls': [f'http://{d}/malicious' for d in threat_domains[:2]],
                'file_hashes': [
                    f'{random.randbytes(20).hex()}' if random.random() > 0.5 else None
                ] if random.random() > 0.3 else []
            },
            'mitre_attack': {
                'tactic': random.choice(['Initial Access', 'Execution', 'Persistence', 'Privilege Escalation', 'Defense Evasion', 'Credential Access', 'Discovery', 'Lateral Movement', 'Collection', 'Exfiltration', 'Command and Control']),
                'technique': f'T{random.randint(1000, 1600)}'
            },
            'tags': random.sample(['malware', 'exploit', 'c2', 'exfiltration', 'lateral-movement', 'reconnaissance', 'persistence'], k=random.randint(1, 3)),
            'source': random.choice(['IDS', 'SIEM', 'EDR', 'Firewall', 'Proxy', 'Email Gateway', 'Threat Feed', 'Manual Analysis']),
            'status': random.choice(['active', 'investigating', 'contained', 'resolved']) if random.random() > 0.7 else 'active'
        }
        
        threats.append(threat)
    
    return threats

if __name__ == '__main__':
    print("🔄 Generating 10,000 threat intelligence datasets...")
    print("=" * 80)
    
    threats = generate_large_threat_dataset(10000)
    
    # Save to file
    output_file = 'live_threat_data.json'
    with open(output_file, 'w') as f:
        json.dump(threats, f, indent=2)
    
    print(f"✅ Successfully generated {len(threats):,} threat datasets")
    print(f"💾 Saved to: {output_file}")
    print(f"📊 File size: {len(json.dumps(threats)) / 1024 / 1024:.2f} MB")
    
    # Statistics
    severity_counts = {}
    category_counts = {}
    
    for threat in threats:
        severity = threat['severity']
        category = threat['category']
        
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print("\n📈 Dataset Statistics:")
    print("-" * 80)
    print("\nSeverity Distribution:")
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = severity_counts.get(severity, 0)
        percentage = (count / len(threats)) * 100
        print(f"  {severity:12s}: {count:5,} ({percentage:5.2f}%)")
    
    print("\nTop Categories:")
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for category, count in sorted_categories:
        percentage = (count / len(threats)) * 100
        print(f"  {category:20s}: {count:5,} ({percentage:5.2f}%)")
    
    print("\n" + "=" * 80)
    print("✅ Dataset generation complete!")
    print("💡 Use 'python run_production_server.py' to load the dashboard with this data")
