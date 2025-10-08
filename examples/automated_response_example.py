"""
Example: Automated Threat Response System
"""

from src.response.automated_response import (
    AutomatedResponseSystem,
    ResponseRule,
    BlockIPAction,
    BlockDomainAction,
    QuarantineFileAction,
    IsolateHostAction,
    AlertSecurityTeamAction
)


def example_automated_response():
    """Example of automated threat response"""
    print("=" * 80)
    print("Automated Threat Response System")
    print("=" * 80)
    
    # Initialize response system
    print("\n1. Initializing automated response system...")
    response_system = AutomatedResponseSystem()
    print("   System initialized")
    
    # Create default rules
    print("\n2. Creating default response rules...")
    response_system.create_default_rules()
    print(f"   Created {len(response_system.rules)} default rules")
    
    for rule in response_system.rules:
        print(f"   - {rule.name}: {len(rule.actions)} actions")
    
    # Create custom rule
    print("\n3. Adding custom response rule...")
    custom_rule = ResponseRule(
        name="phishing_response",
        conditions={
            'categories': ['phishing'],
            'min_confidence': 0.75
        },
        actions=[
            BlockDomainAction(),
            AlertSecurityTeamAction(email_addresses=['security@example.com'])
        ]
    )
    response_system.add_rule(custom_rule)
    print("   Custom rule added for phishing threats")
    
    # Simulate threat detection
    print("\n4. Simulating threat detections...")
    print("-" * 80)
    
    # Example 1: Critical threat
    threat1 = {
        'id': 'T001',
        'text': 'Critical ransomware attack detected',
        'severity': 'CRITICAL',
        'confidence': 0.95,
        'category': 'malware',
        'entities': {
            'ips': ['192.168.1.100', '10.0.0.50'],
            'domains': ['malicious.com'],
            'hashes': ['abc123def456']
        },
        'affected_hosts': ['workstation-01', 'server-02']
    }
    
    print("\n   Threat #1: Critical Ransomware Attack")
    print(f"   - Severity: {threat1['severity']}")
    print(f"   - Confidence: {threat1['confidence']:.2%}")
    
    response1 = response_system.process_threat(threat1)
    
    print(f"\n   Response Summary:")
    print(f"   - Rules matched: {len(response1['rules_matched'])}")
    for rule_name in response1['rules_matched']:
        print(f"      • {rule_name}")
    
    print(f"   - Actions executed: {len(response1['actions_executed'])}")
    for action in response1['actions_executed']:
        print(f"      • {action['action']}: {'✓' if action['success'] else '✗'}")
    
    # Example 2: High severity phishing
    threat2 = {
        'id': 'T002',
        'text': 'Phishing campaign targeting employees',
        'severity': 'HIGH',
        'confidence': 0.85,
        'category': 'phishing',
        'entities': {
            'domains': ['fake-login.com', 'phish-site.net'],
            'emails': ['attacker@evil.com']
        }
    }
    
    print("\n" + "-" * 80)
    print("\n   Threat #2: Phishing Campaign")
    print(f"   - Severity: {threat2['severity']}")
    print(f"   - Confidence: {threat2['confidence']:.2%}")
    
    response2 = response_system.process_threat(threat2)
    
    print(f"\n   Response Summary:")
    print(f"   - Rules matched: {len(response2['rules_matched'])}")
    for rule_name in response2['rules_matched']:
        print(f"      • {rule_name}")
    
    print(f"   - Actions executed: {len(response2['actions_executed'])}")
    for action in response2['actions_executed']:
        print(f"      • {action['action']}: {'✓' if action['success'] else '✗'}")
    
    # Example 3: Medium severity threat (shouldn't trigger critical rules)
    threat3 = {
        'id': 'T003',
        'text': 'Suspicious network activity detected',
        'severity': 'MEDIUM',
        'confidence': 0.60,
        'category': 'unknown',
        'entities': {
            'ips': ['203.0.113.10']
        }
    }
    
    print("\n" + "-" * 80)
    print("\n   Threat #3: Suspicious Network Activity")
    print(f"   - Severity: {threat3['severity']}")
    print(f"   - Confidence: {threat3['confidence']:.2%}")
    
    response3 = response_system.process_threat(threat3)
    
    print(f"\n   Response Summary:")
    print(f"   - Rules matched: {len(response3['rules_matched'])}")
    if response3['rules_matched']:
        for rule_name in response3['rules_matched']:
            print(f"      • {rule_name}")
    else:
        print(f"      No rules matched (severity/confidence below thresholds)")
    
    # Show statistics
    print("\n" + "=" * 80)
    print("5. System Statistics:")
    print("=" * 80)
    
    stats = response_system.get_statistics()
    
    print(f"   Total responses: {stats['total_responses']}")
    print(f"   Successful responses: {stats['successful_responses']}")
    print(f"   Success rate: {stats['success_rate']*100:.1f}%")
    print(f"   Active rules: {stats['active_rules']}")
    
    print(f"\n   Actions executed:")
    for action_name, count in stats['action_counts'].items():
        print(f"      • {action_name}: {count}")
    
    # Show response history
    print("\n6. Recent response history:")
    print("-" * 80)
    
    history = response_system.get_response_history(limit=3)
    for i, response in enumerate(history, 1):
        print(f"\n   Response #{i}:")
        print(f"   - Threat ID: {response['threat_id']}")
        print(f"   - Timestamp: {response['timestamp']}")
        print(f"   - Rules matched: {', '.join(response['rules_matched'])}")
        print(f"   - Actions: {len(response['actions_executed'])}")
        print(f"   - Success: {'✓' if response['success'] else '✗'}")
    
    print("\n" + "=" * 80)
    print("Automated Response System Demo Complete")
    print("=" * 80)


if __name__ == "__main__":
    try:
        example_automated_response()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
