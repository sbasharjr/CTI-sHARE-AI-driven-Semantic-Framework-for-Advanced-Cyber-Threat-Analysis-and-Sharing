"""
Example: Blockchain-based Threat Intelligence Verification
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.blockchain.verification import BlockchainVerificationService
from datetime import datetime


def example_blockchain_verification():
    """Example of blockchain verification for threat intelligence"""
    print("=" * 80)
    print("Blockchain-based Threat Intelligence Verification")
    print("=" * 80)
    
    # Initialize blockchain service
    print("\n1. Initializing blockchain verification service...")
    blockchain_service = BlockchainVerificationService(difficulty=2)
    print("   Blockchain initialized with genesis block")
    
    # Submit threats to blockchain
    print("\n2. Submitting threat intelligence to blockchain...")
    print("-" * 80)
    
    threats = [
        {
            'id': 'THREAT-001',
            'timestamp': datetime.now().isoformat(),
            'severity': 'CRITICAL',
            'category': 'ransomware',
            'description': 'Ransomware attack on healthcare facility',
            'entities': {
                'ips': ['192.168.1.50'],
                'domains': ['malware-c2.com'],
                'hashes': ['abc123def456789']
            },
            'source': 'Organization A',
            'confidence': 0.95
        },
        {
            'id': 'THREAT-002',
            'timestamp': datetime.now().isoformat(),
            'severity': 'HIGH',
            'category': 'phishing',
            'description': 'Phishing campaign targeting financial sector',
            'entities': {
                'domains': ['fake-bank.com'],
                'emails': ['attacker@evil.com']
            },
            'source': 'Organization B',
            'confidence': 0.88
        },
        {
            'id': 'THREAT-003',
            'timestamp': datetime.now().isoformat(),
            'severity': 'MEDIUM',
            'category': 'malware',
            'description': 'Trojan detected in software supply chain',
            'entities': {
                'hashes': ['xyz789abc123def']
            },
            'source': 'Organization C',
            'confidence': 0.75
        }
    ]
    
    threat_hashes = {}
    for threat in threats:
        threat_hash = blockchain_service.submit_threat(threat)
        threat_hashes[threat['id']] = threat_hash
        print(f"   • {threat['id']}: {threat['description'][:50]}...")
        print(f"     Hash: {threat_hash}")
    
    # Check pending threats
    print(f"\n   Pending threats: {len(blockchain_service.blockchain.pending_threats)}")
    
    # Commit threats to blockchain
    print("\n3. Mining block to commit threats to blockchain...")
    new_block = blockchain_service.commit_threats()
    
    if new_block:
        print(f"   ✓ Block #{new_block.index} mined successfully")
        print(f"   Block hash: {new_block.hash}")
        print(f"   Previous hash: {new_block.previous_hash}")
        print(f"   Threats in block: {new_block.data['count']}")
        print(f"   Nonce: {new_block.nonce}")
    
    # Verify blockchain integrity
    print("\n4. Verifying blockchain integrity...")
    is_valid = blockchain_service.blockchain.verify_chain()
    print(f"   Blockchain valid: {'✓' if is_valid else '✗'}")
    
    # Verify individual threats
    print("\n5. Verifying individual threat authenticity...")
    print("-" * 80)
    
    for threat_id, threat_hash in threat_hashes.items():
        is_authentic = blockchain_service.verify_threat_authenticity(threat_id, threat_hash)
        print(f"   • {threat_id}: {'✓ VERIFIED' if is_authentic else '✗ NOT VERIFIED'}")
    
    # Try verifying with wrong hash
    print("\n6. Testing verification with tampered data...")
    fake_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    is_authentic = blockchain_service.verify_threat_authenticity('THREAT-001', fake_hash)
    print(f"   Tampered threat verification: {'✓' if is_authentic else '✗ FAILED (as expected)'}")
    
    # Retrieve threat from blockchain
    print("\n7. Retrieving verified threat from blockchain...")
    verified_threat = blockchain_service.get_verified_threat('THREAT-001')
    
    if verified_threat:
        print(f"   ✓ Threat retrieved successfully")
        print(f"   - Threat ID: {verified_threat['threat']['threat_id']}")
        print(f"   - Severity: {verified_threat['threat']['severity']}")
        print(f"   - Block Index: {verified_threat['block_index']}")
        print(f"   - Block Hash: {verified_threat['block_hash']}")
        print(f"   - Verified: {verified_threat['verified']}")
    
    # Get blockchain statistics
    print("\n8. Blockchain Statistics:")
    print("=" * 80)
    
    stats = blockchain_service.blockchain.get_chain_statistics()
    
    print(f"   Total blocks: {stats['total_blocks']}")
    print(f"   Total threats: {stats['total_threats']}")
    print(f"   Pending threats: {stats['pending_threats']}")
    print(f"   Chain valid: {'✓' if stats['is_valid'] else '✗'}")
    print(f"   Latest block hash: {stats['latest_block_hash']}")
    
    # Query threats by severity
    print("\n9. Querying threats by criteria...")
    print("-" * 80)
    
    critical_threats = blockchain_service.blockchain.get_threats_by_severity('CRITICAL')
    print(f"   Critical threats: {len(critical_threats)}")
    for threat in critical_threats:
        print(f"      • {threat['threat_id']}: {threat['category']}")
    
    phishing_threats = blockchain_service.blockchain.get_threats_by_category('phishing')
    print(f"\n   Phishing threats: {len(phishing_threats)}")
    for threat in phishing_threats:
        print(f"      • {threat['threat_id']}: {threat['severity']}")
    
    # Export blockchain
    print("\n10. Exporting blockchain...")
    chain_export = blockchain_service.blockchain.export_chain()
    print(f"   Exported {len(chain_export)} blocks")
    print(f"   Export size: ~{len(str(chain_export))} bytes")
    
    print("\n" + "=" * 80)
    print("Key Benefits of Blockchain Verification:")
    print("=" * 80)
    print("✓ Immutability: Threat data cannot be altered once recorded")
    print("✓ Transparency: All parties can verify threat intelligence")
    print("✓ Trust: No central authority needed")
    print("✓ Auditability: Complete history of threat submissions")
    print("✓ Integrity: Cryptographic verification of data")
    print("=" * 80)


if __name__ == "__main__":
    try:
        example_blockchain_verification()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
