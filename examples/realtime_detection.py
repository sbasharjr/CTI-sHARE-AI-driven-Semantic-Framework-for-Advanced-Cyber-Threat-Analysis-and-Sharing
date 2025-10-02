"""
Example of real-time threat detection
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.realtime.detector import RealTimeThreatDetector, ThreatMonitor
from src.semantic_analysis.semantic_analyzer import ThreatSemanticAnalyzer
from src.utils.data_loader import ThreatDataLoader


def main():
    print("=" * 80)
    print("Real-time Threat Detection System")
    print("=" * 80)
    print()
    
    # Initialize components
    print("Initializing components...")
    semantic_analyzer = ThreatSemanticAnalyzer()
    detector = RealTimeThreatDetector(semantic_analyzer=semantic_analyzer)
    monitor = ThreatMonitor(detector)
    
    # Start real-time detection
    print("Starting real-time threat detection...")
    detector.start()
    print()
    
    # Create sample threat data
    loader = ThreatDataLoader()
    sample_data = loader.create_sample_data(num_samples=20)
    
    # Simulate real-time threat feed
    print("Simulating real-time threat feed...")
    for i, row in sample_data.iterrows():
        threat_data = {
            'text': row['summary'],
            'features': [row['severity'], i]
        }
        
        detector.add_threat_data(threat_data)
        print(f"  [+] Submitted threat {i+1}/20")
        time.sleep(0.5)  # Simulate delay between threats
    
    # Wait for processing to complete
    print("\nWaiting for threat processing to complete...")
    time.sleep(3)
    
    # Get statistics
    print("\n" + "=" * 80)
    print("Detection Statistics")
    print("=" * 80)
    stats = detector.get_statistics()
    print(f"Total processed: {stats['total_processed']}")
    print(f"Threats detected: {stats['threats_detected']}")
    print(f"Detection rate: {stats['detection_rate']:.2%}")
    print(f"Average confidence: {stats['average_confidence']:.2%}")
    
    # Get recent threats
    print("\n" + "=" * 80)
    print("Recent Detected Threats")
    print("=" * 80)
    recent_threats = detector.get_recent_threats(limit=5)
    for i, threat in enumerate(recent_threats, 1):
        print(f"\nThreat {i}:")
        print(f"  Timestamp: {threat['timestamp']}")
        print(f"  Confidence: {threat['confidence']:.2%}")
        if 'semantic_analysis' in threat and 'severity' in threat['semantic_analysis']:
            severity = threat['semantic_analysis']['severity']
            print(f"  Severity: {severity['severity']} (level {severity['level']}/5)")
    
    # Get alerts
    print("\n" + "=" * 80)
    print("Recent Alerts")
    print("=" * 80)
    alerts = monitor.get_alerts(limit=5)
    for i, alert in enumerate(alerts, 1):
        print(f"\nAlert {i}:")
        print(f"  {alert['summary']}")
        print(f"  Time: {alert['timestamp']}")
    
    # Stop detector
    print("\n\nStopping real-time detection...")
    detector.stop()
    
    print("\n" + "=" * 80)
    print("Real-time detection demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
