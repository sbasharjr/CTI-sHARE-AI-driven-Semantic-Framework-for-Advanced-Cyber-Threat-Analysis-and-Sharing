"""
Example of running the threat sharing API server
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.threat_sharing.api import ThreatSharingAPI
from src.semantic_analysis.semantic_analyzer import ThreatSemanticAnalyzer
from src.realtime.detector import RealTimeThreatDetector


def main():
    print("=" * 80)
    print("Threat Intelligence Sharing API Server")
    print("=" * 80)
    print()
    
    # Initialize components
    print("Initializing components...")
    semantic_analyzer = ThreatSemanticAnalyzer()
    detector = RealTimeThreatDetector(semantic_analyzer=semantic_analyzer)
    
    # Start real-time detector
    detector.start()
    
    # Initialize API
    api = ThreatSharingAPI(
        detector=detector,
        semantic_analyzer=semantic_analyzer
    )
    
    print("\nAPI Endpoints:")
    print("  GET  /api/health              - Health check")
    print("  POST /api/threats/submit      - Submit threat intelligence")
    print("  GET  /api/threats             - Get threats (paginated)")
    print("  POST /api/threats/analyze     - Analyze threat text")
    print("  GET  /api/statistics          - Get detection statistics")
    print("  GET  /api/threats/search      - Search threats")
    print()
    
    print("Starting API server on http://0.0.0.0:5000")
    print("Press Ctrl+C to stop")
    print("=" * 80)
    print()
    
    try:
        api.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        detector.stop()
        print("Server stopped.")


if __name__ == "__main__":
    main()
