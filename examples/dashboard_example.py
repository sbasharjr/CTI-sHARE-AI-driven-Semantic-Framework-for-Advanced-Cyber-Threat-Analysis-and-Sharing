"""
Example of running the threat intelligence dashboard
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dashboard.dashboard import ThreatDashboard
from src.semantic_analysis.semantic_analyzer import ThreatSemanticAnalyzer
from src.realtime.detector import RealTimeThreatDetector


def main():
    print("=" * 80)
    print("Threat Intelligence Dashboard")
    print("=" * 80)
    print()
    
    # Initialize components
    print("Initializing components...")
    semantic_analyzer = ThreatSemanticAnalyzer()
    detector = RealTimeThreatDetector(semantic_analyzer=semantic_analyzer)
    
    # Start real-time detector
    detector.start()
    
    # Initialize dashboard
    dashboard = ThreatDashboard(
        threat_detector=detector,
        semantic_analyzer=semantic_analyzer
    )
    
    print("\nDashboard Endpoints:")
    print("  GET  /                                - Main dashboard page")
    print("  GET  /api/dashboard/stats             - Overall statistics")
    print("  GET  /api/dashboard/threats/recent    - Recent threats")
    print("  GET  /api/dashboard/threats/timeline  - Threat timeline")
    print("  GET  /api/dashboard/threats/categories - Threats by category")
    print("  GET  /api/dashboard/threats/severity  - Threats by severity")
    print("  GET  /api/dashboard/threats/geo       - Geographic distribution")
    print("  GET  /api/dashboard/health            - Health check")
    print()
    
    print("Starting dashboard on http://0.0.0.0:5001")
    print("Press Ctrl+C to stop")
    print("=" * 80)
    print()
    
    try:
        dashboard.run(host='0.0.0.0', port=5001, debug=False)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        detector.stop()
        print("Dashboard stopped.")


if __name__ == "__main__":
    main()
