#!/usr/bin/env python3
"""
Production server runner using Waitress (Windows-compatible WSGI server)
"""

import os
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir / 'src'))

def run_production_server():
    """Run the CTI-sHARE dashboard using Waitress production server"""
    try:
        from waitress import serve
        from wsgi import application
        
        print("=" * 80)
        print("🛡️  CTI-sHARE Dashboard - Production Server (Waitress)")
        print("=" * 80)
        print("🚀 Starting production WSGI server...")
        print("📡 Server URL: http://localhost:5001")
        print("🔧 WSGI Server: Waitress")
        print("⚡ Environment: Production")
        print("=" * 80)
        
        # Production server configuration
        serve(
            application,
            host='0.0.0.0',
            port=5001,
            threads=6,                    # Number of threads
            connection_limit=1000,        # Maximum number of simultaneous connections
            cleanup_interval=30,          # Seconds between cleanup operations
            channel_timeout=120,          # Channel timeout in seconds
            log_socket_errors=True,       # Log socket errors
            max_request_header_size=262144,  # 256KB max request header
            max_request_body_size=1073741824,  # 1GB max request body
            expose_tracebacks=False,      # Don't expose tracebacks in production
            asyncore_use_poll=True,       # Use poll() instead of select()
            ipv4=True,                    # Enable IPv4
            ipv6=False,                   # Disable IPv6 for simplicity
        )
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Install with: pip install waitress")
        return False
    except Exception as e:
        print(f"❌ Error starting production server: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run_production_server()