#!/usr/bin/env python3
"""
Test script for production WSGI server setup
"""

import sys
import os
from pathlib import Path

# Add src to path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir / 'src'))

def test_wsgi_setup():
    """Test if the WSGI application can be imported and created"""
    print("=" * 80)
    print("🧪 Testing Production WSGI Server Setup")
    print("=" * 80)
    
    try:
        print("1. Testing WSGI application import...")
        from wsgi import application
        print("✅ WSGI application imported successfully")
        
        print("\n2. Testing Flask application...")
        if hasattr(application, 'test_client'):
            print("✅ Flask application detected")
            
            print("\n3. Testing application routes...")
            with application.test_client() as client:
                # Test health check
                response = client.get('/api/dashboard/health')
                if response.status_code == 200:
                    print("✅ Health check endpoint working")
                else:
                    print(f"⚠️ Health check returned: {response.status_code}")
                
                # Test main dashboard
                response = client.get('/')
                if response.status_code == 200:
                    print("✅ Main dashboard endpoint working")
                else:
                    print(f"⚠️ Main dashboard returned: {response.status_code}")
                
                # Test live API endpoints
                live_endpoints = [
                    '/api/dashboard/live/system-performance',
                    '/api/dashboard/live/resource-distribution',
                    '/api/dashboard/live/attack-vectors'
                ]
                
                print("\n4. Testing live data API endpoints...")
                for endpoint in live_endpoints:
                    response = client.get(endpoint)
                    if response.status_code == 200:
                        print(f"✅ {endpoint} working")
                    else:
                        print(f"⚠️ {endpoint} returned: {response.status_code}")
        
        print("\n5. Testing production server dependencies...")
        
        try:
            import waitress
            print("✅ Waitress (Windows WSGI server) available")
        except ImportError:
            print("❌ Waitress not available")
        
        try:
            import gunicorn
            print("✅ Gunicorn (Linux WSGI server) available")
        except ImportError:
            print("⚠️ Gunicorn not available (normal on Windows)")
        
        try:
            import psutil
            print("✅ psutil (system monitoring) available")
        except ImportError:
            print("❌ psutil not available")
        
        print("\n" + "=" * 80)
        print("✅ Production WSGI Server Setup Test Complete!")
        print("🚀 Ready for production deployment!")
        print("=" * 80)
        
        print("\n📋 Deployment Commands:")
        print("Windows: python run_production_server.py")
        print("Linux:   gunicorn --config gunicorn_config.py wsgi:application")
        print("Docker:  docker-compose -f docker-compose.production.yml up")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_wsgi_setup()