#!/usr/bin/env python3
"""
Test Alert System and Push Notifications
CTI-sHARE Threat Intelligence Dashboard
"""

import time
import webbrowser
import subprocess
import sys
from pathlib import Path

def test_alert_system():
    """
    Test the complete alert system implementation including:
    - Push notifications with hide panel
    - Header positioning
    - Interactive controls
    - Different alert types
    """
    
    print("🔔 Testing CTI-sHARE Alert System")
    print("=" * 50)
    
    # Check if the dashboard file exists
    dashboard_path = Path(__file__).parent / "src" / "dashboard" / "templates" / "dashboard.html"
    
    if not dashboard_path.exists():
        print("❌ Dashboard template not found!")
        return False
    
    print("✅ Dashboard template found")
    
    # Start the dashboard server
    print("\n🚀 Starting dashboard server...")
    
    try:
        # Try to start the enhanced dashboard
        process = subprocess.Popen([
            sys.executable, "run_enhanced_dashboard.py"
        ], capture_output=False)
        
        print("⏳ Waiting for server to start...")
        time.sleep(3)
        
        # Open the dashboard in browser
        dashboard_url = "http://localhost:8080"
        print(f"🌐 Opening dashboard: {dashboard_url}")
        webbrowser.open(dashboard_url)
        
        print("\n🎯 ALERT SYSTEM TEST INSTRUCTIONS:")
        print("-" * 40)
        print("1. 🔔 Look for the notification bell in the header (top-right)")
        print("2. 📊 Scroll down to 'Alert System Testing' section")
        print("3. 🚨 Click 'Critical Alert' button to test push notifications")
        print("4. ⚠️ Click 'Warning Alert' button to test panel notifications")
        print("5. ℹ️ Click 'Info Alert' button to test info notifications")
        print("6. 🔔 Click the bell icon to open/close the notification panel")
        print("7. ✅ Test 'Mark All Read' and 'Clear All' functions")
        print("8. ⏰ Wait for automatic alerts (every 10-25 seconds)")
        print("9. 📱 Test push notification close buttons")
        print("10. 🎛️ Test notification action buttons (Investigate, Block, etc.)")
        
        print("\n🎨 FEATURES TO VERIFY:")
        print("-" * 25)
        print("✅ Bell icon with notification badge")
        print("✅ Sliding notification panel")  
        print("✅ Push notifications with auto-hide")
        print("✅ Different alert types (Critical, Warning, Info)")
        print("✅ Interactive notification actions")
        print("✅ Mark as read/unread functionality")
        print("✅ Time stamps and relative time display")
        print("✅ Responsive design and animations")
        print("✅ Click outside to close panel")
        print("✅ Browser notification permission request")
        
        print(f"\n🎉 Alert System is ready for testing!")
        print(f"📍 Dashboard URL: {dashboard_url}")
        print("Press Ctrl+C to stop the server when done testing...")
        
        # Keep the server running
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n⏹️ Stopping server...")
            process.terminate()
            process.wait()
            print("✅ Server stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return False

def validate_alert_features():
    """Validate that all alert system features are properly implemented"""
    
    print("\n🔍 VALIDATING ALERT SYSTEM FEATURES:")
    print("-" * 40)
    
    dashboard_path = Path(__file__).parent / "src" / "dashboard" / "templates" / "dashboard.html"
    
    try:
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key components
        checks = [
            ('Alert Bell Icon', 'class="alert-bell"'),
            ('Notification Panel', 'class="notification-panel"'),
            ('Push Notification Styles', 'class="push-notification"'),
            ('Alert Badge', 'class="alert-badge"'),
            ('Notification Controls', 'markAllAsRead()'),
            ('Push Notification Function', 'showPushNotification'),
            ('Alert System Initialization', 'initializeAlertSystem'),
            ('Notification Types (Critical)', 'type: \'critical\''),
            ('Notification Types (Warning)', 'type: \'warning\''),
            ('Notification Types (Info)', 'type: \'info\''),
            ('Interactive Actions', 'performNotificationAction'),
            ('Time Formatting', 'getTimeAgo'),
            ('Auto Alert Generation', 'generateRandomAlert'),
            ('Panel Toggle', 'toggleNotificationPanel'),
        ]
        
        for name, pattern in checks:
            if pattern in content:
                print(f"✅ {name}")
            else:
                print(f"❌ {name}")
        
        print(f"\n📊 Feature Implementation Status:")
        passed = sum(1 for _, pattern in checks if pattern in content)
        total = len(checks)
        print(f"✅ {passed}/{total} features implemented ({(passed/total)*100:.1f}%)")
        
        if passed == total:
            print("🎉 All alert system features are properly implemented!")
            return True
        else:
            print(f"⚠️ {total-passed} features need attention")
            return False
            
    except Exception as e:
        print(f"❌ Error validating features: {e}")
        return False

if __name__ == "__main__":
    print("🔔 CTI-sHARE Alert System Test Suite")
    print("=" * 50)
    
    # Validate implementation
    validation_passed = validate_alert_features()
    
    if validation_passed:
        print("\n✅ Validation passed! Starting interactive test...")
        test_alert_system()
    else:
        print("\n❌ Validation failed! Please check implementation.")
        sys.exit(1)
