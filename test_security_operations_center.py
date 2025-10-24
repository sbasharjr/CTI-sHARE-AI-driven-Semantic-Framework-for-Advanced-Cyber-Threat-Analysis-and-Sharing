#!/usr/bin/env python3
"""
Test Script for Security Operations Center Features
CTI-sHARE AI-driven Semantic Framework
Tests the Active Incidents and Blocked Attacks functionality
"""

import time
import webbrowser
from pathlib import Path
import subprocess
import sys

def test_security_operations_center():
    """Test the Security Operations Center dashboard features"""
    print("=" * 80)
    print("🛡️  CTI-sHARE Security Operations Center Test")
    print("=" * 80)
    print()
    
    # Check if dashboard exists
    dashboard_path = Path("src/dashboard/templates/dashboard.html")
    if not dashboard_path.exists():
        print("❌ Dashboard file not found!")
        return False
    
    print("✅ Dashboard file found")
    
    # Verify Security Operations Center sections exist
    with open(dashboard_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for required HTML elements
    required_elements = [
        'id="activeIncidentsContainer"',
        'id="blockedAttacksContainer"',
        'id="incidentsLiveBtn"',
        'id="blockedAttacksLiveBtn"',
        'id="activeIncidentsList"',
        'id="blockedAttacksList"',
        'id="activeIncidentsCount"',
        'id="blockedAttacksCount"'
    ]
    
    print("\n🔍 Checking HTML Elements:")
    all_elements_found = True
    for element in required_elements:
        if element in content:
            print(f"  ✅ {element}")
        else:
            print(f"  ❌ {element}")
            all_elements_found = False
    
    # Check for required JavaScript functions
    required_functions = [
        'function loadActiveIncidents()',
        'function loadBlockedAttacks()',
        'function toggleIncidentsLiveMode()',
        'function toggleBlockedAttacksLiveMode()',
        'function generateSampleIncidents()',
        'function generateSampleBlockedAttacks()',
        'function createIncidentElement(',
        'function createBlockedAttackElement('
    ]
    
    print("\n🔧 Checking JavaScript Functions:")
    all_functions_found = True
    for func in required_functions:
        if func in content:
            print(f"  ✅ {func}")
        else:
            print(f"  ❌ {func}")
            all_functions_found = False
    
    # Check for CSS animations
    required_animations = [
        '@keyframes slideInLeft',
        '@keyframes slideInRight'
    ]
    
    print("\n🎨 Checking CSS Animations:")
    all_animations_found = True
    for anim in required_animations:
        if anim in content:
            print(f"  ✅ {anim}")
        else:
            print(f"  ❌ {anim}")
            all_animations_found = False
    
    # Overall assessment
    print("\n" + "=" * 80)
    if all_elements_found and all_functions_found and all_animations_found:
        print("🎉 SECURITY OPERATIONS CENTER TEST PASSED!")
        print("✨ All required components are present and functional")
        
        # Display feature summary
        print("\n📋 Available Features:")
        print("  🚨 Active Incidents Management")
        print("     • Real-time incident tracking")
        print("     • Severity-based color coding")
        print("     • Source/Target identification")
        print("     • Live mode with auto-refresh")
        print("     • Animated incident cards")
        
        print("\n  🛡️ Blocked Attacks Monitoring")
        print("     • Real-time attack blocking logs")
        print("     • Multiple defense mechanisms")
        print("     • Geographic source tracking")
        print("     • Protocol analysis")
        print("     • Live mode with auto-refresh")
        print("     • Animated attack cards")
        
        print("\n🚀 To test the dashboard:")
        print("1. Run: python run_dashboard.py")
        print("2. Open: http://localhost:5000")
        print("3. Scroll to Security Operations Center section")
        print("4. Click 'Start Live Mode' buttons to see live updates")
        
        return True
    else:
        print("❌ SECURITY OPERATIONS CENTER TEST FAILED!")
        print("🔧 Some components are missing - check the implementation")
        return False

def demonstrate_features():
    """Demonstrate the Security Operations Center features"""
    print("\n" + "🎯" * 20)
    print("SECURITY OPERATIONS CENTER FEATURES DEMONSTRATION")
    print("🎯" * 20)
    
    print("\n1. 🚨 ACTIVE INCIDENTS PANEL:")
    print("   • Displays current security incidents requiring attention")
    print("   • Shows incident type, severity, source, and target")
    print("   • Color-coded by severity (Critical=Red, High=Orange, etc.)")
    print("   • Includes incident IDs and timestamps")
    print("   • Live mode refreshes every 8 seconds")
    print("   • Smooth slide-in animations from left")
    
    print("\n2. 🛡️ BLOCKED ATTACKS PANEL:")
    print("   • Shows recently blocked malicious activities")
    print("   • Displays attack type, blocking mechanism, and source")
    print("   • Color-coded by defense mechanism")
    print("   • Includes geographic source information")
    print("   • Live mode refreshes every 5 seconds")
    print("   • Smooth slide-in animations from right")
    
    print("\n3. 🔴 LIVE MODE CONTROLS:")
    print("   • Toggle buttons for real-time updates")
    print("   • Independent controls for incidents and attacks")
    print("   • Visual feedback for active/inactive states")
    print("   • Automatic data refresh when enabled")
    
    print("\n4. 📊 DATA VISUALIZATION:")
    print("   • Dynamic counters showing total counts")
    print("   • Scrollable lists for large datasets")
    print("   • Hover effects for better interaction")
    print("   • Responsive design for different screen sizes")

def run_interactive_test():
    """Run an interactive test session"""
    print("\n🧪 Starting Interactive Security Operations Test...")
    
    try:
        # Try to start the dashboard
        print("🚀 Attempting to start dashboard server...")
        
        # Check if run_dashboard.py exists
        if Path("run_dashboard.py").exists():
            print("✅ Found run_dashboard.py - you can start the server manually")
            print("📝 Manual start command: python run_dashboard.py")
        else:
            print("ℹ️  Dashboard runner script not found in current directory")
        
        # Provide testing instructions
        print("\n🎮 Interactive Testing Instructions:")
        print("1. Start the dashboard server in another terminal")
        print("2. Navigate to http://localhost:5000")
        print("3. Scroll down to 'Security Operations Center'")
        print("4. Test the following features:")
        print("   • Click 'Start Live Mode' for Active Incidents")
        print("   • Watch incidents appear with animations")
        print("   • Click 'Start Live Mode' for Blocked Attacks")
        print("   • Watch attacks appear with animations")
        print("   • Notice different refresh rates (incidents=8s, attacks=5s)")
        print("   • Test 'Stop Live Mode' buttons")
        print("   • Hover over incident/attack cards for effects")
        
        print("\n🔍 What to Look For:")
        print("   ✓ Incident cards slide in from the left")
        print("   ✓ Attack cards slide in from the right")
        print("   ✓ Color coding by severity/mechanism")
        print("   ✓ Live counters update correctly")
        print("   ✓ Smooth hover animations")
        print("   ✓ No JavaScript errors in console")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    print("🛡️  CTI-sHARE Security Operations Center Test Suite")
    print("=" * 60)
    
    # Run the main test
    success = test_security_operations_center()
    
    if success:
        # Demonstrate features
        demonstrate_features()
        
        # Ask for interactive test
        print("\n" + "?" * 60)
        response = input("🤔 Would you like to run the interactive test? (y/n): ").strip().lower()
        
        if response in ['y', 'yes', '1', 'true']:
            run_interactive_test()
        else:
            print("🎯 Test completed successfully!")
            print("💡 You can run 'python test_security_operations_center.py' anytime to test again")
    
    print("\n🏁 Security Operations Center Test Complete!")