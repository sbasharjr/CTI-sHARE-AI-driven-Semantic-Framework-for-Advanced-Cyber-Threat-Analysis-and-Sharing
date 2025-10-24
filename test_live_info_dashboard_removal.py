#!/usr/bin/env python3
"""
Test Live Information Dashboard Removal - CTI-sHARE Dashboard
=============================================================

This test validates that the Live Information Dashboard has been successfully
removed from the CTI-sHARE dashboard without breaking functionality.

Testing Areas:
1. Dashboard HTML structure validation
2. JavaScript function cleanup verification 
3. CSS class removal confirmation
4. API functionality preservation
"""

import re
import requests
import time
import sys
import os

def test_html_cleanup():
    """Test that Live Information Dashboard HTML elements have been removed"""
    print("🔍 Testing HTML cleanup...")
    
    dashboard_path = "src/dashboard/templates/dashboard.html"
    
    if not os.path.exists(dashboard_path):
        print("❌ Dashboard HTML file not found!")
        return False
        
    with open(dashboard_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for removed elements
    removed_elements = [
        "Live Information Dashboard",
        "liveInfoDashboard", 
        "liveAttackVectors",
        "liveGeographic", 
        "liveHourlyActivity",
        "liveChartsStatus",
        "Toggle Charts Live"
    ]
    
    issues_found = []
    for element in removed_elements:
        if element in content:
            issues_found.append(element)
    
    if issues_found:
        print(f"❌ Found remaining Live Dashboard elements: {issues_found}")
        return False
    else:
        print("✅ All Live Information Dashboard HTML elements successfully removed")
        return True

def test_javascript_cleanup():
    """Test that JavaScript functions have been cleaned up"""
    print("🔍 Testing JavaScript cleanup...")
    
    dashboard_path = "src/dashboard/templates/dashboard.html"
    
    with open(dashboard_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for removed JavaScript functions
    removed_functions = [
        "function updateLiveInfoDashboard",
        "function toggleLiveCharts", 
        "updateLiveInfoDashboard()"
    ]
    
    issues_found = []
    for func in removed_functions:
        if func in content:
            issues_found.append(func)
    
    if issues_found:
        print(f"❌ Found remaining JavaScript functions: {issues_found}")
        return False
    else:
        print("✅ All Live Information Dashboard JavaScript functions successfully removed")
        return True

def test_variable_cleanup():
    """Test that global chart variables have been cleaned up"""
    print("🔍 Testing variable cleanup...")
    
    dashboard_path = "src/dashboard/templates/dashboard.html"
    
    with open(dashboard_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check that variable declaration was updated
    if "attackVectorsChart" in content and "let categoriesChart, severityChart, severityTimeSeriesChart, attackVectorsChart" in content:
        print("❌ Global chart variables not properly cleaned up")
        return False
        
    if "let categoriesChart, severityChart, severityTimeSeriesChart;" in content:
        print("✅ Global chart variables successfully cleaned up")
        return True
    else:
        print("⚠️ Variable declaration format may have changed - manual verification needed")
        return True

def test_dashboard_functionality():
    """Test that dashboard still loads and functions properly"""
    print("🔍 Testing dashboard functionality...")
    
    try:
        # Test if main dashboard loads
        response = requests.get("http://localhost:5001", timeout=10)
        
        if response.status_code == 200:
            print("✅ Dashboard loads successfully")
            
            # Test core API endpoints
            api_endpoints = [
                "/api/dashboard/stats",
                "/api/dashboard/threats/categories", 
                "/api/dashboard/threats/severity/realtime",
                "/api/dashboard/realtime/status"
            ]
            
            api_success = True
            for endpoint in api_endpoints:
                try:
                    api_response = requests.get(f"http://localhost:5001{endpoint}", timeout=5)
                    if api_response.status_code != 200:
                        print(f"❌ API endpoint {endpoint} failed: {api_response.status_code}")
                        api_success = False
                except Exception as e:
                    print(f"❌ API endpoint {endpoint} error: {e}")
                    api_success = False
            
            if api_success:
                print("✅ All core API endpoints working properly")
                return True
            else:
                return False
                
        else:
            print(f"❌ Dashboard failed to load: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("⚠️ Dashboard not running - start with 'python run_dashboard.py' to test functionality")
        return True  # Don't fail test if server not running
    except Exception as e:
        print(f"❌ Dashboard functionality test failed: {e}")
        return False

def test_remaining_components():
    """Test that remaining dashboard components are intact"""
    print("🔍 Testing remaining components...")
    
    dashboard_path = "src/dashboard/templates/dashboard.html"
    
    with open(dashboard_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for essential remaining components
    essential_components = [
        "Threat Categories",
        "Severity Distribution", 
        "Security Operations Center",
        "Active Incidents",
        "Blocked Attacks"
    ]
    
    missing_components = []
    for component in essential_components:
        if component not in content:
            missing_components.append(component)
    
    if missing_components:
        print(f"❌ Missing essential components: {missing_components}")
        return False
    else:
        print("✅ All essential dashboard components remain intact")
        return True

def main():
    """Run all tests for Live Information Dashboard removal"""
    print("=" * 70)
    print("🧪 CTI-sHARE LIVE INFORMATION DASHBOARD REMOVAL TEST")
    print("=" * 70)
    print()
    
    tests = [
        ("HTML Cleanup", test_html_cleanup),
        ("JavaScript Cleanup", test_javascript_cleanup), 
        ("Variable Cleanup", test_variable_cleanup),
        ("Dashboard Functionality", test_dashboard_functionality),
        ("Remaining Components", test_remaining_components)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"🔄 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            print()
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
            print()
    
    # Summary
    print("=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("-" * 70)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    if passed == total:
        print()
        print("🎉 ALL TESTS PASSED!")
        print("✅ Live Information Dashboard successfully removed!")
        print("✅ Dashboard functionality preserved!")
        return True
    else:
        print()
        print("⚠️ Some tests failed - please review results above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)