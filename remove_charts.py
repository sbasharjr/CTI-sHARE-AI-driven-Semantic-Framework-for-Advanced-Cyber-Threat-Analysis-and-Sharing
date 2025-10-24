#!/usr/bin/env python3
"""
Chart Removal Script for CTI-sHARE Dashboard
Removes Attack Vectors, Geographic Distribution, and Hourly Activity Pattern charts
"""

import re
from pathlib import Path

def remove_charts_from_dashboard():
    """Remove the three specified charts from dashboard.html"""
    
    dashboard_path = Path("src/dashboard/templates/dashboard.html")
    if not dashboard_path.exists():
        print("❌ Dashboard file not found!")
        return False
    
    print("🔧 Starting chart removal process...")
    
    with open(dashboard_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_length = len(content)
    print(f"📊 Original file size: {original_length} characters")
    
    # Remove HTML sections for the three charts
    print("\n1️⃣ Removing HTML sections...")
    
    # Remove Attack Vectors section
    attack_vectors_pattern = r'''<div class="grid-2">
                <div>
                    <div class="chart-header">
                        <h3>🎯 Attack Vectors \(Live\)</h3>.*?</canvas>
                    </div>
                </div>
                <div>
                    <div class="chart-header">
                        <h3>🌍 Geographic Distribution \(Live\)</h3>.*?</canvas>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 20px;">
                <div class="chart-header">
                    <h3>⏰ Hourly Activity Pattern \(Live\)</h3>.*?</canvas>
                </div>
            </div>'''
    
    content = re.sub(attack_vectors_pattern, '', content, flags=re.DOTALL)
    
    # Remove CSS styles
    print("2️⃣ Removing CSS styles...")
    css_pattern = r'/\* Attack Vectors Information Panel Styles \*/.*?/\* Responsive adjustments for attack vectors info \*/.*?}'
    content = re.sub(css_pattern, '', content, flags=re.DOTALL)
    
    # Remove JavaScript chart initializations
    print("3️⃣ Removing JavaScript chart initialization...")
    js_init_pattern = r'// Attack Vectors Chart \(Bar\).*?// Hourly Activity Chart \(Area\).*?}\);'
    content = re.sub(js_init_pattern, '', content, flags=re.DOTALL)
    
    # Update loadInitialData function
    print("4️⃣ Updating loadInitialData function...")
    load_initial_pattern = r'setTimeout\(\(\) => \{.*?updateAttackVectorsChart\(\);.*?updateGeographicChart\(\);.*?updateHourlyActivityChart\(\);.*?\}, 1000\);'
    content = re.sub(load_initial_pattern, '', content, flags=re.DOTALL)
    
    # Remove function definitions
    print("5️⃣ Removing function definitions...")
    functions_pattern = r'// Enhanced Active Chart Functions.*?// Enhanced chart interaction functions'
    content = re.sub(functions_pattern, '// Enhanced chart interaction functions', content, flags=re.DOTALL)
    
    # Update resetAllCharts function
    print("6️⃣ Updating resetAllCharts function...")
    reset_charts_pattern = r'const charts = \[categoriesChart, severityChart, severityTimeSeriesChart, attackVectorsChart, geoDistributionChart, hourlyActivityChart\];'
    content = re.sub(reset_charts_pattern, 'const charts = [categoriesChart, severityChart, severityTimeSeriesChart];', content)
    
    # Update pauseAllLiveModes function
    print("7️⃣ Updating pauseAllLiveModes function...")
    pause_intervals_pattern = r"const intervals = \[[\s\S]*?'attackVectorsInterval',[\s\S]*?'geographicInterval',[\s\S]*?'hourlyActivityInterval',[\s\S]*?\];"
    content = re.sub(pause_intervals_pattern, "const intervals = [\n                'severityRealtimeInterval',\n                'advancedRealtimeInterval'\n            ];", content)
    
    pause_buttons_pattern = r"const buttons = \[[\s\S]*?'attackVectorsLiveBtn',[\s\S]*?'geographicLiveBtn',[\s\S]*?'hourlyActivityLiveBtn',[\s\S]*?\];"
    content = re.sub(pause_buttons_pattern, "const buttons = [\n                'severityRealtimeBtn',\n                'advancedRealtimeBtn'\n            ];", content)
    
    # Update startAllLiveModes function
    print("8️⃣ Updating startAllLiveModes function...")
    start_all_pattern = r"// Start all chart live modes[\s\S]*?if \(document\.getElementById\('attackVectorsLiveBtn'\)\) startAttackVectorsLiveMode\(\);[\s\S]*?if \(document\.getElementById\('geographicLiveBtn'\)\) startGeographicLiveMode\(\);[\s\S]*?if \(document\.getElementById\('hourlyActivityLiveBtn'\)\) startHourlyActivityLiveMode\(\);[\s\S]*?toggleSeverityRealtime\(\);[\s\S]*?toggleAdvancedRealtime\(\);"
    content = re.sub(start_all_pattern, "// Start all chart live modes\n            toggleSeverityRealtime();\n            toggleAdvancedRealtime();", content)
    
    # Remove variable declarations
    print("9️⃣ Updating variable declarations...")
    var_pattern = r'let categoriesChart, severityChart, severityTimeSeriesChart, attackVectorsChart, geoDistributionChart, hourlyActivityChart;'
    content = re.sub(var_pattern, 'let categoriesChart, severityChart, severityTimeSeriesChart;', content)
    
    new_length = len(content)
    print(f"📊 New file size: {new_length} characters")
    print(f"📉 Removed: {original_length - new_length} characters")
    
    # Write the updated content
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Chart removal completed successfully!")
    return True

if __name__ == "__main__":
    print("🛠️  CTI-sHARE Chart Removal Tool")
    print("=" * 50)
    
    success = remove_charts_from_dashboard()
    
    if success:
        print("\n🎉 All charts removed successfully!")
        print("📋 Removed:")
        print("   • Attack Vectors (Live) chart")
        print("   • Geographic Distribution (Live) chart") 
        print("   • Hourly Activity Pattern (Live) chart")
        print("\n📝 Please test the dashboard to ensure it works correctly.")
    else:
        print("\n❌ Chart removal failed!")