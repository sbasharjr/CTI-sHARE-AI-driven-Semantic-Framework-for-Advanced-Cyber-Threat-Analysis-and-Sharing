#!/usr/bin/env python3
"""
CTI-sHARE Portable Package Creator
==================================

Creates a portable version of CTI-sHARE that can run on any Windows machine
with Python installed, or can be packaged with a Python interpreter.
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path
from datetime import datetime

def create_portable_package():
    """Create a portable package of CTI-sHARE"""
    
    print("=" * 80)
    print("🛡️  CTI-sHARE Dashboard - Portable Package Creator")
    print("=" * 80)
    
    project_root = Path(__file__).resolve().parent
    package_name = f"CTI-sHARE-Portable-{datetime.now().strftime('%Y%m%d')}"
    package_dir = project_root / package_name
    
    # Clean and create package directory
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir()
    
    print(f"📦 Creating package: {package_name}")
    
    # Essential files to include
    files_to_copy = [
        'run_production_lite.py',
        'wsgi.py',
        'misp_integration.py',
        'requirements.txt',
        'README.md',
        'PRODUCTION_DEPLOYMENT_GUIDE.md',
        'MISP_INTEGRATION_GUIDE.md',
        'live_threat_data.json',
    ]
    
    # Directories to copy
    dirs_to_copy = [
        'src',
        'config',
        'data',
    ]
    
    # Copy files
    print("📁 Copying essential files...")
    for file_name in files_to_copy:
        file_path = project_root / file_name
        if file_path.exists():
            shutil.copy2(file_path, package_dir)
            print(f"   ✅ {file_name}")
        else:
            print(f"   ⚠️  {file_name} (not found)")
    
    # Copy directories
    print("📂 Copying directories...")
    for dir_name in dirs_to_copy:
        dir_path = project_root / dir_name
        if dir_path.exists():
            shutil.copytree(dir_path, package_dir / dir_name)
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ⚠️  {dir_name}/ (not found)")
    
    # Create launcher scripts
    print("🚀 Creating launcher scripts...")
    
    # Windows batch launcher
    batch_launcher = f'''@echo off
title CTI-sHARE Dashboard - Portable
echo.
echo ======================================================
echo 🛡️  CTI-sHARE Dashboard - Portable Version
echo ======================================================
echo.
echo 🚀 Starting CTI-sHARE Dashboard...
echo 📡 Server will be available at: http://localhost:5001
echo 🔧 Version: Portable (Lite)
echo.
echo Prerequisites:
echo - Python 3.8+ installed
echo - Dependencies: pip install flask flask-cors waitress
echo.
echo Press Ctrl+C to stop the server
echo ======================================================
echo.

python run_production_lite.py

echo.
echo ======================================================
echo Server stopped. Press any key to exit...
pause > nul
'''
    
    with open(package_dir / 'Start-CTI-sHARE.bat', 'w') as f:
        f.write(batch_launcher)
    
    # PowerShell launcher
    ps_launcher = '''# CTI-sHARE Dashboard - PowerShell Launcher
Write-Host "=" -Repeat 80 -ForegroundColor Cyan
Write-Host "🛡️  CTI-sHARE Dashboard - Portable Version" -ForegroundColor Yellow
Write-Host "=" -Repeat 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "🚀 Starting CTI-sHARE Dashboard..." -ForegroundColor Green
Write-Host "📡 Server will be available at: http://localhost:5001" -ForegroundColor White
Write-Host "🔧 Version: Portable (Lite)" -ForegroundColor White
Write-Host ""
Write-Host "Prerequisites:" -ForegroundColor Yellow
Write-Host "- Python 3.8+ installed" -ForegroundColor White
Write-Host "- Dependencies: pip install flask flask-cors waitress" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Red
Write-Host "=" -Repeat 80 -ForegroundColor Cyan
Write-Host ""

try {
    python run_production_lite.py
} catch {
    Write-Host "❌ Error starting server: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Make sure Python is installed and in PATH" -ForegroundColor Yellow
    Write-Host "💡 Install dependencies with: pip install flask flask-cors waitress" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=" -Repeat 80 -ForegroundColor Cyan
Write-Host "Server stopped. Press any key to exit..." -ForegroundColor Yellow
Read-Host
'''
    
    with open(package_dir / 'Start-CTI-sHARE.ps1', 'w') as f:
        f.write(ps_launcher)
    
    # Installation script
    install_script = '''@echo off
title CTI-sHARE Dependencies Installer
echo.
echo ======================================================
echo 🛡️  CTI-sHARE Dashboard - Dependencies Installer
echo ======================================================
echo.
echo This script will install the required Python packages
echo for CTI-sHARE Dashboard to run properly.
echo.
echo Required packages:
echo - Flask (Web framework)
echo - Flask-CORS (Cross-origin resource sharing)
echo - Waitress (Production WSGI server)
echo - Requests (HTTP library)
echo.
echo ======================================================
echo.

set /p continue="Press Enter to continue or Ctrl+C to cancel..."

echo.
echo 🔄 Installing dependencies...
echo.

pip install flask flask-cors waitress requests

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Dependencies installed successfully!
    echo.
    echo You can now run CTI-sHARE by executing:
    echo - Start-CTI-sHARE.bat
    echo - Or: python run_production_lite.py
) else (
    echo.
    echo ❌ Installation failed!
    echo.
    echo Please check:
    echo - Python is installed and in PATH
    echo - Internet connection is available
    echo - Run as Administrator if needed
)

echo.
echo ======================================================
pause
'''
    
    with open(package_dir / 'Install-Dependencies.bat', 'w') as f:
        f.write(install_script)
    
    # Create README for portable version
    portable_readme = f'''# CTI-sHARE Dashboard - Portable Version

## Overview
This is a portable version of the CTI-sHARE Dashboard that can run on any Windows machine with Python installed.

## Quick Start

### Step 1: Install Dependencies
1. Run `Install-Dependencies.bat` as Administrator
2. This will install required Python packages

### Step 2: Start the Dashboard
1. Double-click `Start-CTI-sHARE.bat`
2. Wait for the server to start
3. Open your browser to: http://localhost:5001

## Alternative Methods

### PowerShell
1. Right-click `Start-CTI-sHARE.ps1`
2. Select "Run with PowerShell"

### Command Line
```bash
python run_production_lite.py
```

## Features Included
- ✅ Core Dashboard Interface
- ✅ Threat Intelligence Management
- ✅ MISP Integration (Simulated)
- ✅ Real-time Monitoring Interface
- ✅ Security Operations Center
- ✅ Professional UI with Charts

## System Requirements
- Windows 10/11
- Python 3.8 or higher
- 2GB RAM minimum
- 500MB disk space
- Internet connection (for MISP integration)

## Dependencies
The following Python packages are required:
- flask
- flask-cors
- waitress
- requests

## Configuration
- Server runs on port 5001 by default
- Configuration files are in the `config/` directory
- Sample data is loaded automatically

## Troubleshooting

### Python Not Found
- Install Python from https://python.org
- Make sure "Add to PATH" is checked during installation

### Permission Errors
- Run `Install-Dependencies.bat` as Administrator
- Check Windows Firewall settings

### Port Already in Use
- Change port in `run_production_lite.py` (line with `port=5001`)
- Or stop other services using port 5001

## Lite Version Features
This portable version includes:
- Core dashboard functionality
- Simulated AI analysis (keyword-based)
- MISP integration interface (simulated responses)
- Sample threat data for demonstration
- All UI components and charts

Note: Advanced ML features require the full installation with all dependencies.

## Security Notes
- Run only on trusted networks
- Configure firewall appropriately
- Use strong authentication for production

## Support
- GitHub: https://github.com/sbasharjr/CTI-sHARE-AI-driven-Semantic-Framework-for-Advanced-Cyber-Threat-Analysis-and-Sharing
- Documentation: See included markdown files

---
Build Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Version: Portable v1.0.0
'''
    
    with open(package_dir / 'README-PORTABLE.txt', 'w') as f:
        f.write(portable_readme)
    
    # Create ZIP archive
    print("🗜️  Creating ZIP archive...")
    zip_file = project_root / f"{package_name}.zip"
    
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = Path(root) / file
                arc_name = file_path.relative_to(package_dir)
                zipf.write(file_path, arc_name)
    
    # Calculate sizes
    dir_size = sum(f.stat().st_size for f in package_dir.rglob('*') if f.is_file())
    zip_size = zip_file.stat().st_size
    
    print("\n" + "=" * 80)
    print("🎉 PORTABLE PACKAGE CREATED SUCCESSFULLY!")
    print("=" * 80)
    print(f"📁 Package Directory: {package_dir}")
    print(f"📦 ZIP Archive: {zip_file}")
    print(f"📏 Directory Size: {dir_size / (1024*1024):.1f} MB")
    print(f"📏 ZIP Size: {zip_size / (1024*1024):.1f} MB")
    print("\n🚀 To use the portable version:")
    print("   1. Extract the ZIP file on target machine")
    print("   2. Run 'Install-Dependencies.bat' as Administrator")
    print("   3. Run 'Start-CTI-sHARE.bat' to start the dashboard")
    print("   4. Open browser to: http://localhost:5001")
    print("=" * 80)
    
    return package_dir, zip_file

if __name__ == "__main__":
    create_portable_package()