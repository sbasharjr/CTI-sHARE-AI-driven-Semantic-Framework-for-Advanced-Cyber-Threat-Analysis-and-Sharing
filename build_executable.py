#!/usr/bin/env python3
"""
CTI-sHARE Production Executable Builder
======================================

This script creates a standalone executable for the CTI-sHARE Dashboard
using PyInstaller. The executable includes all dependencies and can run
without requiring Python installation.

Requirements:
- PyInstaller: pip install pyinstaller
- All CTI-sHARE dependencies from requirements.txt

Usage:
    python build_executable.py

Output:
    - Single executable file: dist/CTI-sHARE-Production.exe
    - Or directory distribution: dist/CTI-sHARE-Production/
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
from datetime import datetime

class CTIShareExecutableBuilder:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent
        self.build_dir = self.project_root / 'build'
        self.dist_dir = self.project_root / 'dist'
        self.spec_file = self.project_root / 'CTI-sHARE-Production.spec'
        
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        print("🔍 Checking dependencies...")
        
        try:
            import PyInstaller
            print("✅ PyInstaller found")
        except ImportError:
            print("❌ PyInstaller not found. Install with: pip install pyinstaller")
            return False
        
        # Check requirements.txt dependencies
        requirements_file = self.project_root / 'requirements.txt'
        if requirements_file.exists():
            with open(requirements_file, 'r') as f:
                requirements = f.read().splitlines()
            
            missing_deps = []
            for req in requirements:
                if req.strip() and not req.startswith('#'):
                    package_name = req.split('>=')[0].split('==')[0].split('<')[0].strip()
                    try:
                        __import__(package_name.replace('-', '_'))
                        print(f"✅ {package_name}")
                    except ImportError:
                        missing_deps.append(package_name)
                        print(f"❌ {package_name} - Missing")
            
            if missing_deps:
                print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
                print("💡 Install with: pip install -r requirements.txt")
                return False
        
        return True
    
    def prepare_build_environment(self):
        """Prepare the build environment"""
        print("🛠️  Preparing build environment...")
        
        # Clean previous builds
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        if self.dist_dir.exists():
            shutil.rmtree(self.dist_dir)
        
        # Create directories
        self.build_dir.mkdir(exist_ok=True)
        self.dist_dir.mkdir(exist_ok=True)
        
        print("✅ Build environment prepared")
    
    def create_build_info(self):
        """Create build information file"""
        build_info = {
            "project": "CTI-sHARE Dashboard",
            "version": "v1.0.0",
            "build_date": datetime.now().isoformat(),
            "python_version": sys.version,
            "platform": sys.platform,
            "architecture": "x64" if sys.maxsize > 2**32 else "x86"
        }
        
        build_info_file = self.project_root / 'build_info.json'
        with open(build_info_file, 'w') as f:
            json.dump(build_info, f, indent=2)
        
        return build_info
    
    def run_pyinstaller(self):
        """Run PyInstaller to create the executable"""
        print("🚀 Building executable with PyInstaller...")
        
        try:
            # Run PyInstaller with the spec file
            cmd = [
                sys.executable, '-m', 'PyInstaller',
                '--clean',  # Clean cache and temporary files
                '--noconfirm',  # Replace output directory without asking
                str(self.spec_file)
            ]
            
            print(f"📝 Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✅ PyInstaller completed successfully")
                return True
            else:
                print("❌ PyInstaller failed")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error running PyInstaller: {e}")
            return False
    
    def create_launcher_script(self):
        """Create a launcher script for the executable"""
        launcher_content = f'''@echo off
title CTI-sHARE Dashboard - Production Server
echo.
echo ======================================================
echo 🛡️  CTI-sHARE Dashboard - Production Server
echo ======================================================
echo.
echo 🚀 Starting CTI-sHARE Dashboard...
echo 📡 Server will be available at: http://localhost:5001
echo 🔧 WSGI Server: Waitress
echo ⚡ Environment: Production
echo.
echo Press Ctrl+C to stop the server
echo ======================================================
echo.

"%~dp0CTI-sHARE-Production.exe"

echo.
echo ======================================================
echo Server stopped. Press any key to exit...
pause > nul
'''
        
        launcher_file = self.dist_dir / 'Start-CTI-sHARE-Dashboard.bat'
        with open(launcher_file, 'w') as f:
            f.write(launcher_content)
        
        print(f"✅ Launcher script created: {launcher_file}")
    
    def create_readme(self):
        """Create README for the executable distribution"""
        readme_content = '''# CTI-sHARE Dashboard - Standalone Executable

## Overview
This is a standalone executable version of the CTI-sHARE Dashboard that includes all dependencies and can run without requiring Python installation.

## Quick Start

### Option 1: Using the Launcher (Recommended)
1. Double-click `Start-CTI-sHARE-Dashboard.bat`
2. Wait for the server to start
3. Open your web browser and go to: http://localhost:5001

### Option 2: Direct Execution
1. Double-click `CTI-sHARE-Production.exe`
2. Open your web browser and go to: http://localhost:5001

## Features Included
- ✅ AI-Powered Threat Analysis
- ✅ MISP Framework Integration
- ✅ Real-time Monitoring
- ✅ Advanced Analytics Dashboard
- ✅ Smart Alert System
- ✅ Collaborative Threat Sharing

## System Requirements
- Windows 10/11 (64-bit)
- Minimum 4GB RAM
- 2GB available disk space
- Network connectivity (for MISP integration)

## Configuration
- Default server port: 5001
- Configuration files are included in the executable
- MISP integration can be configured through the web interface

## Troubleshooting

### Server Won't Start
- Check if port 5001 is available
- Run as Administrator if needed
- Check Windows Firewall settings

### Performance Issues
- Ensure minimum 4GB RAM available
- Close unnecessary applications
- Check antivirus software (may need to whitelist)

### MISP Integration
- Verify MISP server connectivity
- Check API key configuration
- Ensure proper network access

## Security Notes
- The executable handles sensitive threat intelligence data
- All communications are encrypted
- Access is logged for security purposes
- Report security vulnerabilities responsibly

## Support
- GitHub Repository: https://github.com/sbasharjr/CTI-sHARE-AI-driven-Semantic-Framework-for-Advanced-Cyber-Threat-Analysis-and-Sharing
- Documentation: Included in the executable package

## Build Information
- Build Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Version: v1.0.0
- Platform: Windows x64
- Python Version: {sys.version}

---
© 2024 CTI-sHARE Project | Bashar Intelligence Team
'''
        
        readme_file = self.dist_dir / 'README-EXECUTABLE.txt'
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        print(f"✅ README created: {readme_file}")
    
    def verify_executable(self):
        """Verify the built executable"""
        exe_path = self.dist_dir / 'CTI-sHARE-Production.exe'
        
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"✅ Executable created successfully!")
            print(f"📁 Location: {exe_path}")
            print(f"📏 Size: {size_mb:.1f} MB")
            return True
        else:
            print("❌ Executable not found!")
            return False
    
    def create_distribution_package(self):
        """Create a complete distribution package"""
        print("📦 Creating distribution package...")
        
        # Create package directory
        package_name = f"CTI-sHARE-Dashboard-Standalone-{datetime.now().strftime('%Y%m%d')}"
        package_dir = self.dist_dir / package_name
        package_dir.mkdir(exist_ok=True)
        
        # Copy executable and related files
        exe_path = self.dist_dir / 'CTI-sHARE-Production.exe'
        if exe_path.exists():
            shutil.copy2(exe_path, package_dir)
        
        launcher_path = self.dist_dir / 'Start-CTI-sHARE-Dashboard.bat'
        if launcher_path.exists():
            shutil.copy2(launcher_path, package_dir)
        
        readme_path = self.dist_dir / 'README-EXECUTABLE.txt'
        if readme_path.exists():
            shutil.copy2(readme_path, package_dir)
        
        # Copy essential documentation
        docs_to_copy = [
            'README.md',
            'PRODUCTION_DEPLOYMENT_GUIDE.md',
            'MISP_INTEGRATION_GUIDE.md'
        ]
        
        for doc in docs_to_copy:
            doc_path = self.project_root / doc
            if doc_path.exists():
                shutil.copy2(doc_path, package_dir)
        
        print(f"✅ Distribution package created: {package_dir}")
        return package_dir
    
    def build(self):
        """Main build process"""
        print("=" * 80)
        print("🛡️  CTI-sHARE Dashboard - Executable Builder")
        print("=" * 80)
        
        # Check dependencies
        if not self.check_dependencies():
            return False
        
        # Prepare environment
        self.prepare_build_environment()
        
        # Create build info
        build_info = self.create_build_info()
        print(f"📋 Build Info: {build_info['project']} {build_info['version']}")
        
        # Run PyInstaller
        if not self.run_pyinstaller():
            return False
        
        # Verify executable
        if not self.verify_executable():
            return False
        
        # Create additional files
        self.create_launcher_script()
        self.create_readme()
        
        # Create distribution package
        package_dir = self.create_distribution_package()
        
        print("\n" + "=" * 80)
        print("🎉 BUILD COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📁 Executable: {self.dist_dir / 'CTI-sHARE-Production.exe'}")
        print(f"📦 Package: {package_dir}")
        print(f"🚀 Launcher: {self.dist_dir / 'Start-CTI-sHARE-Dashboard.bat'}")
        print("\n💡 To test the executable:")
        print("   1. Double-click 'Start-CTI-sHARE-Dashboard.bat'")
        print("   2. Open browser to: http://localhost:5001")
        print("=" * 80)
        
        return True

def main():
    """Main entry point"""
    builder = CTIShareExecutableBuilder()
    success = builder.build()
    
    if not success:
        print("\n❌ Build failed!")
        sys.exit(1)
    else:
        print("\n✅ Build completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()