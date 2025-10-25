# CTI-sHARE Production Executable Creation Guide

## Overview
This guide provides multiple methods to create executable versions of the CTI-sHARE Dashboard for production deployment.

## 🎯 Available Options

### 1. Portable Version (Recommended)
Creates a lightweight, portable version that requires Python on the target machine.

**Files Created:**
- `run_production_lite.py` - Simplified production server
- `Start-CTI-sHARE-Portable.bat` - Windows launcher
- `Install-Dependencies-Lite.bat` - Dependency installer

**Usage:**
1. Run `Install-Dependencies-Lite.bat` as Administrator
2. Run `Start-CTI-sHARE-Portable.bat` to start the dashboard
3. Open browser to http://localhost:5001

### 2. PyInstaller Executable
Creates a standalone .exe file with all dependencies bundled.

**Requirements:**
```bash
pip install pyinstaller
```

**Build Commands:**

#### Full Version (Large but complete):
```bash
pyinstaller --onefile --name="CTI-sHARE-Production" ^
  --add-data="src;src" ^
  --add-data="config;config" ^
  --add-data="data;data" ^
  --add-data="live_threat_data.json;." ^
  --hidden-import=waitress ^
  --hidden-import=flask ^
  --hidden-import=flask_cors ^
  run_production_server.py
```

#### Lite Version (Smaller and faster):
```bash
pyinstaller --onefile --name="CTI-sHARE-Production-Lite" ^
  --add-data="src/dashboard/templates/dashboard.html;src/dashboard/templates" ^
  --add-data="config/config.yaml;config" ^
  --add-data="live_threat_data.json;." ^
  --hidden-import=waitress ^
  --hidden-import=flask ^
  --hidden-import=flask_cors ^
  run_production_lite.py
```

#### Using Spec File:
```bash
pyinstaller CTI-sHARE-Production.spec
```

### 3. Docker Container (Cross-platform)
Creates a containerized version that runs anywhere Docker is available.

**Build:**
```bash
docker build -f Dockerfile.production -t cti-share-production .
```

**Run:**
```bash
docker run -p 5001:5001 cti-share-production
```

### 4. Python Wheel Package
Creates a distributable Python package.

**Build:**
```bash
python setup.py bdist_wheel
```

**Install:**
```bash
pip install dist/cti_share-1.0.0-py3-none-any.whl
```

## 🛠️ Build Scripts Created

### Build-Executable.bat
Automated build script for PyInstaller executable:
```batch
@echo off
title CTI-sHARE Executable Builder
python build_executable.py
pause
```

### build_executable.py
Comprehensive Python script that:
- Checks dependencies
- Prepares build environment
- Runs PyInstaller
- Creates launcher scripts
- Packages distribution

### create_portable.py
Creates portable package with:
- All source files
- Launcher scripts
- Installation guides
- ZIP archive for distribution

## 📋 Production Files Overview

### Core Application Files
- `run_production_server.py` - Full production server
- `run_production_lite.py` - Lightweight server (no heavy ML)
- `wsgi.py` - WSGI application entry point
- `misp_integration.py` - MISP framework integration

### Build Configuration
- `CTI-sHARE-Production.spec` - Full PyInstaller spec
- `CTI-sHARE-Lite.spec` - Lightweight PyInstaller spec
- `requirements.txt` - Python dependencies

### Launcher Scripts
- `Start-CTI-sHARE-Portable.bat` - Portable version launcher
- `Install-Dependencies-Lite.bat` - Dependency installer
- `Build-Executable.bat` - Build automation script

## 🎯 Recommended Approach

### For Development/Testing:
Use the **Portable Version**:
1. Run `Install-Dependencies-Lite.bat`
2. Run `Start-CTI-sHARE-Portable.bat`

### For Production Deployment:
Use **Docker Container**:
1. Build: `docker build -f Dockerfile.production -t cti-share .`
2. Run: `docker run -p 5001:5001 cti-share`

### For Standalone Distribution:
Use **PyInstaller Lite Version**:
1. Run: `pyinstaller CTI-sHARE-Lite.spec`
2. Distribute: `dist/CTI-sHARE-Production-Lite.exe`

## 🔧 Features by Version

### Full Version
- ✅ Complete ML/AI capabilities
- ✅ All dependencies included
- ✅ Advanced threat analysis
- ✅ Complete MISP integration
- ❌ Large file size (~500MB+)
- ❌ Longer build time

### Lite Version
- ✅ Core dashboard functionality
- ✅ MISP integration interface
- ✅ Real-time monitoring UI
- ✅ Smaller file size (~50MB)
- ✅ Faster build time
- ❌ Simulated ML analysis
- ❌ Limited advanced features

### Portable Version
- ✅ Smallest footprint
- ✅ Easy to modify/debug
- ✅ Quick setup
- ✅ All UI features
- ❌ Requires Python on target
- ❌ Manual dependency installation

## 🚀 Quick Start Commands

### Create Portable Version:
```bash
# Manual setup
copy run_production_lite.py to target machine
copy src/ folder to target machine
run Install-Dependencies-Lite.bat
run Start-CTI-sHARE-Portable.bat
```

### Create Executable:
```bash
# Install PyInstaller
pip install pyinstaller

# Build lite executable
pyinstaller CTI-sHARE-Lite.spec

# Result: dist/CTI-sHARE-Production-Lite.exe
```

### Create Docker Image:
```bash
# Build production image
docker build -f Dockerfile.production -t cti-share .

# Run container
docker run -p 5001:5001 cti-share

# Access: http://localhost:5001
```

## 🔍 Troubleshooting

### PyInstaller Issues:
- **Large file size**: Use lite version or exclude unnecessary modules
- **Missing dependencies**: Add to hidden_imports in spec file
- **Runtime errors**: Check for missing data files in add-data

### Portable Version Issues:
- **Python not found**: Install Python and add to PATH
- **Dependencies fail**: Run as Administrator
- **Port conflicts**: Change port in run_production_lite.py

### Docker Issues:
- **Build fails**: Check Dockerfile syntax and dependencies
- **Container won't start**: Check port conflicts and permissions
- **Performance issues**: Allocate more memory to Docker

## 📊 Comparison Table

| Method | Size | Dependencies | Portability | Setup Time |
|--------|------|--------------|-------------|------------|
| Portable | ~5MB | Python Required | High | 2 minutes |
| Lite Executable | ~50MB | None | High | 10 minutes |
| Full Executable | ~500MB | None | High | 30 minutes |
| Docker | ~200MB | Docker Required | Very High | 5 minutes |

## 🎯 Recommendations

- **Quick Testing**: Use Portable Version
- **Production Deployment**: Use Docker
- **Standalone Distribution**: Use Lite Executable
- **Full Features**: Use Full Executable (if size isn't a concern)

---

## 📞 Support

For build issues or questions:
- Check the specific error messages
- Verify all dependencies are installed
- Ensure sufficient disk space (1GB+ for builds)
- Run build processes as Administrator if needed

**Build Date**: October 25, 2024
**Version**: v1.0.0