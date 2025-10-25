@echo off
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
    echo - Start-CTI-sHARE-Portable.bat
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