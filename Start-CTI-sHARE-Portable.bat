@echo off
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