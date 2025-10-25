@echo off
title CTI-sHARE Executable Builder
echo.
echo ======================================================
echo 🛡️  CTI-sHARE Dashboard - Executable Builder
echo ======================================================
echo.
echo This script will create a standalone executable for
echo the CTI-sHARE Dashboard that can run without Python.
echo.
echo Prerequisites:
echo - Python with all dependencies installed
echo - PyInstaller (pip install pyinstaller)
echo.
echo ======================================================
echo.

set /p continue="Press Enter to continue or Ctrl+C to cancel..."

echo.
echo 🚀 Starting build process...
echo.

python build_executable.py

echo.
echo ======================================================
echo Build process completed!
echo.
echo If successful, you'll find:
echo - CTI-sHARE-Production.exe in the 'dist' folder
echo - Start-CTI-sHARE-Dashboard.bat launcher
echo - Complete distribution package
echo.
echo ======================================================
echo.
pause