@echo off
REM Production deployment script for CTI-sHARE Dashboard on Windows

echo ================================================================================
echo CTI-sHARE Dashboard - Production Deployment (Windows)
echo ================================================================================

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install/update dependencies
echo Installing production dependencies...
pip install -r requirements.txt

REM Start production server using Waitress (Windows-compatible)
echo Starting production server...
echo Server will be available at: http://localhost:5001
echo Press Ctrl+C to stop the server
echo ================================================================================

python run_production_server.py

pause