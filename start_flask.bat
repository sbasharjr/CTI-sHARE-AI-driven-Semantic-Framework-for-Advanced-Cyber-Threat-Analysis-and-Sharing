@echo off
echo Starting CTI-sHARE Flask Application...
echo.

REM Try different Python executables
if exist ".venv\Scripts\python.exe" (
    echo Using virtual environment Python...
    .venv\Scripts\python.exe flask_app.py
) else if exist "C:\Python\python.exe" (
    echo Using system Python...
    C:\Python\python.exe flask_app.py
) else (
    echo Using default Python...
    python flask_app.py
)

pause