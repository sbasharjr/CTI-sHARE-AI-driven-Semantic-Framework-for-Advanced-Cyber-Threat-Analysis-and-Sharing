#!/bin/bash

# Production deployment script for CTI-sHARE Dashboard on Linux/Unix

echo "================================================================================"
echo "CTI-sHARE Dashboard - Production Deployment (Linux/Unix)"
echo "================================================================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install/update dependencies
echo "Installing production dependencies..."
pip install -r requirements.txt

# Start production server using Gunicorn
echo "Starting production server with Gunicorn..."
echo "Server will be available at: http://localhost:5001"
echo "Press Ctrl+C to stop the server"
echo "================================================================================"

# Option 1: Run with Gunicorn and custom config
gunicorn --config gunicorn_config.py wsgi:application

# Option 2: Simple Gunicorn command (uncomment to use instead)
# gunicorn --bind 0.0.0.0:5001 --workers 4 --timeout 30 wsgi:application

# Option 3: Use Waitress if Gunicorn is not available (uncomment to use)
# python run_production_server.py