#!/bin/bash
# Quick start script for Docker deployment

set -e

echo "=================================================="
echo "Threat Analysis Framework - Docker Quick Start"
echo "=================================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is available
if ! docker compose version &> /dev/null; then
    echo "Warning: Docker Compose plugin not found. Using docker-compose command."
    COMPOSE_CMD="docker-compose"
else
    COMPOSE_CMD="docker compose"
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data/raw data/processed models logs
echo "✓ Directories created"
echo ""

# Parse command line arguments
ACTION="${1:-up}"

case $ACTION in
    up|start)
        echo "Starting Threat Analysis Framework..."
        $COMPOSE_CMD up -d
        echo ""
        echo "✓ Framework started successfully!"
        echo ""
        echo "Access the API at: http://localhost:5000/api/health"
        echo ""
        echo "View logs with: $COMPOSE_CMD logs -f"
        echo "Stop with: ./docker-start.sh stop"
        ;;
    
    down|stop)
        echo "Stopping Threat Analysis Framework..."
        $COMPOSE_CMD down
        echo "✓ Framework stopped"
        ;;
    
    restart)
        echo "Restarting Threat Analysis Framework..."
        $COMPOSE_CMD restart
        echo "✓ Framework restarted"
        ;;
    
    logs)
        echo "Showing logs (Ctrl+C to exit)..."
        $COMPOSE_CMD logs -f
        ;;
    
    build)
        echo "Building Docker image..."
        $COMPOSE_CMD build
        echo "✓ Build complete"
        ;;
    
    status)
        echo "Container status:"
        $COMPOSE_CMD ps
        ;;
    
    test)
        echo "Testing API endpoints..."
        echo ""
        
        # Wait for the service to be ready
        echo "Waiting for API to be ready..."
        sleep 5
        
        # Health check
        echo "1. Testing health endpoint..."
        if curl -f http://localhost:5000/api/health 2>/dev/null; then
            echo "   ✓ Health check passed"
        else
            echo "   ✗ Health check failed"
        fi
        echo ""
        
        # Statistics
        echo "2. Testing statistics endpoint..."
        if curl -f http://localhost:5000/api/statistics 2>/dev/null; then
            echo "   ✓ Statistics endpoint working"
        else
            echo "   ✗ Statistics endpoint failed"
        fi
        echo ""
        ;;
    
    clean)
        echo "Cleaning up Docker resources..."
        $COMPOSE_CMD down -v
        docker image prune -f
        echo "✓ Cleanup complete"
        ;;
    
    help|*)
        echo "Usage: ./docker-start.sh [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  up, start    - Start the framework (default)"
        echo "  down, stop   - Stop the framework"
        echo "  restart      - Restart the framework"
        echo "  logs         - Show logs"
        echo "  build        - Build Docker image"
        echo "  status       - Show container status"
        echo "  test         - Test API endpoints"
        echo "  clean        - Clean up Docker resources"
        echo "  help         - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./docker-start.sh up       # Start the framework"
        echo "  ./docker-start.sh logs     # View logs"
        echo "  ./docker-start.sh stop     # Stop the framework"
        ;;
esac
