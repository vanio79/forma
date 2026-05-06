#!/bin/bash
# Forma Server Management CLI
# Usage: ./server.sh [start|stop|status|logs|restart]

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$PROJECT_DIR/.server.pid"
LOG_FILE="$PROJECT_DIR/server.log"
PORT=8000

start_server() {
    if is_running; then
        echo "Server is already running (PID: $(cat $PID_FILE))"
        return 0
    fi
    
    echo "Starting Forma server..."
    cd "$PROJECT_DIR"
    nohup uv run python -m forma.main > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    echo $SERVER_PID > "$PID_FILE"
    
    # Wait for server to be ready
    echo "Waiting for server to start..."
    for i in {1..10}; do
        sleep 1
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "Server started successfully (PID: $SERVER_PID)"
            echo "Access the Web UI at: http://localhost:$PORT"
            return 0
        fi
    done
    
    echo "Server failed to start. Check logs: $LOG_FILE"
    return 1
}

stop_server() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        echo "Stopping Forma server (PID: $PID)..."
        
        # Kill the process and its children
        kill -TERM "$PID" 2>/dev/null || true
        
        # Wait for graceful shutdown
        sleep 2
        
        # Force kill if still running
        if kill -0 "$PID" 2>/dev/null; then
            echo "Force killing server..."
            kill -9 "$PID" 2>/dev/null || true
        fi
        
        rm -f "$PID_FILE"
    else
        echo "No PID file found, checking port $PORT..."
    fi
    
    # Also kill any process on port 8000
    fuser -k $PORT/tcp 2>/dev/null || true
    
    echo "Server stopped"
}

is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            return 0
        fi
    fi
    
    # Check if something is running on port 8000
    if lsof -i :$PORT > /dev/null 2>&1; then
        return 0
    fi
    
    return 1
}

check_status() {
    if is_running; then
        PID=$(cat "$PID_FILE" 2>/dev/null || echo "unknown")
        echo "Server is running (PID: $PID)"
        
        # Check health endpoint
        HEALTH=$(curl -s http://localhost:$PORT/health 2>/dev/null)
        if [ -n "$HEALTH" ]; then
            echo "Health check: $HEALTH"
            echo "Web UI: http://localhost:$PORT"
        else
            echo "Warning: Server process exists but health check failed"
        fi
        return 0
    else
        echo "Server is not running"
        return 1
    fi
}

view_logs() {
    if [ -f "$LOG_FILE" ]; then
        echo "=== Last 50 lines of server.log ==="
        tail -50 "$LOG_FILE"
        echo ""
        echo "Full log: $LOG_FILE"
    else
        echo "No log file found"
    fi
}

restart_server() {
    echo "Restarting Forma server..."
    stop_server
    sleep 2
    start_server
}

case "${1:-status}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    status)
        check_status
        ;;
    logs)
        view_logs
        ;;
    restart)
        restart_server
        ;;
    *)
        echo "Usage: $0 {start|stop|status|logs|restart}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the Forma server"
        echo "  stop    - Stop the Forma server"
        echo "  status  - Check if server is running"
        echo "  logs    - View recent server logs"
        echo "  restart - Restart the server"
        exit 1
        ;;
esac