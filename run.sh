#!/bin/bash

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source public_grievance_urgency_classification_system_venv/bin/activate

echo -e "${BLUE}Starting Public Grievance Urgency Classification System...${NC}\n"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Stopping services...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    wait $BACKEND_PID 2>/dev/null || true
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

# Start backend server in background
echo -e "${GREEN}Starting backend API on port 8000...${NC}"
uvicorn backend.api:app --port 8000 > /tmp/backend.log 2>&1 &
BACKEND_PID=$!

# Give backend time to start
sleep 2

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${YELLOW}Backend failed to start. Error log:${NC}"
    cat /tmp/backend.log
    exit 1
fi

# Start frontend in foreground
echo -e "${GREEN}Starting frontend on port 8501...${NC}\n"
export STREAMLIT_SERVER_HEADLESS=true
streamlit run frontend/app.py --logger.level=warning
