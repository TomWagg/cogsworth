#!/usr/bin/env bash
# Start the cogsworth worker and Streamlit app together.
# Usage: bash webapp/run.sh [extra streamlit args]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO/webapp/results"

# Reset any jobs that were left in 'running' state by a previous crashed worker
python "$REPO/webapp/worker.py" --reset-only 2>/dev/null || true

# Launch worker in the background
echo "Starting cogsworth job worker..."
python "$REPO/webapp/worker.py" &
WORKER_PID=$!

cleanup() {
    echo "Stopping worker (PID $WORKER_PID)..."
    kill "$WORKER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting Streamlit on http://0.0.0.0:8501"
streamlit run "$REPO/webapp/app.py" \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    "$@"
