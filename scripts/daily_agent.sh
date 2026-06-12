#!/bin/zsh
# Morning research agent — logs to logs/agent.log

set -euo pipefail

PROJECT="/Users/erinlee/Agentic AI"
PYTHON="/opt/anaconda3/bin/python"
LOG_DIR="$PROJECT/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT"

echo "=== $(date -Iseconds) ===" >> "$LOG_DIR/agent.log"
"$PYTHON" agent.py >> "$LOG_DIR/agent.log" 2>&1
"$PYTHON" scripts/export_site.py >> "$LOG_DIR/agent.log" 2>&1
