#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# (Istersen) veriyi tazele
python -m fetch.prices
python -m fetch.compute

# Backend (FastAPI) arka planda
export AI_SERVER_URL="${AI_SERVER_URL:-http://127.0.0.1:8008}"
uvicorn backend.ai_server:APP --host 0.0.0.0 --port 8008 --env-file .env &
UVICORN_PID=$!

# Streamlit (ön planda)
streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501

# Streamlit kapaninca backend’i durdur
kill "$UVICORN_PID" 2>/dev/null || true
