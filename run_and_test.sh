#!/bin/bash
# Run backend and quick API test. Use: ./run_and_test.sh
cd "$(dirname "$0")"

echo "=== 1. Checking .env and OPENAI_API_KEY ==="
if [ ! -f .env ]; then
  echo "ERROR: .env not found. Copy .env.example to .env and add your OPENAI_API_KEY."
  exit 1
fi
if ! grep -q "OPENAI_API_KEY=sk-" .env 2>/dev/null; then
  echo "WARNING: OPENAI_API_KEY may not be set in .env (expected value starting with sk-)."
fi
echo "OK: .env present"

echo ""
echo "=== 2. Starting backend on http://127.0.0.1:8000 ==="
echo "    (Stop with Ctrl+C. Then open http://127.0.0.1:8000 in your browser.)"
echo ""
.venv/bin/uvicorn backend:app --host 127.0.0.1 --port 8000
