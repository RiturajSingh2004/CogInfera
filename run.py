"""
CogInfera — Run Script
Starts FastAPI (uvicorn) and Streamlit concurrently.
"""

import subprocess
import sys
import time
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

print("=" * 60)
print("  CogInfera — Starting Services")
print("=" * 60)
print()

# Start FastAPI
print("[1/2] Starting FastAPI on http://localhost:8000 ...")
api_proc = subprocess.Popen(
    [PYTHON, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
    cwd=ROOT,
)

# Brief pause so the API is ready before Streamlit tries to hit it
time.sleep(3)

# Start Streamlit
print("[2/2] Starting Streamlit on http://localhost:8501 ...")
ui_proc = subprocess.Popen(
    [PYTHON, "-m", "streamlit", "run", os.path.join(ROOT, "ui", "app.py"),
     "--server.port", "8501",
     "--server.headless", "true",
     "--theme.base", "dark"],
    cwd=ROOT,
)

print()
print("  FastAPI  →  http://localhost:8000")
print("  Streamlit → http://localhost:8501")
print()
print("  Press Ctrl+C to stop both services.")
print("=" * 60)

try:
    api_proc.wait()
except KeyboardInterrupt:
    print("\nShutting down...")
    api_proc.terminate()
    ui_proc.terminate()
