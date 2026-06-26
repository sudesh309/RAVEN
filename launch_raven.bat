@echo off
setlocal

REM ── RAVEN – Requirement Analysis and Visualisation Engine ──────────────────
REM  Double-click this file to launch the RAVEN GUI in your browser.
REM  Requirements: Python 3.10+ on PATH, dependencies installed (see README).
REM ───────────────────────────────────────────────────────────────────────────

REM Move to the directory containing this script (the project root)
cd /d "%~dp0"

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

REM Install / upgrade dependencies silently on first run
echo Installing dependencies ...
python -m pip install -e ".[io]" --quiet
python -m pip install rdflib --quiet

REM Launch the GUI (opens browser automatically)
echo Starting RAVEN GUI ...
python -m reqgraph gui

pause
