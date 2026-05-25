@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv" (
    echo ERROR: virtual environment not found.
    echo Run install_windows_gpu.bat first.
    pause
    exit /b 1
)

if not exist "app.py" (
    echo ERROR: app.py not found.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run app.py

pause