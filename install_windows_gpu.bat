@echo off
setlocal

cd /d "%~dp0"

echo ============================================================
echo RFRApp Windows GPU installer
echo ============================================================

echo.
echo [1/9] Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found.
    echo Install Python 3.11 from python.org and add it to PATH.
    pause
    exit /b 1
)

echo.
echo [2/9] Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv --copies
    if errorlevel 1 (
        echo ERROR: failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

echo.
echo [3/9] Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: failed to upgrade pip.
    pause
    exit /b 1
)

echo.
echo [4/9] Installing base requirements...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: failed to install requirements.
    pause
    exit /b 1
)

echo.
echo [5/9] Installing PyTorch CUDA 12.8...
".venv\Scripts\python.exe" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo ERROR: failed to install PyTorch.
    pause
    exit /b 1
)

echo.
echo [6/9] Checking CUDA Toolkit...

where nvcc >nul 2>nul
if errorlevel 1 (
    echo ERROR: nvcc not found.
    echo Install CUDA Toolkit 12.8.
    pause
    exit /b 1
)

echo nvcc found:
where nvcc

echo.
echo [7/9] Activating Visual Studio C++ build environment...

where cl >nul 2>nul
if errorlevel 1 (
    echo cl.exe not found in current shell.
    echo Trying to activate Visual Studio Build Tools automatically...

    set "VCVARS64="

    for %%E in (BuildTools Community Professional Enterprise) do (
        if exist "%ProgramFiles%\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat" (
            set "VCVARS64=%ProgramFiles%\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat"
        )

        if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat" (
            set "VCVARS64=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat"
        )
    )

    if "%VCVARS64%"=="" (
        echo ERROR: vcvars64.bat not found.
        echo Install Visual Studio Build Tools 2022 with "Desktop development with C++".
        pause
        exit /b 1
    )

    echo Found vcvars64:
    echo %VCVARS64%

    call "%VCVARS64%"
)

where cl >nul 2>nul
if errorlevel 1 (
    echo ERROR: cl.exe still not found after Visual Studio environment activation.
    pause
    exit /b 1
)

echo cl found:
where cl

echo.
echo [8/9] Building DCN...

set DISTUTILS_USE_SDK=1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8

cd model\dcn

if exist build (
    rmdir /s /q build
)

del /s /q DCN2*.pyd 2>nul

"%~dp0.venv\Scripts\python.exe" setup.py build_ext --inplace
if errorlevel 1 (
    echo ERROR: DCN build failed.
    pause
    exit /b 1
)

cd "%~dp0"

echo.
echo [9/9] Checking imports...

".venv\Scripts\python.exe" -c "from model.dcn.modules.deform_conv import DeformConv; print('DCN OK')"
if errorlevel 1 (
    echo ERROR: DCN import failed.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "from model.RFR_framework import RFR; print('RFR OK')"
if errorlevel 1 (
    echo ERROR: RFR import failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Installation completed successfully.
echo ============================================================
pause