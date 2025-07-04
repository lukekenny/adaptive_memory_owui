@echo off
REM Installation script for OWUI Adaptive Memory Plugin (Windows)
REM This script handles installation and verification in one step

setlocal enabledelayedexpansion

REM Colors not available in batch, using plain text
echo ===================================================
echo OWUI Adaptive Memory Plugin Installation (Windows)
echo ===================================================
echo.

REM Step 1: Check Python version
echo [*] Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] Python not found. Please install Python 3.8 or higher
    echo     Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [+] Found Python %PYTHON_VERSION%

REM Step 2: Create virtual environment
if not exist "venv" (
    echo [*] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [X] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [+] Virtual environment created
) else (
    echo [!] Virtual environment already exists
)

REM Step 3: Activate virtual environment
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [!] Failed to activate virtual environment, using system Python
)

REM Step 4: Upgrade pip
echo [*] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

REM Step 5: Install dependencies
echo [*] Installing dependencies...
if exist "requirements.txt" (
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [X] Failed to install some dependencies
        echo [!] Trying to install core dependencies individually...
        
        for %%p in (pydantic numpy aiohttp pytz) do (
            echo [*] Installing %%p...
            pip install %%p
        )
    )
    echo [+] Dependencies installed
) else (
    echo [X] requirements.txt not found
    pause
    exit /b 1
)

REM Step 6: Run quick verification
echo.
echo ===================================================
echo Running Quick Verification
echo ===================================================
echo.

python quick_verify.py
if %errorlevel% neq 0 (
    echo [X] Quick verification failed
    echo [!] Attempting full verification with auto-fix...
    
    python post_install_verification.py --auto-fix
    if %errorlevel% neq 0 (
        echo [X] Installation verification failed
        pause
        exit /b 1
    )
)

REM Step 7: Create necessary directories
echo [*] Creating plugin directories...
if not exist "logs" mkdir logs
if not exist "memory-bank" mkdir memory-bank
if not exist "tests\unit" mkdir tests\unit
if not exist "tests\integration" mkdir tests\integration

REM Step 8: Check for Docker/OpenWebUI
echo.
echo ===================================================
echo Checking OpenWebUI Integration
echo ===================================================
echo.

docker --version >nul 2>&1
if %errorlevel% equ 0 (
    docker ps | findstr openwebui >nul 2>&1
    if %errorlevel% equ 0 (
        echo [+] OpenWebUI container detected
        echo [*] Testing OpenWebUI integration...
        python verify_openwebui_integration.py
        if %errorlevel% neq 0 (
            echo [!] OpenWebUI integration test failed
            echo [!] This is normal if OpenWebUI is not yet configured
        )
    ) else (
        echo [!] OpenWebUI container not running
        echo [!] Start OpenWebUI before uploading the plugin
    )
) else (
    echo [!] Docker not installed - skipping OpenWebUI check
)

REM Final summary
echo.
echo ===================================================
echo Installation Complete!
echo ===================================================
echo.
echo Next steps:
echo 1. Upload adaptive_memory_v4.0.py to OpenWebUI:
echo    - Go to Workspace -^> Functions
echo    - Click '+' to add new function
echo    - Upload the file and save
echo.
echo 2. Enable the filter for your models:
echo    - Go to Workspace -^> Models
echo    - Select models to use with memory
echo    - Enable the Adaptive Memory filter
echo.
echo 3. Configure the plugin (optional):
echo    - Click the settings icon on the filter
echo    - Adjust memory settings as needed
echo.
echo 4. Start chatting with persistent memory!
echo.
echo For troubleshooting, run:
echo   python post_install_verification.py --save-report
echo.
echo [+] Installation completed successfully!
echo.

REM Deactivate virtual environment
call deactivate >nul 2>&1

pause