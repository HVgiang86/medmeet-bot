@echo off
SETLOCAL EnableDelayedExpansion

echo ==================================
echo MedBot - Setup and Launch Script
echo ==================================

:: Check if Poetry is available
where poetry >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Poetry is not found. Please install Poetry before running this script.
    echo You can install it using: curl -sSL https://install.python-poetry.org ^| python3 -
    pause
    exit /b 1
) else (
    echo Poetry is available.
)

:: Check for .env file
if not exist ".env" (
    echo Warning: .env file not found.
    echo Creating a basic .env file...
    echo # Add your environment variables here > .env
    echo CREATED_AT=%date% %time% >> .env
)

:: Install dependencies using Poetry
echo Installing project dependencies...
call poetry install
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies. Exiting...
    exit /b 1
)
echo Dependencies installed successfully.

:: Check if Ollama is running
echo Checking if Ollama is running...
powershell -Command "if ((Get-NetTCPConnection -LocalPort 11434 -ErrorAction SilentlyContinue).Count -eq 0) { Write-Host 'Ollama is not running. Please start Ollama before proceeding.' -ForegroundColor Red; exit 1 } else { Write-Host 'Ollama is running.' -ForegroundColor Green }"
if %ERRORLEVEL% NEQ 0 (
    echo Please start Ollama and run this script again.
    pause
    exit /b 1
)

:: Check if required models are available
echo Checking if required models are available...
call poetry run python -c "import os; os.system('ollama list | findstr \"llama3.2 nomic-embed-text\"')"

echo Starting Streamlit application...
call poetry run streamlit run app.py

echo Application has stopped.
pause
ENDLOCAL