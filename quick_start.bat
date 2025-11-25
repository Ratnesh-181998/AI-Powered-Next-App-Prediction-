@echo off
REM Quick Start Script for iPhone App Prediction System
REM This script sets up and runs the complete pipeline

echo ============================================================
echo iPhone App Prediction System - Quick Start
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Step 1: Installing dependencies...
echo ------------------------------------------------------------
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.

echo Step 2: Creating directories...
echo ------------------------------------------------------------
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "models" mkdir models
if not exist "logs" mkdir logs
echo Directories created successfully
echo.

echo Step 3: Generating sample data (100 users, 7 days)...
echo ------------------------------------------------------------
python src\data_ingestion\generate_data.py --num-users 100 --days 7
if errorlevel 1 (
    echo ERROR: Failed to generate data
    pause
    exit /b 1
)
echo.

echo Step 4: Engineering features...
echo ------------------------------------------------------------
python src\feature_engineering\feature_engineer.py
if errorlevel 1 (
    echo ERROR: Failed to engineer features
    pause
    exit /b 1
)
echo.

echo Step 5: Training model...
echo ------------------------------------------------------------
python src\models\train_xgboost.py
if errorlevel 1 (
    echo ERROR: Failed to train model
    pause
    exit /b 1
)
echo.

echo ============================================================
echo SETUP COMPLETE!
echo ============================================================
echo.
echo The system is ready to use. You can now:
echo.
echo 1. Run the API server:
echo    python src\api\app.py
echo.
echo 2. Test predictions:
echo    python src\models\predictor.py
echo.
echo 3. Make API requests:
echo    curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d @sample_request.json
echo.
echo ============================================================
pause
