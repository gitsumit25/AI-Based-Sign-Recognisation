@echo off
setlocal
cd /d "%~dp0"
echo ===============================================
echo  SIGNOVA - Real-Time Sign Language Recognition + Accuracy Studio
echo ===============================================
where py >nul 2>nul
if errorlevel 1 (
  echo Python Launcher not found. Install Python 3.10 from python.org and try again.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating Python virtual environment...
  py -3.10 -m venv .venv
  if errorlevel 1 (
    echo Could not create the environment. Make sure Python 3.10 is installed.
    pause
    exit /b 1
  )
)
echo Installing/checking dependencies. First run may take several minutes...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed. Review the error above.
  pause
  exit /b 1
)
echo Starting backend in a separate terminal...
start "SIGNOVA Backend" "%~dp0.venv\Scripts\python.exe" -m uvicorn backend.app:app --host 127.0.0.1 --port 8001
 timeout /t 4 /nobreak >nul
start "" "http://127.0.0.1:8001"
echo.
echo SIGNOVA should now be available at http://127.0.0.1:8001
echo Keep the SIGNOVA Backend terminal open while using the website.
echo If the browser did not open, copy the address above into Chrome.
timeout /t 5 /nobreak >nul
exit /b 0
