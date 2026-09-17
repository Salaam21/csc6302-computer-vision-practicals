@echo off
cd /d "%~dp0"
echo.
echo CSC 6302 Capstone - Road and transport demo
echo Installing packages if needed (first time can take several minutes)...
echo.
python -m pip install -r requirements-demo.txt
if errorlevel 1 (
  echo pip failed. Try: py -3.11 -m pip install -r requirements-demo.txt
  pause
  exit /b 1
)
echo.
echo Starting Gradio. Browser should open on http://127.0.0.1:7860
echo First vehicle count downloads YOLOv8n weights once (~6 MB) then caches them.
echo Leave this window open while you use the app. Ctrl+C to stop.
echo.
python app.py
pause
