@echo off
rem Drag a leaf photo (or a folder of photos) onto this file, or run:  run_predict.bat "C:\path\to\leaf.jpg"
rem Optional extra arguments are passed through, e.g.:  run_predict.bat "leaf.jpg" --variant image --no-cam
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo.
  echo Usage: drag a leaf photo ^(or a folder of photos^) onto run_predict.bat
  echo   https://static.vecteezy.com/system/resources/previews/063/105/054/large_2x/close-up-nature-view-of-fresh-green-leaf-against-lush-blurred-background-natural-organic-pattern-calm-forest-atmosphere-free-photo.jpeg     or run:  run_predict.bat "C:\path\to\leaf.jpg"
  echo.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo.
  echo ERROR: .venv\Scripts\python.exe not found. Keep this file inside the fruit_health folder and see README.md ^(Environment setup^).
  echo.
  pause
  exit /b 1
)
set "ATTR=%~a1"
if "%ATTR:~0,1%"=="d" (
  ".venv\Scripts\python.exe" -I src\predict.py --folder "%~1" %2 %3 %4 %5 %6 %7 %8 %9
) else (
  ".venv\Scripts\python.exe" -I src\predict.py --image "%~1" %2 %3 %4 %5 %6 %7 %8 %9
)
echo.
echo Results (CSV table and heatmap images) are saved in: %~dp0outputs\predictions\
echo.
pause
