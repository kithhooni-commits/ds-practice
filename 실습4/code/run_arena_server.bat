@echo off
chcp 65001 > nul
setlocal

echo ========================================================
echo   🥊 4-Player AR Shadow Boxing & Battle Arena 실행
echo ========================================================
echo.

set "PYTHON_EXE=C:\Users\%USERNAME%\.conda\envs\pjt-4\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo [1] Host 대형 3D 링 화면: https://localhost:8000/arena
echo [2] 4인 파이터 웹캠 접속 주소 (다른 랩탑 브라우저):
echo     - Fighter 1 (Red)   : https://147.47.201.63:8000/client?id=client_1
echo     - Fighter 2 (Cyan)  : https://147.47.201.63:8000/client?id=client_2
echo     - Fighter 3 (Gold)  : https://147.47.201.63:8000/client?id=client_3
echo     - Fighter 4 (Green) : https://147.47.201.63:8000/client?id=client_4
echo.
echo 서버를 시작합니다... (종료: Ctrl+C)
echo.

"%PYTHON_EXE%" "%~dp0run_arena_server.py"
pause
