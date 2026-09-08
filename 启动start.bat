@echo off
rem UVR-MDX-NET-Inst_HQ_3 portable starter
setlocal
cd /d "%~dp0"
rem open browser then start server
start "" "http://127.0.0.1:8900/"
"%~dp0runtime\python.exe" "%~dp0server.py"
pause
