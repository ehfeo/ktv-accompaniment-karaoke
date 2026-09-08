@echo off
rem UVR-MDX-NET-Inst_HQ_3 stop script
setlocal
echo Stopping UVR-MDX-NET-Inst_HQ_3 server...
set "FOUND="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8900" ^| findstr "LISTENING"') do (
  set "FOUND=1"
  echo Killing PID %%p ...
  taskkill /F /PID %%p >nul 2>&1
)
if defined FOUND (
  echo Server stopped.
) else (
  echo No running server found on port 8900.
)
echo.
pause
