@echo off
rem UVR window launcher (no console)
cd /d "%~dp0"
"%~dp0runtime\pythonw.exe" "%~dp0main_window.py" > "%~dp0runtime_error.log" 2>&1

