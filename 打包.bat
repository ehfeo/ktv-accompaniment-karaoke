@echo off
rem One-click release build (onedir)
cd /d "%~dp0"
"%~dp0runtime\python.exe" build_release.py
pause