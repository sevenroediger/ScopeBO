@echo off
cd /d "%~dp0Webapp"
call conda activate scope_bo && python app.py
pause