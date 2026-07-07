@echo off
cd /d "%~dp0"
call conda activate scope_bo && python .\Webapp\app.py
pause