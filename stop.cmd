@echo off
setlocal
cd /d "%~dp0"
where pwsh >nul 2>nul
if errorlevel 1 goto use_windows_powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File "scripts\stop.ps1" %*
goto done
:use_windows_powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\stop.ps1" %*
:done
pause
endlocal
