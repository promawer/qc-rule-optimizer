@echo off
setlocal
cd /d "%~dp0"
where pwsh >nul 2>nul
if errorlevel 1 goto use_windows_powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File "scripts\dev.ps1" %*
goto done
:use_windows_powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\dev.ps1" %*
:done
if not errorlevel 1 goto end
echo.
echo Startup failed. Check the messages above and .run\app.log
pause
:end
endlocal
