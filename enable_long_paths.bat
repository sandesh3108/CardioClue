@echo off
echo =====================================
echo Enabling Windows Long Path Support
echo =====================================
echo.
echo This script requires Administrator privileges.
echo You will be asked to grant permission.
echo.
pause

:: Enable Long Path Support in Registry
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d 1 /f

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Long path support has been enabled!
    echo.
    echo *** IMPORTANT: RESTART YOUR COMPUTER for changes to take effect ***
    echo.
    echo After restarting, run:
    echo   pip install -r requirements.txt
    echo.
) else (
    echo.
    echo [ERROR] Failed to enable long path support.
    echo Please run this script as Administrator.
    echo.
)

pause

