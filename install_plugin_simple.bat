@echo off
echo OrthoRoute KiCad Plugin Installation
echo ===================================
echo.

REM Test the plugin imports first
echo Testing plugin imports...
python test_plugin_imports.py
if errorlevel 1 (
    echo.
    echo ERROR: Plugin imports failed. Please fix the issues above before installing.
    pause
    exit /b 1
)

echo.
echo Plugin imports are working correctly!
echo.

REM Find KiCad installation
set KICAD_VERSIONS=9.0 8.0 7.0 6.0
set KICAD_FOUND=0

for %%v in (%KICAD_VERSIONS%) do (
    if exist "%USERPROFILE%\Documents\KiCad\%%v" (
        set KICAD_VERSION=%%v
        set KICAD_FOUND=1
        goto :found_kicad
    )
)

:found_kicad
if %KICAD_FOUND%==0 (
    echo ERROR: Could not find KiCad documents directory.
    echo Please ensure KiCad is installed and you have run it at least once.
    pause
    exit /b 1
)

echo Found KiCad %KICAD_VERSION% documents directory
set PLUGIN_DIR=%USERPROFILE%\Documents\KiCad\%KICAD_VERSION%\scripting\plugins\OrthoRoute

echo.
echo Installing plugin to: %PLUGIN_DIR%
echo.

REM Create plugin directory
if not exist "%PLUGIN_DIR%" (
    mkdir "%PLUGIN_DIR%"
    echo Created plugin directory
)

REM Copy plugin files
echo Copying plugin files...
xcopy "kicad_plugin\*" "%PLUGIN_DIR%\" /Y /E /Q
if errorlevel 1 (
    echo ERROR: Failed to copy plugin files
    pause
    exit /b 1
)

echo.
echo ✓ Plugin installed successfully!
echo.
echo Next steps:
echo 1. Restart KiCad if it's currently running
echo 2. Open the PCB Editor
echo 3. Look for "OrthoRoute GPU Autorouter" in the toolbar
echo 4. If the icon doesn't appear, check Tools ^> Plugin and Content Manager
echo.
echo The plugin files are located at:
echo %PLUGIN_DIR%
echo.
pause
