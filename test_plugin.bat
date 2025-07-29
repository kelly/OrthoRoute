@echo off
echo Testing OrthoRoute Plugin Installation...
echo.

echo Checking plugin directory...
if exist "kicad_plugin\icon.png" (
    echo ✓ Icon file found
) else (
    echo ✗ Icon file missing
)

if exist "kicad_plugin\register_plugin.py" (
    echo ✓ Plugin registration file found
) else (
    echo ✗ Plugin registration file missing
)

if exist "kicad_plugin\orthoroute_kicad.py" (
    echo ✓ Main plugin file found
) else (
    echo ✗ Main plugin file missing
)

echo.
echo Testing Python imports...
python -c "import sys; print('Python version:', sys.version)"
python -c "import cupy; print('CuPy available:', cupy.__version__)" 2>nul || echo "CuPy not available"
python -c "import tkinter; print('Tkinter available')" 2>nul || echo "Tkinter not available"

echo.
echo Plugin files ready for KiCad installation
echo Copy the kicad_plugin folder to your KiCad scripting/plugins directory
pause
