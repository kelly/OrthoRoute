@echo off
echo ========================================
echo OrthoRoute Pre-Package Test Suite
echo ========================================

REM Change to the script directory
cd /d "%~dp0"

echo.
echo Running comprehensive tests...
python run_all_tests.py

echo.
echo ========================================
echo Test run complete!
echo Check comprehensive_test_results.json for details
echo ========================================

pause
