#!/usr/bin/env python3
"""
Minimal test to diagnose the venv re-execution issue
"""
import sys
import os
from pathlib import Path

print(f"Python executable: {sys.executable}")
print(f"Virtual env active: {os.environ.get('ORTHOROUTE_VENV_ACTIVE', 'No')}")
print(f"Current working directory: {os.getcwd()}")

# Test PyQt6 import
try:
    import PyQt6.QtCore
    print("✓ PyQt6.QtCore imported successfully")
    
    from PyQt6.QtWidgets import QApplication
    print("✓ PyQt6.QtWidgets imported successfully")
    
    # Try creating a QApplication (this is often where crashes occur)
    app = QApplication([])
    print("✓ QApplication created successfully")
    
    print("🎉 All Qt tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Qt error: {e}")
    import traceback
    print(f"Traceback: {traceback.format_exc()}")

print("Test completed.")
