#!/usr/bin/env python3
"""
Check KiCad's Python environment to understand where to install packages
"""

import sys
import os
import subprocess

print("=" * 60)
print("KICAD PYTHON ENVIRONMENT DIAGNOSTIC")
print("=" * 60)

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: ")
for path in sys.path:
    print(f"  {path}")

print("\n" + "=" * 60)
print("TRYING TO IMPORT KIPY")
print("=" * 60)

try:
    import kipy
    print(f"✅ kipy successfully imported from: {kipy.__file__}")
    print(f"   kipy version: {getattr(kipy, '__version__', 'unknown')}")
except ImportError as e:
    print(f"❌ kipy import failed: {e}")
    
print("\n" + "=" * 60)
print("PACKAGE INSTALLATION LOCATIONS")
print("=" * 60)

# Check where packages would be installed
try:
    import site
    print("Site packages directories:")
    for path in site.getsitepackages():
        print(f"  {path}")
    
    user_site = site.getusersitepackages()
    print(f"User site packages: {user_site}")
    
except Exception as e:
    print(f"Error getting site packages: {e}")

print("\n" + "=" * 60)
print("ENVIRONMENT VARIABLES")
print("=" * 60)

relevant_vars = ['PYTHONPATH', 'PATH', 'KICAD_USER_DIR', 'APPDATA']
for var in relevant_vars:
    value = os.environ.get(var, 'Not set')
    print(f"{var}: {value}")

print("\n" + "=" * 60)
print("INSTALLATION COMMAND FOR THIS ENVIRONMENT")
print("=" * 60)

print(f"To install kicad-python in this Python environment, run:")
print(f"{sys.executable} -m pip install kicad-python")
