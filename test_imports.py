"""
Test script to debug import issues in KiCad plugin directory
Run this from the KiCad plugin directory to see what imports work
"""
import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("=== Import Test for OrthoRoute Plugin ===")
print(f"Current directory: {current_dir}")
print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
print()

# Test basic imports
print("1. Testing basic imports...")
try:
    import cupy as cp
    print("✓ CuPy import successful")
except ImportError as e:
    print(f"✗ CuPy import failed: {e}")

try:
    import numpy as np
    print("✓ NumPy import successful")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")

try:
    import wx
    print("✓ wxPython import successful")
except ImportError as e:
    print(f"✗ wxPython import failed: {e}")

print()

# Test OrthoRoute imports
print("2. Testing OrthoRoute imports...")

# Test standalone wave router
try:
    from standalone_wave_router import WaveRouter
    print("✓ standalone_wave_router import successful")
except ImportError as e:
    print(f"✗ standalone_wave_router import failed: {e}")

# Test gpu_engine
try:
    from gpu_engine import OrthoRouteEngine
    print("✓ gpu_engine import successful")
except ImportError as e:
    print(f"✗ gpu_engine import failed: {e}")

# Test full plugin
try:
    from full_plugin import OrthoRoutePlugin
    print("✓ full_plugin import successful")
except ImportError as e:
    print(f"✗ full_plugin import failed: {e}")

print()
print("=== Import Test Complete ===")

# Try to create an engine instance
print("3. Testing engine creation...")
try:
    from gpu_engine import OrthoRouteEngine
    engine = OrthoRouteEngine()
    print("✓ Engine creation successful")
    print(f"Engine ID: {engine.engine_id}")
except Exception as e:
    print(f"✗ Engine creation failed: {e}")

print()
print("=== Test Complete ===")
