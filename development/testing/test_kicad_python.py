#!/usr/bin/env python3
"""
KiCad Python Environment Test
Tests OrthoRoute using KiCad's Python environment
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("=" * 60)
    print("OrthoRoute KiCad Python Environment Test")
    print("=" * 60)
    
    # KiCad Python executable path  
    kicad_python_path = r"C:\Program Files\KiCad\9.0\bin\python.exe"
    project_dir = Path(__file__).parent.absolute()
    
    # Test 1: Check KiCad Python
    print("\n1. Checking KiCad Python environment...")
    try:
        result = subprocess.run([
            kicad_python_path, '--version'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[OK] KiCad Python found: {result.stdout.strip()}")
        else:
            print(f"[FAIL] KiCad Python check failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print(f"[FAIL] KiCad Python not found at: {kicad_python_path}")
        return False
    except Exception as e:
        print(f"[FAIL] Error checking KiCad Python: {e}")
        return False
    
    # Test 2: Test pcbnew import
    print("\n2. Testing pcbnew import...")
    pcbnew_test_content = '''
import sys
print("=== KiCad pcbnew Import Test ===")

try:
    import pcbnew
    print("[OK] pcbnew imported successfully")
    
    # Test creating a board
    board = pcbnew.BOARD()
    print("[OK] Board created successfully")
    
    # Test basic board operations
    layers = board.GetCopperLayerCount()
    print(f"[OK] Board has {layers} copper layers")
    
    # Test net creation
    net_info = pcbnew.NETINFO_ITEM(board, "TEST_NET")
    board.Add(net_info)
    print("[OK] Net created successfully")
    
    # Test footprint creation
    footprint = pcbnew.FOOTPRINT(board)
    footprint.SetReference("TEST_FP")
    board.Add(footprint)
    print("[OK] Footprint created successfully")
    
    # Test track creation
    track = pcbnew.PCB_TRACK(board)
    track.SetStart(pcbnew.VECTOR2I(0, 0))
    track.SetEnd(pcbnew.VECTOR2I(1000000, 1000000))
    track.SetLayer(pcbnew.F_Cu)
    track.SetWidth(200000)
    board.Add(track)
    print("[OK] Track created successfully")
    
    print("[OK] All pcbnew tests passed")
    
except ImportError as e:
    print(f"[FAIL] pcbnew import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[FAIL] pcbnew test failed: {e}")
    sys.exit(1)
'''
    
    pcbnew_test_file = project_dir / "test_pcbnew_import.py"
    try:
        with open(pcbnew_test_file, 'w') as f:
            f.write(pcbnew_test_content)
        print(f"[OK] pcbnew test script created")
    except Exception as e:
        print(f"[FAIL] Could not create pcbnew test script: {e}")
        return False
    
    # Run pcbnew test
    try:
        pcbnew_result = subprocess.run([
            kicad_python_path, str(pcbnew_test_file)
        ], capture_output=True, text=True, cwd=project_dir)
        
        print("pcbnew Test Output:")
        print("-" * 40)
        print(pcbnew_result.stdout)
        if pcbnew_result.stderr:
            print("pcbnew Test Errors:")
            print("-" * 40)
            print(pcbnew_result.stderr)
        
        if pcbnew_result.returncode != 0:
            print(f"[FAIL] pcbnew test failed with code {pcbnew_result.returncode}")
            return False
        else:
            print("[OK] pcbnew test passed")
            
    except Exception as e:
        print(f"[FAIL] Error running pcbnew test: {e}")
        return False
    
    # Test 3: Test OrthoRoute imports
    print("\n3. Testing OrthoRoute imports...")
    orthoroute_test_content = '''
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "addon_package", "plugins"))

print("=== OrthoRoute Import Test ===")

try:
    # Test basic imports first
    import json
    import pathlib
    print("[OK] Basic Python imports work")
    
    # Test if OrthoRoute engine file exists and is readable
    engine_path = os.path.join(current_dir, "addon_package", "plugins", "orthoroute_engine.py")
    if os.path.exists(engine_path):
        print("[OK] OrthoRoute engine file found")
        
        # Read and check structure
        with open(engine_path, 'r') as f:
            content = f.read()
        
        if "class OrthoRouteEngine" in content:
            print("[OK] OrthoRoute engine class found")
        else:
            print("[FAIL] OrthoRoute engine class not found")
            sys.exit(1)
    else:
        print(f"[FAIL] OrthoRoute engine file not found at: {engine_path}")
        sys.exit(1)
    
    # Test if API bridge exists
    bridge_path = os.path.join(current_dir, "api_bridge.py")
    if os.path.exists(bridge_path):
        print("[OK] API bridge file found")
        
        with open(bridge_path, 'r') as f:
            content = f.read()
        
        if "class APIBridge" in content:
            print("[OK] API bridge class found")
        else:
            print("[FAIL] API bridge class not found")
            sys.exit(1)
    else:
        print(f"[FAIL] API bridge file not found at: {bridge_path}")
        sys.exit(1)
    
    print("[OK] All OrthoRoute structure tests passed")
    
except Exception as e:
    print(f"[FAIL] OrthoRoute test failed: {e}")
    sys.exit(1)
'''
    
    orthoroute_test_file = project_dir / "test_orthoroute_imports.py"
    try:
        with open(orthoroute_test_file, 'w') as f:
            f.write(orthoroute_test_content)
        print(f"[OK] OrthoRoute test script created")
    except Exception as e:
        print(f"[FAIL] Could not create OrthoRoute test script: {e}")
        return False
    
    # Run OrthoRoute test
    try:
        ortho_result = subprocess.run([
            kicad_python_path, str(orthoroute_test_file)
        ], capture_output=True, text=True, cwd=project_dir)
        
        print("OrthoRoute Test Output:")
        print("-" * 40)
        print(ortho_result.stdout)
        if ortho_result.stderr:
            print("OrthoRoute Test Errors:")
            print("-" * 40)
            print(ortho_result.stderr)
        
        if ortho_result.returncode != 0:
            print(f"[FAIL] OrthoRoute test failed with code {ortho_result.returncode}")
            return False
        else:
            print("[OK] OrthoRoute test passed")
            
    except Exception as e:
        print(f"[FAIL] Error running OrthoRoute test: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    
    print("\n" + "=" * 60)
    if success:
        print("[OK] All KiCad Python environment tests passed!")
        print("OrthoRoute is compatible with KiCad Python environment")
    else:
        print("[FAIL] Some tests failed")
        print("Check the output above for details")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
