#!/usr/bin/env python3
"""
Simple KiCad Test Script
Tests OrthoRoute with KiCad CLI
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("OrthoRoute KiCad CLI Test")
    print("=" * 60)
    
    # KiCad installation path
    kicad_cli_path = r"C:\Program Files\KiCad\9.0\bin\kicad-cli.exe"
    project_dir = Path(__file__).parent.absolute()
    
    # Test 1: Check KiCad CLI version
    print("\n1. Checking KiCad CLI...")
    try:
        result = subprocess.run([
            kicad_cli_path, '--version'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[OK] KiCad CLI found: {result.stdout.strip()}")
        else:
            print(f"[FAIL] KiCad CLI check failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print(f"[FAIL] KiCad CLI not found at: {kicad_cli_path}")
        return False
    except Exception as e:
        print(f"[FAIL] Error checking KiCad CLI: {e}")
        return False
    
    # Test 2: Create a simple headless test script
    print("\n2. Creating simple test script...")
    simple_test_content = '''
import sys
print("=== KiCad Headless Test ===")

try:
    import pcbnew
    print("[OK] pcbnew imported successfully")
    
    # Try to get board (might be None in headless mode)
    board = pcbnew.GetBoard()
    if board:
        print("[OK] Board available")
        
        # Test basic board operations
        try:
            bounds = board.GetBoardEdgesBoundingBox()
            width = bounds.GetWidth()
            height = bounds.GetHeight()
            layers = board.GetCopperLayerCount()
            print(f"[OK] Board size: {width/1e6:.2f} x {height/1e6:.2f} mm")
            print(f"[OK] Copper layers: {layers}")
            
            # Test footprints
            footprints = list(board.GetFootprints())
            print(f"[OK] Found {len(footprints)} footprints")
            
            # Test nets
            netcodes = board.GetNetsByNetcode()
            print(f"[OK] Found {len(netcodes)} nets")
            
        except Exception as e:
            print(f"[FAIL] Board operations failed: {e}")
    else:
        print("[INFO] No board loaded (normal for headless mode)")
        
    print("[OK] KiCad API test completed successfully")
    
except ImportError as e:
    print(f"[FAIL] pcbnew import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[FAIL] Test failed: {e}")
    sys.exit(1)
'''
    
    simple_test_file = project_dir / "simple_kicad_test.py"
    try:
        with open(simple_test_file, 'w') as f:
            f.write(simple_test_content)
        print(f"[OK] Test script created: {simple_test_file}")
    except Exception as e:
        print(f"[FAIL] Could not create test script: {e}")
        return False
    
    # Test 3: Run KiCad CLI with test board
    print("\n3. Running KiCad CLI test...")
    try:
        cli_result = subprocess.run([
            kicad_cli_path, 'pcb', 
            '--input', 'test_board.kicad_pcb',
            '--python-script', str(simple_test_file)
        ], capture_output=True, text=True, cwd=project_dir)
        
        print("CLI Output:")
        print("-" * 40)
        print(cli_result.stdout)
        if cli_result.stderr:
            print("CLI Errors:")
            print("-" * 40)
            print(cli_result.stderr)
        
        if cli_result.returncode == 0:
            print("[OK] KiCad CLI test completed successfully")
            return True
        else:
            print(f"[FAIL] KiCad CLI test failed with code {cli_result.returncode}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Error running KiCad CLI test: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    print("\n" + "=" * 60)
    if success:
        print("[OK] All tests passed!")
        print("OrthoRoute is ready for KiCad testing")
    else:
        print("[FAIL] Some tests failed")
        print("Check the output above for details")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
