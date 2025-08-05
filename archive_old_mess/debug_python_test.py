#!/usr/bin/env python3
"""
Debug script to test Python executable detection
"""

import subprocess
import sys
from pathlib import Path

def test_python_executable(python_exe):
    """Test a Python executable with different strategies"""
    print(f"\n=== Testing: {python_exe} ===")
    
    # Test 1: Simple execution
    try:
        test_cmd = [str(python_exe), "-c", "print('SIMPLE_OK')"]
        print(f"Test command: {test_cmd}")
        result = subprocess.run(test_cmd, capture_output=True, timeout=5, shell=False)
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT: {result.stdout.decode('utf-8', errors='ignore')}")
        if result.stderr:
            print(f"STDERR: {result.stderr.decode('utf-8', errors='ignore')}")
        
        if result.returncode == 0 and b'SIMPLE_OK' in result.stdout:
            print("✅ Simple test PASSED")
            return True
        else:
            print("❌ Simple test FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Simple test EXCEPTION: {e}")
        return False

def main():
    print("=== Python Executable Detection Test ===")
    
    # Test current Python
    print(f"Current sys.executable: {sys.executable}")
    
    python_candidates = [
        sys.executable,
        "python",
        "python3", 
        "py"
    ]
    
    working_pythons = []
    
    for python_exe in python_candidates:
        if test_python_executable(python_exe):
            working_pythons.append(python_exe)
    
    print(f"\n=== SUMMARY ===")
    print(f"Working Python executables: {working_pythons}")
    
    if working_pythons:
        print("✅ At least one Python executable works")
        
        # Test server script
        plugin_dir = Path(__file__).parent / "addon_package" / "plugins"
        server_script = plugin_dir / "robust_server.py"
        
        if server_script.exists():
            print(f"\n=== Testing server script: {server_script} ===")
            
            for python_exe in working_pythons[:1]:  # Test with first working Python
                try:
                    cmd = [str(python_exe), str(server_script), "--help"]
                    print(f"Server test command: {cmd}")
                    result = subprocess.run(cmd, capture_output=True, timeout=10, shell=False)
                    print(f"Server return code: {result.returncode}")
                    if result.stdout:
                        print(f"Server STDOUT: {result.stdout.decode('utf-8', errors='ignore')[:500]}")
                    if result.stderr:
                        print(f"Server STDERR: {result.stderr.decode('utf-8', errors='ignore')[:500]}")
                except Exception as e:
                    print(f"Server test exception: {e}")
        else:
            print(f"❌ Server script not found: {server_script}")
    else:
        print("❌ No working Python executables found")

if __name__ == "__main__":
    main()
