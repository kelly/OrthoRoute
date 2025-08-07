#!/usr/bin/env python3
"""
Debug test for KiCad subprocess launching
This helps diagnose exactly what's failing when launching external processes from KiCad
"""

import sys
import os
import subprocess
import tempfile
import time
from pathlib import Path

def test_subprocess_from_kicad():
    """Test various subprocess launching methods"""
    
    print("=== KiCad Subprocess Debug Test ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Environment PATH: {os.environ.get('PATH', 'Not found')[:200]}...")
    
    # Create a simple test script
    test_script_content = '''
import sys
import time
print(f"TEST SCRIPT: Started with args: {sys.argv}")
print(f"TEST SCRIPT: Python executable: {sys.executable}")
print(f"TEST SCRIPT: Working directory: {os.getcwd()}")
time.sleep(1)
print("TEST SCRIPT: Completed successfully")
'''
    
    # Write test script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script_content)
        test_script_path = f.name
    
    print(f"Created test script: {test_script_path}")
    
    try:
        # Test 1: Basic Python call
        print("\n--- Test 1: Basic subprocess call ---")
        try:
            result = subprocess.run([sys.executable, "--version"], capture_output=True, timeout=5)
            print(f"SUCCESS: {result.stdout.decode().strip()}")
        except Exception as e:
            print(f"FAILED: {e}")
        
        # Test 2: Run our test script directly
        print("\n--- Test 2: Direct script execution ---")
        try:
            result = subprocess.run([sys.executable, test_script_path, "test_arg"], 
                                  capture_output=True, timeout=10, shell=False)
            print(f"Return code: {result.returncode}")
            print(f"STDOUT: {result.stdout.decode()}")
            print(f"STDERR: {result.stderr.decode()}")
        except Exception as e:
            print(f"FAILED: {e}")
        
        # Test 3: Popen with different configurations
        print("\n--- Test 3: Popen tests ---")
        
        configs = [
            {"name": "Basic", "kwargs": {}},
            {"name": "Detached", "kwargs": {"creationflags": getattr(subprocess, 'DETACHED_PROCESS', 0)}},
            {"name": "New Console", "kwargs": {"creationflags": getattr(subprocess, 'CREATE_NEW_CONSOLE', 0)}},
            {"name": "Shell=True", "kwargs": {"shell": True}},
        ]
        
        for config in configs:
            try:
                print(f"\nTesting {config['name']}...")
                cmd = [sys.executable, test_script_path, f"popen_{config['name'].lower()}"]
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    **config['kwargs']
                )
                
                print(f"Process started, PID: {process.pid}")
                
                # Wait a bit and check if still running
                time.sleep(2)
                poll_result = process.poll()
                
                if poll_result is None:
                    # Still running, terminate and get output
                    process.terminate()
                    stdout, stderr = process.communicate(timeout=5)
                    print(f"SUCCESS: Process ran successfully")
                    print(f"Output: {stdout.decode()[:200]}...")
                else:
                    # Process ended
                    stdout, stderr = process.communicate(timeout=2)
                    print(f"Process ended with code: {poll_result}")
                    print(f"STDOUT: {stdout.decode()}")
                    print(f"STDERR: {stderr.decode()}")
                    
            except Exception as e:
                print(f"FAILED {config['name']}: {e}")
    
    finally:
        # Clean up test script
        try:
            os.unlink(test_script_path)
            print(f"\nCleaned up test script: {test_script_path}")
        except:
            pass
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_subprocess_from_kicad()
