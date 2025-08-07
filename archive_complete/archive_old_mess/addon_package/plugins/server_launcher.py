#!/usr/bin/env python3
"""
Simple server launcher for OrthoRoute - designed to work around KiCad subprocess issues
This script serves as an intermediary to launch the main server with better compatibility
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: python server_launcher.py <work_dir> [timeout]")
        print("❌ ERROR: Insufficient arguments")
        return 1  # Return error code instead of sys.exit
    
    work_dir = sys.argv[1]
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
    
    # Find the main server script
    launcher_dir = Path(__file__).parent
    server_script = launcher_dir / "orthoroute_standalone_server.py"
    
    if not server_script.exists():
        print(f"ERROR: Server script not found: {server_script}")
        return 1  # Return error code instead of sys.exit
    
    print(f"LAUNCHER: Starting OrthoRoute server")
    print(f"LAUNCHER: Work directory: {work_dir}")
    print(f"LAUNCHER: Server script: {server_script}")
    print(f"LAUNCHER: Timeout: {timeout}")
    
    # Launch the main server
    cmd = [
        sys.executable,
        str(server_script),
        "--work-dir", str(work_dir),
        "--timeout", str(timeout)
    ]
    
    print(f"LAUNCHER: Command: {' '.join(cmd)}")
    
    try:
        # Use os.execv to replace this process entirely
        # This avoids any subprocess complications
        os.execv(sys.executable, cmd)
        
    except Exception as e:
        print(f"LAUNCHER ERROR: Failed to launch server: {e}")
        
        # Fallback: Try subprocess approach
        try:
            process = subprocess.Popen(cmd, shell=False)
            print(f"LAUNCHER: Fallback subprocess started, PID: {process.pid}")
            process.wait()  # Wait for completion
            
        except Exception as e2:
            print(f"LAUNCHER ERROR: Fallback also failed: {e2}")
            return 1  # Return error code instead of sys.exit
    
    return 0  # Success

if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        print(f"❌ Server launcher failed with code: {exit_code}")
    # Don't call sys.exit() in case this is run within KiCad context
