#!/usr/bin/env python3
"""
Test the actual subprocess launching to isolate the issue
"""

import os
import sys
import subprocess
import tempfile
import time
from pathlib import Path

def test_server_launch():
    """Test launching the server the same way the plugin does"""
    
    print("🧪 Testing standalone server launch...")
    
    # Create temporary work directory
    work_dir = Path(tempfile.mkdtemp(prefix="orthoroute_debug_"))
    print(f"📁 Work directory: {work_dir}")
    
    try:
        # Find server script (same logic as plugin)
        plugin_dir = Path(__file__).parent / "addon_package" / "plugins"
        server_script = plugin_dir / "orthoroute_standalone_server.py"
        
        if not server_script.exists():
            print(f"❌ Server script not found: {server_script}")
            return False
        
        print(f"✅ Server script found: {server_script}")
        
        # Test the exact command that would be used
        python_exe = sys.executable
        cmd = [
            str(python_exe),
            str(server_script.absolute()),
            "--work-dir", str(work_dir.absolute())
        ]
        
        print(f"🚀 Command to test: {cmd}")
        print(f"📂 Python executable: {python_exe}")
        print(f"📁 Server script (absolute): {server_script.absolute()}")
        print(f"💼 Work directory (absolute): {work_dir.absolute()}")
        
        # Test if the command works manually first
        print("\n🔬 Testing manual execution...")
        try:
            result = subprocess.run(
                cmd + ["--timeout", "5"],  # Short timeout for test
                capture_output=True,
                text=True,
                timeout=10
            )
            print(f"📤 Return code: {result.returncode}")
            print(f"📤 STDOUT: {result.stdout}")
            print(f"📤 STDERR: {result.stderr}")
            
            if result.returncode == 0:
                print("✅ Manual execution successful!")
            else:
                print("❌ Manual execution failed!")
                return False
                
        except subprocess.TimeoutExpired:
            print("✅ Process timed out as expected (server was running)")
        except Exception as e:
            print(f"❌ Manual execution error: {e}")
            return False
        
        # Now test subprocess.Popen (like the plugin uses)
        print("\n🔬 Testing subprocess.Popen...")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            cwd=str(work_dir.absolute()),
            env=dict(os.environ),
            shell=False
        )
        
        print(f"✅ Process started with PID: {process.pid}")
        
        # Wait and check
        time.sleep(2)
        poll_result = process.poll()
        
        if poll_result is not None:
            stdout, stderr = process.communicate()
            print(f"❌ Process terminated with code: {poll_result}")
            print(f"📤 STDOUT: {stdout.decode('utf-8', errors='ignore')}")
            print(f"📤 STDERR: {stderr.decode('utf-8', errors='ignore')}")
            return False
        else:
            print("✅ Process is running!")
            
            # Check for status file
            status_file = work_dir / "routing_status.json"
            for i in range(50):  # Wait up to 5 seconds
                if status_file.exists():
                    print("✅ Status file created - server communication working!")
                    break
                time.sleep(0.1)
            else:
                print("⚠ No status file after 5 seconds")
            
            # Terminate test process
            process.terminate()
            try:
                process.wait(timeout=2)
                print("✅ Process terminated cleanly")
            except:
                process.kill()
                print("⚠ Process killed")
            
            return True
    
    finally:
        # Cleanup
        try:
            import shutil
            shutil.rmtree(work_dir)
            print("🧹 Cleaned up work directory")
        except Exception as e:
            print(f"⚠ Cleanup error: {e}")

if __name__ == "__main__":
    success = test_server_launch()
    if success:
        print("\n🎉 Server launch test PASSED!")
        print("   The subprocess launching should work in KiCad")
    else:
        print("\n💥 Server launch test FAILED!")
        print("   There's an issue with the server script or subprocess setup")
