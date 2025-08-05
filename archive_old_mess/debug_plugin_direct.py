#!/usr/bin/env python3
"""
Direct test of the plugin's start_server function to debug the "Unknown option" errors
"""

import sys
import os
from pathlib import Path
import tempfile

# Add plugin directory to path
plugin_dir = Path(__file__).parent / "addon_package" / "plugins"
sys.path.insert(0, str(plugin_dir))

# Mock pcbnew and wx for testing
class MockPcbnew:
    class ActionPlugin:
        def defaults(self): pass
        def register(self): pass

class MockWx:
    OK = 1
    ICON_ERROR = 2
    def MessageBox(self, *args): pass

# Mock modules
sys.modules['pcbnew'] = MockPcbnew()
sys.modules['wx'] = MockWx()

# Now import the plugin
from __init__ import OrthoRoutePlugin

def test_start_server():
    """Test the start_server function directly"""
    print("=== Direct Plugin Start Server Test ===")
    
    # Create test work directory
    work_dir = Path(tempfile.mkdtemp(prefix="orthoroute_test_"))
    print(f"Test work directory: {work_dir}")
    
    try:
        # Create plugin instance
        plugin = OrthoRoutePlugin()
        
        # Call start_server method directly
        print("Calling start_server...")
        result = plugin.start_server(work_dir)
        
        if result:
            print(f"✅ start_server SUCCESS! Process PID: {result.pid}")
            
            # Let it run for a few seconds
            import time
            time.sleep(3)
            
            # Check if still running
            poll_result = result.poll()
            if poll_result is None:
                print("✅ Server still running")
                
                # Stop it
                print("Stopping server...")
                plugin.stop_server(work_dir, result)
            else:
                print(f"❌ Server died with code: {poll_result}")
        else:
            print("❌ start_server FAILED!")
            
    except Exception as e:
        print(f"❌ Test exception: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            import shutil
            shutil.rmtree(work_dir)
            print("🧹 Cleanup complete")
        except:
            pass

if __name__ == "__main__":
    test_start_server()
