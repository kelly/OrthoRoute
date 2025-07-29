"""
Simple debug plugin to test KiCad plugin loading
"""

import os
import sys

# Write debug info to a file so we can see what's happening
debug_file = os.path.join(os.path.dirname(__file__), "debug_output.txt")

with open(debug_file, "w") as f:
    f.write("OrthoRoute Plugin Debug Info\n")
    f.write("=" * 30 + "\n")
    f.write(f"Python version: {sys.version}\n")
    f.write(f"Python path: {sys.path}\n")
    f.write(f"Plugin directory: {os.path.dirname(__file__)}\n")
    f.write(f"Available modules: {list(sys.modules.keys())}\n")
    
    # Test if we can import required modules
    try:
        import pcbnew
        f.write("✓ pcbnew imported successfully\n")
        f.write(f"pcbnew version: {getattr(pcbnew, 'Version', 'Unknown')}\n")
    except Exception as e:
        f.write(f"✗ pcbnew import failed: {e}\n")
    
    try:
        import wx
        f.write("✓ wx imported successfully\n")
        f.write(f"wx version: {wx.version()}\n")
    except Exception as e:
        f.write(f"✗ wx import failed: {e}\n")
    
    # Test importing our plugin
    try:
        from orthoroute_kicad import OrthoRouteKiCadPlugin
        f.write("✓ OrthoRouteKiCadPlugin imported successfully\n")
        
        # Try to create an instance
        plugin = OrthoRouteKiCadPlugin()
        f.write("✓ Plugin instance created successfully\n")
        f.write(f"Plugin name: {plugin.GetName()}\n")
        f.write(f"Plugin description: {plugin.GetDescription()}\n")
        
    except Exception as e:
        f.write(f"✗ Plugin import/creation failed: {e}\n")
        import traceback
        f.write(f"Traceback: {traceback.format_exc()}\n")

# Simple KiCad plugin class
try:
    import pcbnew
    
    class DebugPlugin(pcbnew.ActionPlugin):
        def defaults(self):
            self.name = "Debug Plugin Test"
            self.category = "Debug"
            self.description = "Simple debug plugin to test loading"
            
        def Run(self):
            with open(debug_file, "a") as f:
                f.write("Debug plugin Run() method called!\n")
    
    # Register the debug plugin
    DebugPlugin().register()
    
    with open(debug_file, "a") as f:
        f.write("✓ Debug plugin registered successfully\n")
        
except Exception as e:
    with open(debug_file, "a") as f:
        f.write(f"✗ Debug plugin registration failed: {e}\n")
