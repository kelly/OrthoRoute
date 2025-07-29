"""
Minimal OrthoRoute Plugin for KiCad
"""

import os
import sys

# Add current directory to path
plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

try:
    import pcbnew
    import wx
    
    class MinimalOrthoRoutePlugin(pcbnew.ActionPlugin):
        def defaults(self):
            self.name = "OrthoRoute GPU Autorouter"
            self.category = "Route"
            self.description = "GPU-accelerated PCB autorouter using CuPy"
            # Try to set icon path
            try:
                icon_path = os.path.join(plugin_dir, "icon.png")
                if os.path.exists(icon_path):
                    self.icon_file_name = icon_path
                    self.show_toolbar_button = True
                else:
                    self.show_toolbar_button = True
            except:
                self.show_toolbar_button = True
        
        def Run(self):
            wx.MessageBox(
                "OrthoRoute GPU Autorouter is working!\n\nThis is a test message to confirm the plugin is properly loaded.",
                "OrthoRoute Test",
                wx.OK | wx.ICON_INFORMATION
            )
    
    # Register the plugin
    plugin_instance = MinimalOrthoRoutePlugin()
    plugin_instance.register()
    
    # Write success to debug file
    debug_file = os.path.join(plugin_dir, "minimal_plugin_debug.txt")
    with open(debug_file, "w") as f:
        f.write("Minimal OrthoRoute plugin loaded successfully!\n")
        f.write(f"Plugin name: {plugin_instance.name}\n")
        f.write(f"Plugin category: {plugin_instance.category}\n")
        f.write(f"Plugin directory: {plugin_dir}\n")
        f.write(f"Python version: {sys.version}\n")
    
    print("Minimal OrthoRoute plugin registered successfully!")
    
except Exception as e:
    # Write error to debug file
    debug_file = os.path.join(plugin_dir, "minimal_plugin_debug.txt")
    try:
        with open(debug_file, "w") as f:
            f.write(f"Error loading minimal plugin: {e}\n")
            import traceback
            f.write(f"Traceback: {traceback.format_exc()}\n")
    except:
        pass
    print(f"Error loading minimal plugin: {e}")
