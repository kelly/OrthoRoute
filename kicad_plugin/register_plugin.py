import os
import sys
import pcbnew
import wx

# Add the plugin directory to the Python path
plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

try:
    # Try different import approaches
    try:
        from orthoroute_kicad import OrthoRouteKiCadPlugin
    except ImportError:
        # If relative import fails, try absolute
        import sys
        import os
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        if plugin_dir not in sys.path:
            sys.path.insert(0, plugin_dir)
        from orthoroute_kicad import OrthoRouteKiCadPlugin
    
    class OrthoRoutePluginLoader(pcbnew.ActionPlugin):
        def defaults(self):
            self.name = "OrthoRoute GPU Autorouter"
            self.category = "Autorouter"
            self.description = "GPU-accelerated PCB autorouting using CuPy"
            self.show_toolbar_button = True
            
            # Set icon path - try different approaches for better compatibility
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(plugin_dir, "icon.png")
            
            if os.path.exists(icon_path):
                self.icon_file_name = icon_path
                print(f"DEBUG: Icon found at: {icon_path}")
            else:
                print(f"DEBUG: Icon not found at: {icon_path}")
                # Fallback - just use the filename
                self.icon_file_name = "icon.png"
        
        def Run(self):
            try:
                plugin = OrthoRouteKiCadPlugin()
                plugin.Run()
            except Exception as e:
                wx.MessageBox(f"Error running OrthoRoute: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)

    # Register the plugin
    OrthoRoutePluginLoader().register()
    
except Exception as e:
    import traceback
    wx.MessageBox(f"Failed to load OrthoRoute plugin: {str(e)}\n\n{traceback.format_exc()}", "Error", wx.OK | wx.ICON_ERROR)
