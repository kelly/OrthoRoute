"""
Ultra-simple OrthoRoute plugin for testing visibility
"""
import pcbnew
import wx

class OrthoRouteSimplePlugin(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "🟢 ORTHOROUTE TEST - CLICK ME"
        self.category = "Test"
        self.description = "Simple test version of OrthoRoute"
        self.show_toolbar_button = True
    
    def Run(self):
        wx.MessageBox("OrthoRoute plugin is working!\n\nThis confirms the plugin system is detecting OrthoRoute correctly.", 
                     "OrthoRoute Test", wx.OK | wx.ICON_INFORMATION)

# Register the plugin
OrthoRouteSimplePlugin().register()
