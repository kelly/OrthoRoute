"""
Simple test plugin to verify KiCad plugin system is working
"""
import pcbnew
import wx

class TestPlugin(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "TEST PLUGIN - DELETE ME"
        self.category = "Test"
        self.description = "Simple test plugin to verify plugin system works"
        self.show_toolbar_button = True
    
    def Run(self):
        wx.MessageBox("Test plugin is working!", "Test Plugin", wx.OK | wx.ICON_INFORMATION)

# Register the plugin
TestPlugin().register()
