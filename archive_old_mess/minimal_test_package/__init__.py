#!/usr/bin/env python3
"""
KiCad Plugin Entry Point for Minimal Track Test
This is the proper plugin structure for KiCad external plugins
"""

import wx
import pcbnew

class MinimalTrackTestPlugin(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "Minimal Track Test"
        self.category = "Testing"
        self.description = "Draw one simple test track using IPC API"
        self.show_toolbar_button = True
        self.icon_file_name = ""

    def Run(self):
        """Execute the minimal track test"""
        try:
            # Import the actual test function
            from minimal_track_test import main
            main()
            wx.MessageBox("Minimal track test completed successfully!", "Success", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            wx.MessageBox(f"Error running minimal track test: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)

# Register the plugin
MinimalTrackTestPlugin().register()
