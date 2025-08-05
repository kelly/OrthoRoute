#!/usr/bin/env python3
"""
Minimal Test Plugin - Direct SWIG ActionPlugin approach
Testing if KiCad 9.0 needs traditional ActionPlugin registration
"""

import os
import sys

def log_message(message):
    print(f"[PLUGIN] {message}")

try:
    import pcbnew
    log_message("pcbnew imported successfully")
    
    class MinimalTestPlugin(pcbnew.ActionPlugin):
        def defaults(self):
            self.name = "Minimal Test Plugin"
            self.category = "Testing"
            self.description = "Minimal test to verify plugin loading"
            self.show_toolbar_button = True
            
        def Run(self):
            log_message("Minimal Test Plugin executed!")
            # Show a simple message
            try:
                import wx
                wx.MessageBox("Minimal Test Plugin is working!", "Success")
            except:
                log_message("Plugin executed but wx not available for dialog")
    
    # Register the plugin directly
    MinimalTestPlugin().register()
    log_message("Minimal Test Plugin registered successfully")
    
except Exception as e:
    log_message(f"Error: {e}")
    import traceback
    traceback.print_exc()
