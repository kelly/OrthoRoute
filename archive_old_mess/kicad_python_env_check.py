#!/usr/bin/env python3
"""
KiCad Python Environment Check Plugin
"""

import pcbnew
import wx
import sys
import os
import subprocess

class PythonEnvCheckPlugin(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "Python Environment Check"
        self.category = "Development"
        self.description = "Check KiCad's Python environment for package installation"
        
    def Run(self):
        output = []
        output.append("=" * 60)
        output.append("KICAD PYTHON ENVIRONMENT DIAGNOSTIC")
        output.append("=" * 60)
        output.append(f"Python executable: {sys.executable}")
        output.append(f"Python version: {sys.version}")
        output.append("Python path:")
        for path in sys.path:
            output.append(f"  {path}")
        
        output.append("\n" + "=" * 60)
        output.append("TRYING TO IMPORT KIPY")
        output.append("=" * 60)
        
        try:
            import kipy
            output.append(f"✅ kipy successfully imported from: {kipy.__file__}")
            output.append(f"   kipy version: {getattr(kipy, '__version__', 'unknown')}")
        except ImportError as e:
            output.append(f"❌ kipy import failed: {e}")
        
        output.append("\n" + "=" * 60)
        output.append("INSTALLATION COMMAND FOR THIS ENVIRONMENT")
        output.append("=" * 60)
        output.append(f"To install kicad-python in this Python environment, run:")
        output.append(f"{sys.executable} -m pip install kicad-python")
        
        # Show results in dialog
        result_text = "\n".join(output)
        dialog = wx.MessageDialog(None, result_text, "Python Environment Check", wx.OK)
        dialog.ShowModal()
        dialog.Destroy()
        
        # Also print to console
        print(result_text)

PythonEnvCheckPlugin().register()
