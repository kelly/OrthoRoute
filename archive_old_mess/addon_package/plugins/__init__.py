#!/usr/bin/env python3
"""
OrthoRoute KiCad Plugin - IPC API Only
This plugin uses KiCad 9.0+ IPC API exclusively.
No SWIG bindings are used.
"""

# IPC plugins don't use traditional ActionPlugin registration
# They are discovered and run by KiCad based on plugin.json configuration
print("OrthoRoute IPC Plugin package loaded")

# No SWIG registration - IPC plugins work differently
