#!/usr/bin/env python3
"""
Ultra Simple Test - Write immediately on script start
"""

# Write to file IMMEDIATELY - before any imports or complex operations
import os

log_file = os.path.join(os.path.expanduser("~"), "Documents", "kicad_button_test.txt")

# Write basic success message first
with open(log_file, "w") as f:
    f.write("PLUGIN START - Script is running!\n")
    f.flush()

# Now try to add more info
try:
    import sys
    import time
    
    with open(log_file, "a") as f:  # Append mode
        f.write(f"SUCCESS - Python script executed by KiCad!\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Python: {sys.executable}\n")
        f.write(f"Args: {sys.argv}\n")
        f.write(f"Working dir: {os.getcwd()}\n")
        f.flush()
        
    print("Plugin executed successfully!")
    
except Exception as e:
    with open(log_file, "a") as f:
        f.write(f"ERROR: {e}\n")
        f.flush()

print("Script finished!")
