#!/usr/bin/env python3

# Absolute minimal test - no functions, just direct execution
import os

# Write immediately
log_file = os.path.join(os.path.expanduser("~"), "Documents", "kicad_button_test.txt")

try:
    with open(log_file, "w") as f:
        f.write("MINIMAL TEST - Python script executed!\n")
        f.write("This proves KiCad can run Python scripts\n")
        f.flush()
    
    # Try to add more info if possible
    import sys
    with open(log_file, "a") as f:
        f.write(f"Python: {sys.executable}\n")
        f.write(f"Script: {__file__}\n")
        f.write(f"Args: {sys.argv}\n")
        f.flush()
        
except Exception as e:
    # Fallback - create a different file if main one fails
    try:
        with open(os.path.join(os.path.expanduser("~"), "Documents", "kicad_error.txt"), "w") as f:
            f.write(f"ERROR: {e}\n")
    except:
        pass
