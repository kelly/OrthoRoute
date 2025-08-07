#!/usr/bin/env python3
"""
Ultra Simple Test - Just log when loaded
"""

import time
import os

# Log to a file so we can see if KiCad is even trying to load this
log_file = os.path.join(os.path.expanduser("~"), "kicad_plugin_test.log")

try:
    with open(log_file, "a") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] ultra_simple_test.py loaded\n")
        f.flush()
        
    print("Ultra simple test loaded successfully")
    
    # Try to do something minimal
    result = 2 + 2
    
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] Basic math works: 2+2={result}\n")
        f.flush()

except Exception as e:
    try:
        with open(log_file, "a") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] ERROR in ultra_simple_test.py: {e}\n")
            f.flush()
    except:
        pass
