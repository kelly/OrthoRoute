#!/usr/bin/env python3
"""
Ultra Simple Test - True IPC Plugin for KiCad 9.0+
No ActionPlugin registration - purely IPC-based toolbar integration
"""

import time
import os
import sys
import json
import traceback

# Log to a file so we can see if KiCad is even trying to load this
log_file = os.path.join(os.path.expanduser("~"), "kicad_plugin_test.log")

def log_message(message):
    """Helper function to log messages"""
    try:
        with open(log_file, "a") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
            f.flush()
    except Exception as e:
        print(f"Logging error: {e}")

def main():
    """Main entry point for IPC plugin - called directly by KiCad"""
    log_message("🚀 Ultra Simple Test IPC plugin started")
    log_message(f"Working directory: {os.getcwd()}")
    log_message(f"Python executable: {sys.executable}")
    log_message(f"Command line args: {sys.argv}")
    
    try:
        # Log environment variables passed by KiCad
        for key, value in os.environ.items():
            if 'KICAD' in key.upper():
                log_message(f"ENV: {key} = {value}")
        
        # Import KiCad IPC API - try both import methods
        try:
            from kicad import Client as KiCad
            log_message("✅ KiCad IPC API imported via 'kicad' module")
        except ImportError:
            try:
                from kipy import KiCad
                log_message("✅ KiCad IPC API imported via 'kipy' module") 
            except ImportError:
                log_message("❌ Could not import KiCad IPC API")
                raise
        
        # Connect to KiCad
        kicad = KiCad()
        log_message("✅ Connected to KiCad via IPC")
        
        # Get board information
        try:
            board = kicad.get_board()
            if board:
                log_message(f"✅ Board loaded: {board.filename}")
            else:
                log_message("⚠️ No board currently loaded")
        except Exception as e:
            log_message(f"⚠️ Could not get board info: {e}")
        
        # Success - plugin executed
        log_message("✅ Plugin execution completed successfully")
        print("SUCCESS: Ultra Simple Test IPC plugin executed!")
        print(f"Check log file: {log_file}")
        
        return 0
        
    except ImportError as e:
        error_msg = f"❌ KiCad IPC API not available: {e}"
        log_message(error_msg)
        log_message("This suggests either:")
        log_message("1. kicad-python is not installed in the environment")
        log_message("2. KiCad IPC API server is not enabled")
        log_message("3. Plugin is not being launched by KiCad properly")
        print(error_msg)
        print("Check the log file for more details:", log_file)
        return 1
        
    except Exception as e:
        error_msg = f"❌ ERROR in ultra_simple_test.py: {e}"
        log_message(error_msg)
        log_message(f"Traceback: {traceback.format_exc()}")
        print(error_msg)
        return 1

if __name__ == "__main__":
    sys.exit(main())
