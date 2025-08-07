#!/usr/bin/env python3
"""
Ultra Simple Test - IPC Plugin for PCM Distribution
PCM package version that uses IPC runtime without SWIG dependencies
"""

import time
import os
import sys
import json
import traceback

# Log to a file so we can see if KiCad is even trying to load this
log_file = os.path.join(os.path.expanduser("~"), "kicad_pcm_ipc_plugin_test.log")

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
    """Main entry point for PCM IPC plugin - called directly by KiCad"""
    log_message("🚀 Ultra Simple Test PCM IPC plugin started")
    log_message(f"Working directory: {os.getcwd()}")
    log_message(f"Python executable: {sys.executable}")
    log_message(f"Command line args: {sys.argv}")
    
    try:
        # Import KiCad IPC API
        from kipy import KiCad
        log_message("✅ KiCad IPC API (kipy) imported successfully")
        
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
        log_message("✅ PCM IPC Plugin execution completed successfully")
        print("SUCCESS: Ultra Simple Test PCM IPC plugin executed!")
        print(f"Check log file: {log_file}")
        
        return 0
        
    except ImportError as e:
        error_msg = f"❌ KiCad IPC API (kipy) not available: {e}"
        log_message(error_msg)
        print(error_msg)
        print("Install with: pip install kicad-python")
        return 1
        
    except Exception as e:
        error_msg = f"❌ ERROR in ultra_simple_test.py: {e}"
        log_message(error_msg)
        log_message(f"Traceback: {traceback.format_exc()}")
        print(error_msg)
        return 1

if __name__ == "__main__":
    sys.exit(main())
