"""
Simple test for KiCad IPC plugin button
"""

import sys
import os
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=r'C:\Users\Benchoff\Documents\kicad_ipc_debug.log'
)
logger = logging.getLogger(__name__)

def main():
    """Main plugin entry point"""
    logger.info("🚀 OrthoRoute GPU button clicked!")
    print("🚀 OrthoRoute GPU button clicked!")
    
    try:
        # Show a simple message
        import tkinter as tk
        from tkinter import messagebox
        
        # Create a simple popup
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        messagebox.showinfo(
            "OrthoRoute GPU", 
            "🎉 SUCCESS!\n\nThe IPC plugin button is working!\n\n"
            "✅ Button click detected\n"
            "✅ Python code executed\n"
            "✅ IPC plugin system functional\n\n"
            "Next: Implement actual routing logic!"
        )
        
        logger.info("✅ Message box displayed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error showing message: {e}")
        
        # Fallback - create a simple file
        try:
            with open(r'C:\Users\Benchoff\Documents\kicad_plugin_test.txt', 'w') as f:
                f.write(f"OrthoRoute GPU plugin executed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Button click successful!\n")
                f.write(f"Python version: {sys.version}\n")
                f.write(f"Working directory: {os.getcwd()}\n")
            
            logger.info("✅ Test file created successfully")
            
        except Exception as e2:
            logger.error(f"❌ Error creating test file: {e2}")
    
    logger.info("🎯 Plugin execution completed")

if __name__ == "__main__":
    main()
