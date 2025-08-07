#!/usr/bin/env python3
"""
Minimal Track Plugin - KiCad 9.0+ IPC API
Creates one test track to verify basic IPC API functionality
"""

import os
import sys
import time
import traceback

def log_message(message):
    """Log messages with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

# Import KiCad IPC API
try:
    from kipy import KiCad
    from kipy.board_types import Track
    from kipy.util.units import to_mm, from_mm
    from kipy.geometry import Vector2
    IPC_AVAILABLE = True
    log_message("✅ KiCad IPC API (kipy) imported successfully")
except ImportError as e:
    log_message(f"❌ Failed to import KiCad IPC API: {e}")
    log_message("Please install: pip install kicad-python")
    IPC_AVAILABLE = False

class MinimalTrackPlugin:
    """Minimal Track Plugin Class - KiCad 9.0+ Style"""
    
    def __init__(self):
        """Initialize the plugin"""
        self.kicad = None
        self.board = None

    def run(self):
        """Main plugin execution method - IPC Plugin Pattern"""
        log_message("🚀 Minimal Track Plugin Starting...")
        
        if not IPC_AVAILABLE:
            log_message("❌ KiCad IPC API not available")
            return 1
            
        try:
            # Connect to KiCad
            log_message("🔌 Connecting to KiCad...")
            self.kicad = KiCad()
            
            # Get the active board
            log_message("📋 Getting active board...")
            self.board = self.kicad.get_board()
            if not self.board:
                log_message("❌ No active board found")
                return 1
                
            # Create minimal test track
            log_message("✏️  Creating minimal test track...")
            
            track = Track()
            track.start = Vector2(from_mm(5), from_mm(5))    # 5mm, 5mm
            track.end = Vector2(from_mm(15), from_mm(5))     # 15mm, 5mm
            track.width = from_mm(0.25)                      # 0.25mm width
            track.layer = 0                                  # Top layer
            
            # Add track to board
            self.board.add_track(track)
            
            # Refresh board
            self.board.refresh()
            
            log_message("✅ Minimal test track created successfully!")
            log_message("🎯 Minimal Track Plugin completed successfully!")
            
            return 0
            
        except Exception as e:
            log_message(f"❌ Error in Minimal Track Plugin: {e}")
            traceback.print_exc()
            return 1

# IPC Plugin Pattern - Direct instantiation and execution
if __name__ == "__main__":
    try:
        plugin = MinimalTrackPlugin()
        exit_code = plugin.run()
        log_message(f"🏁 Plugin finished with exit code: {exit_code}")
        # Don't call sys.exit() in plugin context - this kills KiCad
        # sys.exit(exit_code)
    except Exception as e:
        log_message(f"❌ Fatal error: {e}")
        traceback.print_exc()
        # Don't call sys.exit(1) - this kills KiCad
        # sys.exit(1)
