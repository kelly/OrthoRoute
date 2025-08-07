#!/usr/bin/env python3
"""
Minimal KiCad IPC Plugin - Just Draws One Test Track
Tests basic IPC API functionality without any complexity
"""

import sys
import os

def main():
    """Main entry point - just draw one simple track"""
    print("🧪 Minimal IPC Plugin - Drawing one test track...")
    
    try:
        # Import KiCad IPC API
        from kipy import KiCad
        from kipy.board_types import Track
        from kipy.util.units import from_mm
        from kipy.geometry import Vector2
        print("✅ KiCad IPC API imported")
    except ImportError as e:
        print(f"❌ Failed to import KiCad IPC API: {e}")
        print("Install with: pip install kicad-python")
        return 1
    
    try:
        # Connect to KiCad
        kicad = KiCad()
        print("✅ Connected to KiCad")
        
        # Get the board
        board = kicad.get_board()
        if not board:
            print("❌ No board found")
            return 1
        
        print("✅ Got board")
        
        # Create one simple track from (10mm, 10mm) to (30mm, 10mm)
        track = Track()
        track.start = Vector2(from_mm(10), from_mm(10))
        track.end = Vector2(from_mm(30), from_mm(10))
        track.width = from_mm(0.25)  # 0.25mm width
        track.layer = 0  # F.Cu layer
        
        print("✅ Track created")
        
        # Add to board
        board.create_items([track])
        print("✅ Track added to board")
        
        # Commit changes
        board.push_commit("Minimal test track")
        print("✅ Changes committed")
        
        # Save board
        board.save()
        print("✅ Board saved")
        
        print("🎉 SUCCESS! One test track created!")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
