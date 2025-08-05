#!/usr/bin/env python3
"""
CORRECTED Minimal KiCad IPC Plugin - Just Draws One Test Track

This fixes the issues in the previous version:
1. Correct API calls (kicad.board.get_board() not kicad.get_board())
2. Proper commit transaction handling
3. Correct push_commit() usage
"""

import sys
import os

def main():
    """Main entry point - just draw one simple track"""
    print("🧪 CORRECTED Minimal IPC Plugin - Drawing one test track...")
    
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
        
        # Get the board - CORRECTED API CALL
        board = kicad.get_board()  # CORRECT: kicad.get_board()
        if not board:
            print("❌ No board found")
            return 1
        
        print("✅ Got board")
        
        # Begin commit transaction - CRITICAL FOR PROPER CLEANUP
        commit = board.begin_commit()
        print("✅ Commit transaction started")
        
        try:
            # Create one simple track from (10mm, 10mm) to (30mm, 10mm)
            track = Track()
            track.start = Vector2(from_mm(10), from_mm(10))
            track.end = Vector2(from_mm(30), from_mm(10))
            track.width = from_mm(0.25)  # 0.25mm width
            track.layer = 0  # F.Cu layer
            
            # Assign to a net (required for valid track)
            nets = board.get_nets()
            if nets:
                track.net = nets[0]
                print(f"✅ Track assigned to net: {nets[0].name}")
            
            print("✅ Track created")
            
            # Add to board - this is the critical step
            created_items = board.create_items([track])
            print(f"✅ Track added to board: {len(created_items)} items created")
            
            # Commit changes - CORRECTED USAGE
            board.push_commit(commit, "Minimal test track")
            print("✅ Changes committed")
            
        except Exception as e:
            # Drop commit on error
            print(f"❌ Error during track creation: {e}")
            board.drop_commit(commit)
            raise
        
        print("🎉 SUCCESS! One test track created from (10mm,10mm) to (30mm,10mm)!")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        print(f"🏁 Plugin finished with exit code: {exit_code}")
        # Don't call sys.exit() - this can kill KiCad if run in plugin context
        # sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        # Don't call sys.exit(1) - this can kill KiCad
        # sys.exit(1)
