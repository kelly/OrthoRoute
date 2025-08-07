#!/usr/bin/env python3
"""
ULTRA MINIMAL KiCad IPC Plugin - Just Track Creation

This follows the exact pattern from the official KiCad IPC API documentation.
The key insight: board.create_items() is the ONLY way to add items to KiCad.
"""

def main():
    """Ultra minimal track creation"""
    try:
        # Official imports
        from kipy import KiCad
        from kipy.board_types import Track
        from kipy.geometry import Vector2
        from kipy.util.units import from_mm
        
        # Connect 
        kicad = KiCad()
        board = kicad.get_board()  # CORRECT API CALL
        
        # Create track
        track = Track()
        track.start = Vector2(from_mm(10), from_mm(10))
        track.end = Vector2(from_mm(30), from_mm(10))
        track.width = from_mm(0.25)
        track.layer = 0  # F.Cu
        
        # Get a net (required)
        nets = board.get_nets()
        if nets:
            track.net = nets[0]
        
        # THE CRITICAL LINE - this is what actually adds the track
        board.create_items([track])
        
        print("Track created successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
