#!/usr/bin/env python3
"""
Debug IPC track creation - Find the correct way to create tracks
"""

import time

def log_message(message):
    """Log messages with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

try:
    from kipy import KiCad
    from kipy.board_types import Track
    from kipy.util.units import to_mm, from_mm
    from kipy.geometry import Vector2
    
    log_message("🔌 Connecting to KiCad...")
    kicad = KiCad()
    
    log_message("📋 Getting active board...")
    board = kicad.get_board()
    
    if board:
        log_message("✅ Board retrieved successfully")
        
        # Check what Track object looks like
        log_message("🔍 Investigating Track object...")
        track = Track()
        log_message(f"Track type: {type(track)}")
        log_message(f"Track attributes: {[attr for attr in dir(track) if not attr.startswith('_')]}")
        
        # Try to set track properties correctly
        try:
            track.start = Vector2(from_mm(10), from_mm(10))
            track.end = Vector2(from_mm(30), from_mm(10))
            track.width = from_mm(0.2)
            track.layer = 'F.Cu'
            
            log_message("✅ Track properties set successfully")
            log_message(f"Track has proto attribute: {hasattr(track, 'proto')}")
            
            if hasattr(track, 'proto'):
                log_message("✅ Track has proto attribute - can be used with create_items")
                
                # Try creating the track
                board.begin_commit()
                result = board.create_items([track])
                board.push_commit()
                log_message("✅ Track created successfully!")
                
            else:
                log_message("❌ Track doesn't have proto attribute")
                
        except Exception as e:
            log_message(f"❌ Error creating track: {e}")
            import traceback
            traceback.print_exc()
            try:
                board.drop_commit()
            except:
                pass
            
    else:
        log_message("❌ No board retrieved")
        
except Exception as e:
    log_message(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
