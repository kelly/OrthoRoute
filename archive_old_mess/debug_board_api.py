#!/usr/bin/env python3
"""
Debug IPC Board API - Check what methods are available
"""

import time

def log_message(message):
    """Log messages with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

try:
    from kipy import KiCad
    from kipy.util.units import to_mm, from_mm
    from kipy.geometry import Vector2
    
    log_message("🔌 Connecting to KiCad...")
    kicad = KiCad()
    
    log_message("📋 Getting active board...")
    board = kicad.get_board()
    
    if board:
        log_message("✅ Board retrieved successfully")
        log_message(f"Board type: {type(board)}")
        log_message("Available methods and attributes:")
        
        methods = [attr for attr in dir(board) if not attr.startswith('_')]
        for method in sorted(methods):
            log_message(f"  - {method}")
            
        # Try to get basic board info
        try:
            log_message(f"Board name: {board.name if hasattr(board, 'name') else 'No name attribute'}")
        except:
            pass
            
        try:
            if hasattr(board, 'get_tracks'):
                tracks = board.get_tracks()
                log_message(f"Number of tracks: {len(tracks) if tracks else 0}")
        except Exception as e:
            log_message(f"Error getting tracks: {e}")
            
    else:
        log_message("❌ No board retrieved")
        
except Exception as e:
    log_message(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
