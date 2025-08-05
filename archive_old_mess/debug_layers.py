#!/usr/bin/env python3
"""
Debug layer specification
"""

import time

def log_message(message):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

try:
    from kipy import KiCad
    from kipy.board_types import Track
    from kipy.geometry import Vector2
    from kipy.util.units import from_mm
    
    log_message("🔌 Connecting to KiCad...")
    kicad = KiCad()
    board = kicad.get_board()
    
    if board:
        # Get visible layers to see what's available
        try:
            visible_layers = board.get_visible_layers()
            log_message(f"✅ Visible layers: {visible_layers}")
        except Exception as e:
            log_message(f"❌ Could not get visible layers: {e}")
        
        # Try to get stackup info
        try:
            stackup = board.get_stackup()
            log_message(f"✅ Stackup available: {type(stackup)}")
            if hasattr(stackup, 'layers'):
                log_message(f"Stackup layers: {stackup.layers}")
        except Exception as e:
            log_message(f"❌ Could not get stackup: {e}")
            
        # Try different layer values
        track = Track()
        
        layer_attempts = [0, 1, 'F_Cu', 'B_Cu', 'In1_Cu', 31, 30]
        
        for layer_val in layer_attempts:
            try:
                track.layer = layer_val
                log_message(f"✅ Layer {layer_val} accepted")
                break
            except Exception as e:
                log_message(f"❌ Layer {layer_val} failed: {e}")
        
except Exception as e:
    log_message(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
