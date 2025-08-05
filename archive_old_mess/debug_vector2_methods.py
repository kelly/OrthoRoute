#!/usr/bin/env python3
"""
Test Vector2.from_xy_mm method
"""

import time

def log_message(message):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

try:
    from kipy.geometry import Vector2
    
    # Test from_xy_mm class method
    try:
        v1 = Vector2.from_xy_mm(10, 10)
        log_message(f"✅ Vector2.from_xy_mm(10, 10) works: {v1}")
    except Exception as e:
        log_message(f"❌ Vector2.from_xy_mm failed: {e}")
        
    # Test setting x,y directly
    try:
        from kipy.util.units import from_mm
        v2 = Vector2()
        v2.x = from_mm(10)
        v2.y = from_mm(10)
        log_message(f"✅ Vector2 with from_mm: {v2}")
    except Exception as e:
        log_message(f"❌ Vector2 from_mm failed: {e}")
        
except Exception as e:
    log_message(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
