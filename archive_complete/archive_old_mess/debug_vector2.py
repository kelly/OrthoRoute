#!/usr/bin/env python3
"""
Debug Vector2 usage
"""

import time

def log_message(message):
    """Log messages with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

try:
    from kipy.geometry import Vector2
    from kipy.util.units import from_mm
    
    log_message("🔍 Testing Vector2...")
    
    # Check Vector2 constructor
    log_message(f"Vector2 type: {Vector2}")
    log_message(f"Vector2 attributes: {[attr for attr in dir(Vector2) if not attr.startswith('_')]}")
    
    # Try different ways to create Vector2
    try:
        v1 = Vector2()
        log_message(f"✅ Vector2() works: {v1}")
    except Exception as e:
        log_message(f"❌ Vector2() failed: {e}")
        
    try:
        v2 = Vector2(from_mm(10))
        log_message(f"✅ Vector2(from_mm(10)) works: {v2}")
    except Exception as e:
        log_message(f"❌ Vector2(from_mm(10)) failed: {e}")
        
    # Try setting x and y attributes
    try:
        v3 = Vector2()
        v3.x = from_mm(10)
        v3.y = from_mm(10)
        log_message(f"✅ Vector2 with x,y attributes: {v3}")
    except Exception as e:
        log_message(f"❌ Vector2 x,y attributes failed: {e}")
        
    # Check what attributes Vector2 actually has
    try:
        v4 = Vector2()
        attrs = [attr for attr in dir(v4) if not attr.startswith('_')]
        log_message(f"Vector2 instance attributes: {attrs}")
    except Exception as e:
        log_message(f"❌ Could not create Vector2 instance: {e}")
        
except Exception as e:
    log_message(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
