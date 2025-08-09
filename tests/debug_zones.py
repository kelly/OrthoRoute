#!/usr/bin/env python3
"""Debug zone extraction to understand the structure"""

from kicad_interface import KiCadInterface

def debug_zones():
    ki = KiCadInterface()
    if not ki.connect():
        print("Failed to connect to KiCad")
        return
    
    try:
        board = ki.board
        zones = board.get_zones()
        print(f"Found {len(zones)} zones")
        
        for i, zone in enumerate(zones):
            print(f"\n=== Zone {i} ===")
            print(f"Zone object: {zone}")
            print(f"Zone type: {type(zone)}")
            
            # Check basic properties
            zone_net = getattr(zone, 'net', None)
            zone_layer = getattr(zone, 'layer', 'Unknown')
            print(f"Net: {zone_net}")
            print(f"Layer: {zone_layer}")
            
            # Check different ways to get filled polygons
            print(f"Has 'filled_polygons': {hasattr(zone, 'filled_polygons')}")
            if hasattr(zone, 'filled_polygons'):
                filled_polys = getattr(zone, 'filled_polygons', None)
                print(f"filled_polygons type: {type(filled_polys)}")
                print(f"filled_polygons value: {filled_polys}")
                
            print(f"Has 'GetFilledPolysList': {hasattr(zone, 'GetFilledPolysList')}")
            print(f"Has 'outline': {hasattr(zone, 'outline')}")
            
            if hasattr(zone, 'outline'):
                outline = getattr(zone, 'outline', None)
                print(f"Outline type: {type(outline)}")
                
            # List all attributes
            attrs = [attr for attr in dir(zone) if not attr.startswith('_')]
            print(f"Zone attributes: {attrs}")
            
            if i >= 1:  # Only check first 2 zones
                break
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_zones()
