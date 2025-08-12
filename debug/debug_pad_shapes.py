#!/usr/bin/env python3
"""
Debug pad polygon shapes API
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from kicad_interface import KiCadInterface

def main():
    """Debug pad polygon shapes."""
    print("Debugging pad polygon shapes...")
    
    try:
        # Create KiCad interface
        kicad_interface = KiCadInterface()
        if not kicad_interface.connect():
            print("Could not connect to KiCad")
            return
            
        board = kicad_interface.board
        if not board:
            print("Could not get board from KiCad")
            return

        # Get first pad and test polygon shapes API
        pads = board.get_pads()
        print(f"Found {len(pads)} pads")
        
        if pads:
            # Test pad polygon shapes API
            try:
                front_shapes = board.get_pad_shapes_as_polygons(pads[:1], 3)  # Test with first pad only
                print(f"Got front pad shapes: {len(front_shapes)}")
                
                if front_shapes and front_shapes[0]:
                    polygon = front_shapes[0]
                    print(f"Polygon type: {type(polygon)}")
                    print(f"Polygon attributes: {[attr for attr in dir(polygon) if not attr.startswith('_')]}")
                    
                    outline = polygon.outline
                    print(f"Outline type: {type(outline)}")
                    print(f"Outline length: {len(outline)}")
                    
                    if outline:
                        first_point = outline[0]
                        print(f"First point type: {type(first_point)}")
                        print(f"First point attributes: {[attr for attr in dir(first_point) if not attr.startswith('_')]}")
                        
                        # Try different ways to get coordinates
                        for attr in ['x', 'y', 'position', 'point', 'coord', 'vector', 'location']:
                            if hasattr(first_point, attr):
                                try:
                                    value = getattr(first_point, attr)
                                    print(f"First point {attr}: {value}")
                                except Exception as e:
                                    print(f"Error getting first point {attr}: {e}")
                        
            except Exception as e:
                print(f"Error testing pad shapes: {e}")
                import traceback
                print(traceback.format_exc())
                        
    except Exception as e:
        print(f"Error debugging KiCad API: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
