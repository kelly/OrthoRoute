#!/usr/bin/env python3
"""
Debug KiCad API attributes to find the correct property names
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from kicad_interface import KiCadInterface

def main():
    """Debug KiCad API attributes."""
    print("Debugging KiCad API attributes...")
    
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
        
        # Get first pad and inspect its attributes
        pads = board.get_pads()
        print(f"Found {len(pads)} pads")
        if pads:
            pad = pads[0]
            print(f"First pad type: {type(pad)}")
            print(f"First pad attributes: {[attr for attr in dir(pad) if not attr.startswith('_')]}")
            
            # Explore pad properties
            print(f"Pad position: {pad.position}")
            print(f"Pad number: {pad.number}")
            print(f"Pad net: {pad.net}")
            print(f"Pad type: {pad.pad_type}")
            
            # Explore padstack for size information
            padstack = pad.padstack
            print(f"Padstack type: {type(padstack)}")
            
            # Get drill size
            drill = padstack.drill
            print(f"Drill type: {type(drill)}")
            print(f"Drill attributes: {[attr for attr in dir(drill) if not attr.startswith('_')]}")
            for attr in ['diameter', 'size', 'width', 'height']:
                if hasattr(drill, attr):
                    try:
                        value = getattr(drill, attr)
                        print(f"Drill {attr}: {value}")
                    except Exception as e:
                        print(f"Error getting drill {attr}: {e}")
            
            # Look at copper layers for size
            copper_layers = padstack.copper_layers
            print(f"Copper layers count: {len(copper_layers)}")
            if copper_layers:
                copper_layer = copper_layers[0]
                print(f"Copper layer size: {copper_layer.size}")
                print(f"Copper layer shape: {copper_layer.shape}")
                print(f"Copper layer layer: {copper_layer.layer}")
        
        # Get first footprint and inspect its attributes  
        footprints = board.get_footprints()
        print(f"Found {len(footprints)} footprints")
        if footprints:
            footprint = footprints[0]
            print(f"First footprint type: {type(footprint)}")
            
            # Try to get reference and value
            try:
                ref_field = footprint.reference_field
                print(f"Reference field type: {type(ref_field)}")
                print(f"Reference field attributes: {[attr for attr in dir(ref_field) if not attr.startswith('_')]}")
                
                # Try to get the text value
                for attr in ['text', 'value', 'content']:
                    if hasattr(ref_field, attr):
                        try:
                            value = getattr(ref_field, attr)
                            print(f"Reference field {attr}: {value}")
                        except Exception as e:
                            print(f"Error getting reference field {attr}: {e}")
                            
            except Exception as e:
                print(f"Error getting reference field: {e}")
                
            try:
                val_field = footprint.value_field  
                print(f"Value field type: {type(val_field)}")
                
                # Try to get the text value
                for attr in ['text', 'value', 'content']:
                    if hasattr(val_field, attr):
                        try:
                            value = getattr(val_field, attr)
                            print(f"Value field {attr}: {value}")
                        except Exception as e:
                            print(f"Error getting value field {attr}: {e}")
                            
            except Exception as e:
                print(f"Error getting value field: {e}")
                        
    except Exception as e:
        print(f"Error debugging KiCad API: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
