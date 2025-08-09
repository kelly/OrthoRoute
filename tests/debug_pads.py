#!/usr/bin/env python3
"""
Debug script to analyze pad shapes using existing KiCadInterface
"""

import sys
import logging
from kicad_interface import KiCadInterface

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#!/usr/bin/env python3
"""
Debug script to analyze pad shapes using existing KiCadInterface
"""

import sys
import logging
from kicad_interface import KiCadInterface

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_large_pads():
    """Find and analyze large pads to understand shape extraction"""
    try:
        # Use existing KiCad interface
        kicad = KiCadInterface()
        if not kicad.connect():
            logger.error("Failed to connect to KiCad")
            return
        
        # Get board data 
        board_data = kicad.get_board_data()
        pads = board_data.get('pads', [])
        
        logger.info(f"Found {len(pads)} total pads")
        
        # Look for large pads (likely the square pad in upper left)
        large_pads = []
        for i, pad in enumerate(pads):
            size_x = pad.get('size_x', 0.0)
            size_y = pad.get('size_y', 0.0)
            
            # Check if this is a large pad
            if size_x > 2.0 or size_y > 2.0:
                large_pads.append((i, pad))
                logger.info(f"\n🔍 LARGE PAD {i}:")
                logger.info(f"  Position: ({pad.get('x', 0):.3f}, {pad.get('y', 0):.3f}) mm")
                logger.info(f"  Size: ({size_x:.3f}, {size_y:.3f}) mm")
                logger.info(f"  Shape: {pad.get('shape', 'Unknown')}")
                logger.info(f"  Drill: {pad.get('drill_diameter', 0):.3f} mm")
                logger.info(f"  Net: {pad.get('net', 'Unknown')}")
                logger.info(f"  Number: {pad.get('number', 'Unknown')}")
                logger.info(f"  Layers: {pad.get('layers', [])}")
        
        if not large_pads:
            logger.info("No large pads found. Looking at all pads with unusual shapes...")
            
            # Look for square pads or pads with unusual shape codes
            for i, pad in enumerate(pads[:10]):  # Check first 10 pads
                logger.info(f"\nPad {i}:")
                logger.info(f"  Position: ({pad.get('x', 0):.3f}, {pad.get('y', 0):.3f}) mm")
                logger.info(f"  Size: ({pad.get('size_x', 0):.3f}, {pad.get('size_y', 0):.3f}) mm")
                logger.info(f"  Shape: {pad.get('shape', 'Unknown')}")
                logger.info(f"  Drill: {pad.get('drill_diameter', 0):.3f} mm")
                logger.info(f"  Net: {pad.get('net', 'Unknown')}")
        
        # Now let's debug the actual KiCad pad objects directly
        logger.info("\n" + "="*50)
        logger.info("DEBUGGING ACTUAL KICAD PAD OBJECTS")
        logger.info("="*50)
        
        # Get raw footprints from KiCad
        footprints = getattr(kicad.board, 'footprints', [])
        logger.info(f"Found {len(footprints)} footprints")
        
        pad_count = 0
        for f_idx, footprint in enumerate(footprints[:5]):  # Check first 5 footprints
            logger.info(f"\n=== Footprint {f_idx}: {getattr(footprint, 'reference', 'Unknown')} ===")
            
            pads = getattr(footprint, 'pads', [])
            for p_idx, pad in enumerate(pads):
                if pad_count >= 10:  # Limit output
                    break
                    
                pad_count += 1
                logger.info(f"\n--- Raw Pad {p_idx} ---")
                
                # Basic info
                pos = getattr(pad, 'position', None)
                x = float(getattr(pos, 'x', 0.0)) / 1000000.0 if pos else 0.0
                y = float(getattr(pos, 'y', 0.0)) / 1000000.0 if pos else 0.0
                
                # Get padstack for detailed analysis
                padstack = getattr(pad, 'padstack', None)
                if padstack:
                    # Get first copper layer
                    copper_layers = getattr(padstack, 'copper_layers', [])
                    if copper_layers:
                        first_layer = copper_layers[0]
                        
                        # Size
                        size = getattr(first_layer, 'size', None)
                        if size:
                            size_x = float(getattr(size, 'x', 0.0)) / 1000000.0
                            size_y = float(getattr(size, 'y', 0.0)) / 1000000.0
                            logger.info(f"Size: ({size_x:.3f}, {size_y:.3f}) mm")
                        
                        # Shape - this is the key part we need to fix
                        shape_obj = getattr(first_layer, 'shape', None)
                        logger.info(f"Shape object: {shape_obj}")
                        logger.info(f"Shape type: {type(shape_obj)}")
                        
                        if shape_obj:
                            # Try different ways to get shape info
                            logger.info(f"Shape attributes: {[attr for attr in dir(shape_obj) if not attr.startswith('_')]}")
                            
                            if hasattr(shape_obj, 'value'):
                                logger.info(f"Shape value: {shape_obj.value}")
                            if hasattr(shape_obj, 'name'):
                                logger.info(f"Shape name: {shape_obj.name}")
                            if hasattr(shape_obj, '__str__'):
                                logger.info(f"Shape string: {str(shape_obj)}")
                            if hasattr(shape_obj, '__repr__'):
                                logger.info(f"Shape repr: {repr(shape_obj)}")
                                
                            # Try to get enum values for shape constants
                            try:
                                shape_class = shape_obj.__class__
                                logger.info(f"Shape class: {shape_class}")
                                logger.info(f"Shape class attributes: {[attr for attr in dir(shape_class) if not attr.startswith('_')]}")
                                
                                # Look for common shape constants
                                shape_constants = {}
                                for attr in dir(shape_class):
                                    if not attr.startswith('_') and attr.isupper():
                                        value = getattr(shape_class, attr)
                                        shape_constants[attr] = value
                                        logger.info(f"  {attr} = {value}")
                                
                            except Exception as e:
                                logger.debug(f"Failed to get shape constants: {e}")
                        
                        # If this is a large pad, do extra analysis
                        if size and hasattr(size, 'x'):
                            size_x = float(getattr(size, 'x', 0.0)) / 1000000.0
                            size_y = float(getattr(size, 'y', 0.0)) / 1000000.0
                            if size_x > 2.0 or size_y > 2.0:
                                logger.info(f"\n🎯 FOUND THE LARGE PAD!")
                                logger.info(f"Position: ({x:.3f}, {y:.3f}) mm")
                                logger.info(f"Size: ({size_x:.3f}, {size_y:.3f}) mm")
                                logger.info(f"Raw shape object: {shape_obj}")
                                logger.info(f"Shape details: {shape_obj.__dict__ if hasattr(shape_obj, '__dict__') else 'No dict'}")
                
                if pad_count >= 10:
                    break
            
            if pad_count >= 10:
                break
                
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        logger.exception("Full traceback:")

if __name__ == "__main__":
    debug_large_pads()
