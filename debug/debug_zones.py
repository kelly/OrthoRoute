#!/usr/bin/env python3

import logging
from kipy import KiCad

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    logger.info("Connecting to KiCad...")
    kicad = KiCad()
    board = kicad.get_board()
    logger.info(f"Board connected: {board}")
    
    zones = board.get_zones()
    logger.info(f"Found {len(zones)} zones")
    
    for i, zone in enumerate(zones):
        logger.info(f"\nZone {i}:")
        logger.info(f"  Type: {type(zone)}")
        logger.info(f"  Dir: {[attr for attr in dir(zone) if not attr.startswith('_')]}")
        
        # Try to get some properties
        try:
            net_name = getattr(zone, 'net_name', 'No net_name')
            logger.info(f"  Net name: {net_name}")
        except:
            logger.info(f"  Error getting net_name")
            
        try:
            layer = getattr(zone, 'layer', 'No layer')
            logger.info(f"  Layer: {layer}")
        except:
            logger.info(f"  Error getting layer")
            
        # Try different outline methods
        for outline_attr in ['outline', 'get_outline', 'get_boundary', 'boundary', 'contour']:
            try:
                outline = getattr(zone, outline_attr, None)
                if outline:
                    logger.info(f"  Found {outline_attr}: {type(outline)}")
                    if hasattr(outline, '__len__'):
                        logger.info(f"    Length: {len(outline)}")
            except Exception as e:
                logger.info(f"  Error with {outline_attr}: {e}")
        
        if i >= 1:  # Limit to first 2 zones
            break
            
except Exception as e:
    logger.error(f"Error: {e}")
    import traceback
    traceback.print_exc()
