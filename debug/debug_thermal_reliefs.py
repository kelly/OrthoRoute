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
        logger.info(f"\n=== Zone {i} Details ===")
        logger.info(f"Zone filled: {zone.filled}")
        logger.info(f"Zone layers: {zone.layers}")
        logger.info(f"Zone net: {zone.net}")
        
        # Check filled polygons
        filled_polygons = zone.filled_polygons
        logger.info(f"Filled polygons keys: {list(filled_polygons.keys())}")
        
        for layer_id, polygon_list in filled_polygons.items():
            logger.info(f"  Layer {layer_id}: {len(polygon_list)} polygons")
            
            for j, polygon in enumerate(polygon_list):
                outline_count = 0
                holes_count = 0
                
                if hasattr(polygon, 'outline'):
                    outline = polygon.outline
                    if hasattr(outline, '__len__'):
                        outline_count = len(outline)
                
                if hasattr(polygon, 'holes'):
                    holes = polygon.holes
                    if holes:
                        holes_count = len(holes)
                        logger.info(f"    Polygon {j}: {outline_count} outline points, {holes_count} holes")
                        
                        # Show details of first few holes
                        for h, hole in enumerate(holes[:3]):
                            if hasattr(hole, '__len__'):
                                logger.info(f"      Hole {h}: {len(hole)} points")
                    else:
                        logger.info(f"    Polygon {j}: {outline_count} outline points, 0 holes")
                else:
                    logger.info(f"    Polygon {j}: {outline_count} outline points, no holes attribute")
                    
        # Try to force refill
        logger.info(f"Zone filled status before refill: {zone.filled}")
        
    # Force refill zones to ensure thermal reliefs are calculated
    logger.info("\nForcing zone refill...")
    board.refill_zones(block=True)
    logger.info("Zone refill complete!")
    
    # Check again after refill
    for i, zone in enumerate(zones):
        logger.info(f"\n=== Zone {i} After Refill ===")
        logger.info(f"Zone filled: {zone.filled}")
        
        filled_polygons = zone.filled_polygons
        for layer_id, polygon_list in filled_polygons.items():
            logger.info(f"  Layer {layer_id}: {len(polygon_list)} polygons")
            
            for j, polygon in enumerate(polygon_list):
                holes_count = 0
                if hasattr(polygon, 'holes') and polygon.holes:
                    holes_count = len(polygon.holes)
                    
                logger.info(f"    Polygon {j}: {holes_count} holes")
                
                if holes_count > 0:
                    logger.info(f"    SUCCESS: Found thermal relief holes!")
                    
except Exception as e:
    logger.error(f"Error: {e}")
    import traceback
    traceback.print_exc()
