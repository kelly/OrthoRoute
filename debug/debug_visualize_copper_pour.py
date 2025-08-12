#!/usr/bin/env python3
"""
Debug script to visualize the actual copper pour geometry with thermal reliefs
"""

import sys
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from kicad_interface import KiCadInterface

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_copper_pour_geometry():
    """Analyze the actual geometry of the copper pour"""
    try:
        # Connect to KiCad
        kicad = KiCadInterface()
        if not kicad.connect():
            logger.error("Could not connect to KiCad")
            return
            
        board = kicad.board
        
        if not board:
            logger.error("Could not get board from KiCad")
            return
            
        zones = board.get_zones()
        logger.info(f"Found {len(zones)} zones")
        
        for i, zone in enumerate(zones):
            logger.info(f"\n=== Zone {i} ===")
            logger.info(f"Zone layers: {zone.layers}")
            logger.info(f"Zone net: {zone.net}")
            
            filled_polygons = zone.filled_polygons
            logger.info(f"Filled polygons keys: {list(filled_polygons.keys())}")
            
            for layer_id, polygon_list in filled_polygons.items():
                logger.info(f"  Layer {layer_id}: {len(polygon_list)} polygons")
                
                for j, polygon in enumerate(polygon_list):
                    outline = polygon.outline
                    holes = polygon.holes
                    logger.info(f"    Polygon {j}: {len(outline)} outline points, {len(holes)} holes")
                    
                    if len(outline) > 1000:  # This is the copper pour!
                        logger.info(f"    🎯 FOUND COPPER POUR! {len(outline)} points")
                        
                        # Sample some points to show the complex geometry
                        logger.info("    First few outline points:")
                        for k, point in enumerate(outline[:5]):
                            logger.info(f"      Point {k}: {point} (type: {type(point)})")
                        
                        logger.info("    This is the actual copper geometry with thermal reliefs already cut out!")
                        logger.info("    The thermal reliefs are not separate holes - they're part of the complex outline!")
                        
    except Exception as e:
        logger.error(f"Error analyzing copper pour: {e}")

if __name__ == "__main__":
    analyze_copper_pour_geometry()
