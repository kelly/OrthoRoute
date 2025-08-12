#!/usr/bin/env python3
"""
Real thermal relief visualization using actual KiCad board data
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PyQt6.QtWidgets import QApplication
from orthoroute_window import OrthoRouteWindow
from kicad_interface import KiCadInterface

# Setup logging  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_real_kicad_data():
    """Load actual KiCad board data to show thermal reliefs"""
    try:
        # Connect to KiCad
        kicad = KiCadInterface()
        if not kicad.connect():
            logger.error("Could not connect to KiCad")
            return None
            
        board = kicad.board
        if not board:
            logger.error("Could not get board from KiCad")
            return None

        # Get basic board data
        zones = board.get_zones()
        pads = board.get_pads()
        vias = board.get_vias()
        
        logger.info(f"Found {len(zones)} zones, {len(pads)} pads, {len(vias)} vias")
        
        # Build the data structure that the GUI expects
        board_data = {
            'board_name': 'Real KiCad Board with Thermal Reliefs',
            'bounds': (0, 0, 100, 100),  # Will be calculated properly
            'components': [],
            'pads': [],
            'tracks': [],
            'vias': [],
            'airwires': [],
            'zones': [],
            'copper_pours': [],
            'keepouts': [],
            'nets': {},
            'layers': [{'id': 3, 'name': 'F.Cu'}, {'id': 34, 'name': 'B.Cu'}]
        }
        
        # Process pads
        for i, pad in enumerate(pads):
            try:
                pos = pad.position
                size = pad.size
                
                pad_data = {
                    'x': pos.x / 1000000.0,  # Convert to mm
                    'y': pos.y / 1000000.0,
                    'size_x': size.x / 1000000.0,
                    'size_y': size.y / 1000000.0,
                    'shape': int(pad.shape),
                    'footprint_layer': 'F.Cu',
                    'drill_diameter': pad.drill_size.x / 1000000.0 if pad.drill_size.x > 0 else 0
                }
                board_data['pads'].append(pad_data)
                
                if i < 5:
                    logger.info(f"Pad {i}: pos=({pad_data['x']:.2f}, {pad_data['y']:.2f}), size=({pad_data['size_x']:.2f}, {pad_data['size_y']:.2f})")
                    
            except Exception as e:
                logger.debug(f"Error processing pad {i}: {e}")
        
        # Process vias
        for i, via in enumerate(vias):
            try:
                pos = via.position
                
                via_data = {
                    'x': pos.x / 1000000.0,
                    'y': pos.y / 1000000.0,
                    'via_diameter': via.width / 1000000.0,
                    'drill_diameter': via.drill / 1000000.0
                }
                board_data['vias'].append(via_data)
                
                if i < 5:
                    logger.info(f"Via {i}: pos=({via_data['x']:.2f}, {via_data['y']:.2f}), dia={via_data['via_diameter']:.2f}")
                    
            except Exception as e:
                logger.debug(f"Error processing via {i}: {e}")
        
        # Process zones with thermal reliefs
        for i, zone in enumerate(zones):
            try:
                filled_polygons = zone.filled_polygons
                logger.info(f"Zone {i}: net={zone.net}, layers={zone.layers}")
                
                for layer_id, polygon_list in filled_polygons.items():
                    logger.info(f"  Layer {layer_id}: {len(polygon_list)} polygons")
                    
                    for j, polygon in enumerate(polygon_list):
                        outline = polygon.outline
                        holes = polygon.holes
                        
                        logger.info(f"    Polygon {j}: {len(outline)} outline points, {len(holes)} holes")
                        
                        if len(outline) > 100:  # This is a complex copper pour
                            # Convert the complex outline to our format
                            outline_points = []
                            for point in outline:
                                if hasattr(point, 'point'):
                                    x = point.point.x / 1000000.0
                                    y = point.point.y / 1000000.0
                                    outline_points.append({'x': x, 'y': y})
                            
                            # Convert holes if any
                            hole_points = []
                            for hole in holes:
                                hole_outline = []
                                for point in hole:
                                    if hasattr(point, 'point'):
                                        x = point.point.x / 1000000.0
                                        y = point.point.y / 1000000.0
                                        hole_outline.append({'x': x, 'y': y})
                                if hole_outline:
                                    hole_points.append(hole_outline)
                            
                            pour_data = {
                                'net': str(zone.net),
                                'layers': list(zone.layers),
                                'filled_polygons': {
                                    layer_id: [
                                        {
                                            'outline': outline_points,
                                            'holes': hole_points
                                        }
                                    ]
                                }
                            }
                            board_data['copper_pours'].append(pour_data)
                            
                            logger.info(f"    Added copper pour with {len(outline_points)} points, {len(hole_points)} holes")
                            
            except Exception as e:
                logger.debug(f"Error processing zone {i}: {e}")
        
        # Calculate bounds
        all_x = []
        all_y = []
        
        for pad in board_data['pads']:
            all_x.append(pad['x'])
            all_y.append(pad['y'])
            
        for via in board_data['vias']:
            all_x.append(via['x'])
            all_y.append(via['y'])
            
        for pour in board_data['copper_pours']:
            for layer_id, polygons in pour['filled_polygons'].items():
                for polygon in polygons:
                    for point in polygon['outline']:
                        all_x.append(point['x'])
                        all_y.append(point['y'])
        
        if all_x and all_y:
            margin = 5  # 5mm margin
            board_data['bounds'] = (
                min(all_x) - margin, min(all_y) - margin,
                max(all_x) + margin, max(all_y) + margin
            )
            logger.info(f"Board bounds: {board_data['bounds']}")
        
        return board_data
        
    except Exception as e:
        logger.error(f"Error loading KiCad data: {e}")
        return None

def main():
    app = QApplication(sys.argv)
    
    # Load real KiCad data
    board_data = load_real_kicad_data()
    
    if not board_data:
        logger.error("Failed to load KiCad data")
        return 1
    
    # Create the window
    window = OrthoRouteWindow(board_data, None)
    window.setWindowTitle("OrthoRoute - Real KiCad Thermal Reliefs")
    
    pads_count = len(board_data['pads'])
    pours_count = len(board_data['copper_pours'])
    total_points = sum(len(pour['filled_polygons'][list(pour['filled_polygons'].keys())[0]][0]['outline']) 
                      for pour in board_data['copper_pours'] if pour['filled_polygons'])
    
    logger.info(f"🎯 Loaded real KiCad board: {pads_count} pads, {pours_count} copper pours, {total_points} total outline points")
    logger.info("The thermal reliefs are embedded in the complex copper pour outlines!")
    
    window.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
