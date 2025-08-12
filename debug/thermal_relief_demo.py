#!/usr/bin/env python3
"""
Quick thermal relief demo using the PCB viewer
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PyQt6.QtWidgets import QApplication
from orthoroute_window import OrthoRouteWindow

# Setup logging  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_thermal_relief_demo():
    """Create demo data that simulates the thermal relief copper pour"""
    
    # Simulate the complex 5,505-point copper pour outline that traces around thermal reliefs
    # This represents the actual data we're getting from KiCad
    complex_outline_points = []
    
    # Create a complex polygon that simulates thermal relief cutouts
    # This would normally be the 5,505 points from KiCad
    import math
    
    # Outer boundary
    for i in range(100):
        angle = i * 2 * math.pi / 100
        x = 50 + 40 * math.cos(angle)
        y = 50 + 40 * math.sin(angle)
        complex_outline_points.append({'x': x, 'y': y})
    
    # Add thermal relief "spokes" - complex cutouts in the outline
    # Simulate pad clearances and thermal relief geometry
    for pad_x, pad_y in [(30, 30), (70, 30), (30, 70), (70, 70)]:
        # Create thermal relief spoke pattern around each "pad"
        spoke_points = []
        for angle in [0, 90, 180, 270]:  # 4 thermal spokes
            rad = math.radians(angle)
            # Inner clearance circle
            for r in [2, 3, 4]:
                x = pad_x + r * math.cos(rad)
                y = pad_y + r * math.sin(rad)
                spoke_points.append({'x': x, 'y': y})
        complex_outline_points.extend(spoke_points)
    
    demo_data = {
        'board_name': 'Thermal Relief Demo',
        'bounds': (0, 0, 100, 100),
        'components': [
            {'reference': 'U1', 'x': 50, 'y': 50, 'rotation': 0, 'layer': 'F.Cu'}
        ],
        'pads': [
            {'x': 30, 'y': 30, 'size_x': 3, 'size_y': 3, 'shape': 1, 'footprint_layer': 'F.Cu'},
            {'x': 70, 'y': 30, 'size_x': 3, 'size_y': 3, 'shape': 1, 'footprint_layer': 'F.Cu'},
            {'x': 30, 'y': 70, 'size_x': 3, 'size_y': 3, 'shape': 1, 'footprint_layer': 'F.Cu'},
            {'x': 70, 'y': 70, 'size_x': 3, 'size_y': 3, 'shape': 1, 'footprint_layer': 'F.Cu'},
        ],
        'tracks': [],
        'vias': [
            {'x': 40, 'y': 40, 'via_diameter': 1.0, 'drill_diameter': 0.5},
            {'x': 60, 'y': 60, 'via_diameter': 1.0, 'drill_diameter': 0.5},
        ],
        'airwires': [],
        'zones': [],  # Regular zones
        'copper_pours': [  # This is where the thermal reliefs live!
            {
                'net': 'GND',
                'layers': [3],  # F.Cu
                'filled_polygons': {
                    3: [  # Layer 3 (F.Cu)
                        {
                            'outline': complex_outline_points,  # The 5,505-point complex boundary!
                            'holes': []  # No holes - thermal reliefs are in the outline!
                        }
                    ]
                }
            }
        ],
        'keepouts': [],
        'nets': {},
        'layers': [{'id': 3, 'name': 'F.Cu'}, {'id': 34, 'name': 'B.Cu'}]
    }
    
    return demo_data

def main():
    app = QApplication(sys.argv)
    
    # Create demo data showing thermal reliefs
    demo_data = create_thermal_relief_demo()
    
    # Create the window
    window = OrthoRouteWindow(demo_data, None)
    window.setWindowTitle("OrthoRoute - Thermal Relief Demo (5,505 Points!)")
    
    logger.info("🎯 Demo showing thermal reliefs as complex copper pour outline")
    logger.info(f"Copper pour has {len(demo_data['copper_pours'][0]['filled_polygons'][3][0]['outline'])} outline points")
    logger.info("This simulates the real KiCad data where thermal reliefs are part of the outline!")
    
    window.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
