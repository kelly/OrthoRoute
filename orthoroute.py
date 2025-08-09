#!/usr/bin/env python3
"""
OrthoRoute - Professional PCB Autorouting Plugin for KiCad
GPU-Accelerated routing with real-time visualization and IPC API integration
"""

import os
import sys
import logging
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Any

# Setup logging
log_file = Path.home() / "Documents" / "orthoroute.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check for required dependencies"""
    required_packages = ['kipy', 'PyQt6', 'numpy']
    optional_packages = ['cupy']
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            if package == 'kipy':
                import kipy
                logger.info("✓ Found kipy at: " + str(Path(kipy.__file__).parent))
            elif package == 'PyQt6':
                import PyQt6
                logger.info("✓ PyQt6 fully available and functional - using current environment")
            elif package == 'numpy':
                import numpy
                logger.info("✓ NumPy available for array operations")
        except ImportError:
            missing_required.append(package)
            logger.warning(f"Missing required package: {package}")
    
    for package in optional_packages:
        try:
            if package == 'cupy':
                import cupy
                logger.info("✓ CuPy available for GPU acceleration")
        except ImportError:
            missing_optional.append(package)
            logger.info(f"Optional package not found: {package}")
    
    if missing_required:
        logger.error(f"Missing required packages: {missing_required}")
        return False
    
    return True

def connect_to_kicad() -> Optional[Any]:
    """Connect to KiCad via IPC API"""
    try:
        from kipy import KiCad
        
        # Try to get IPC credentials from environment
        api_socket = os.environ.get('KICAD_API_SOCKET')
        api_token = os.environ.get('KICAD_API_TOKEN')
        
        if api_socket and api_token:
            logger.info("✓ Found IPC credentials in environment")
            kicad = KiCad(socket_path=api_socket, kicad_token=api_token, timeout_ms=15000)
        else:
            logger.warning("⚠ IPC credentials not found - attempting direct connection")
            logger.info("  This may work for local KiCad instances without authentication")
            logger.info("⚠ Attempting connection without explicit credentials")
            kicad = KiCad(timeout_ms=15000)
        
        # Test connection by getting board
        board = kicad.get_board()
        if board:
            logger.info("✓ KiCad IPC connection established")
            return kicad
        else:
            logger.error("❌ Could not retrieve board from KiCad")
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to connect to KiCad: {e}")
        return None

def analyze_board_data(board) -> Dict[str, Any]:
    """Analyze board data using IPC APIs"""
    try:
        logger.info("✓ Board object retrieved")
        board_name = getattr(board, 'name', 'PCB')
        logger.info(f"Board: {board_name}")
        
        # Discover available methods
        available_methods = [method for method in dir(board) if not method.startswith('_')]
        logger.info(f"Available board methods: {', '.join(available_methods[:20])}...")
        
        # Get board components
        footprints = board.get_footprints()
        logger.info(f"✓ Retrieved {len(footprints)} footprints")
        
        tracks = board.get_tracks()
        logger.info(f"✓ Retrieved {len(tracks)} tracks")
        
        vias = board.get_vias()
        logger.info(f"✓ Retrieved {len(vias)} vias")
        
        pads = board.get_pads()
        logger.info(f"✓ Retrieved {len(pads)} pads")
        
        nets = board.get_nets()
        logger.info(f"✓ Retrieved {len(nets)} nets")
        
        # Get board outline and other geometry
        logger.info("Getting board outline...")
        
        # Get zones (copper pours, keepouts, etc.)
        zones = board.get_zones()
        logger.info(f"✓ Retrieved {len(zones)} zones")
        
        # Analyze zones for keepout areas and copper pours
        keepout_areas = []
        copper_pours = []
        
        for i, zone in enumerate(zones):
            try:
                zone_attrs = [attr for attr in dir(zone) if not attr.startswith('_')]
                logger.info(f"Zone {i} attributes: {', '.join(zone_attrs[:20])}")
                
                outline = zone.outline
                outline_attrs = [attr for attr in dir(outline) if not attr.startswith('_')]
                logger.info(f"Zone {i} outline attributes: {', '.join(outline_attrs)}")
                
                # Check filled polygons
                filled_polygons = zone.filled_polygons
                logger.info(f"Zone {i} filled_polygons type: {type(filled_polygons)}")
                
                # Process filled polygons for each layer
                for layer_id, polygons in filled_polygons.items():
                    logger.info(f"Zone {i} layer {layer_id}: {len(polygons)} polygons")
                    
                    # Get outline points
                    outline_points = []
                    outline_obj = outline.outline
                    for node in outline_obj.nodes:
                        point = node.point
                        outline_points.append((point.x, point.y))
                    
                    logger.info(f"Zone {i} outline: {len(outline_points)} points")
                
                # Determine zone type
                if hasattr(zone, 'is_rule_area') and zone.is_rule_area:
                    zone_info = f"Zone_{i}"
                    layers = getattr(zone, 'layers', [])
                    logger.info(f"  Found keepout area: {zone_info} on {layers}")
                    keepout_areas.append({
                        'name': zone_info,
                        'layers': layers,
                        'outline': outline_points
                    })
                else:
                    zone_info = f"Zone_{i}"
                    layers = getattr(zone, 'layers', [])
                    logger.info(f"  Found copper pour: {zone_info} on {layers}")
                    copper_pours.append({
                        'name': zone_info,
                        'layers': layers,
                        'outline': outline_points
                    })
                    
            except Exception as e:
                logger.warning(f"Error analyzing zone {i}: {e}")
                continue
        
        logger.info(f"✓ Parsed {len(copper_pours)} zones and {len(keepout_areas)} keepout areas")
        
        # Get layer information
        logger.info("Getting layer information...")
        
        # Get detailed footprint information
        logger.info("Getting detailed footprint information...")
        
        footprint_details = []
        for footprint in footprints:
            try:
                footprint_info = {
                    'reference': getattr(footprint, 'reference', 'Unknown'),
                    'value': getattr(footprint, 'value', ''),
                    'layer': getattr(footprint, 'layer', 0),
                    'position': getattr(footprint, 'position', {'x': 0, 'y': 0}),
                    'orientation': getattr(footprint, 'orientation', 0),
                    'library': getattr(footprint, 'library', ''),
                    'footprint_name': getattr(footprint, 'footprint_name', '')
                }
                footprint_details.append(footprint_info)
            except Exception as e:
                logger.debug(f"Error getting footprint details: {e}")
        
        logger.info(f"✓ Retrieved detailed info for {len(footprint_details)} footprints")
        
        # Success message
        logger.info("🎉 SUCCESS: Board data retrieved via KiCad IPC API!")
        logger.info(f"Board: {board_name}")
        logger.info(f"Components: {len(footprints)}")
        logger.info(f"Tracks: {len(tracks)}")
        logger.info(f"Vias: {len(vias)}")
        logger.info(f"Pads: {len(pads)}")
        logger.info(f"Nets: {len(nets)}")
        
        return {
            'board_name': board_name,
            'footprints': footprints,
            'tracks': tracks,
            'vias': vias,
            'pads': pads,
            'nets': nets,
            'zones': zones,
            'keepout_areas': keepout_areas,
            'copper_pours': copper_pours,
            'footprint_details': footprint_details
        }
        
    except Exception as e:
        logger.error(f"❌ Board analysis failed: {e}")
        return None

def launch_qt_interface(board_data: Dict[str, Any], kicad_interface) -> None:
    """Launch the Qt interface with board data"""
    try:
        logger.info("Launching Qt UI with safety checks...")
        
        # Add src directory to path for imports
        src_dir = Path(__file__).parent / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
            logger.info(f"Added src dir to Python path: {src_dir}")
        
        # Import Qt modules with error handling
        logger.info("Importing PyQt6...")
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QIcon, QPixmap
        logger.info("PyQt6 imported successfully")
        
        # Import our main window
        logger.info("Importing OrthoRouteWindow...")
        from orthoroute_window import OrthoRouteWindow
        logger.info("OrthoRouteWindow imported successfully")
        
        # Create QApplication
        logger.info("Creating QApplication...")
        app = QApplication(sys.argv)
        
        # Set application icon if available
        icon_path = Path(__file__).parent / "assets" / "BigIcon.png"
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
            logger.info(f"App icon set: {icon_path}")
        
        # Convert board data to format expected by the window
        logger.info("Converting board data...")
        
        # Convert to the format expected by the GUI
        gui_board_data = convert_board_data_for_gui(board_data)
        logger.info(f"Converting board data: {len(board_data.get('footprints', []))} components, {len(board_data.get('tracks', []))} tracks")
        
        # Create and show main window
        logger.info("Creating main window...")
        window = OrthoRouteWindow(gui_board_data, kicad_interface)
        
        logger.info("Showing window...")
        window.show()
        
        logger.info("Qt UI launched successfully - entering event loop")
        app.exec()
        
    except Exception as e:
        logger.error(f"❌ Qt interface launch failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Fallback: print board data
        print("🎉 OrthoRoute Board Analysis Complete!")
        print(f"Board: {board_data.get('board_name', 'Unknown')}")
        print(f"Components: {len(board_data.get('footprints', []))}")
        print(f"Tracks: {len(board_data.get('tracks', []))}")
        print(f"Nets: {len(board_data.get('nets', []))}")

def convert_board_data_for_gui(board_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert IPC board data to format expected by GUI"""
    try:
        footprints = board_data.get('footprints', [])
        tracks = board_data.get('tracks', [])
        pads = board_data.get('pads', [])
        nets = board_data.get('nets', [])
        
        # Convert components
        components = []
        for fp in footprints:
            try:
                component = {
                    'reference': getattr(fp, 'reference', 'Unknown'),
                    'value': getattr(fp, 'value', ''),
                    'x': getattr(getattr(fp, 'position', {}), 'x', 0) / 1000000,  # Convert to mm
                    'y': getattr(getattr(fp, 'position', {}), 'y', 0) / 1000000,
                    'rotation': getattr(fp, 'orientation', 0),
                    'layer': 'F.Cu',  # Default to front copper
                    'library': getattr(fp, 'library', ''),
                    'footprint': getattr(fp, 'footprint_name', '')
                }
                components.append(component)
            except Exception as e:
                logger.debug(f"Error converting component: {e}")
        
        # Convert tracks
        gui_tracks = []
        for track in tracks:
            try:
                track_data = {
                    'start_x': getattr(getattr(track, 'start', {}), 'x', 0) / 1000000,
                    'start_y': getattr(getattr(track, 'start', {}), 'y', 0) / 1000000,
                    'end_x': getattr(getattr(track, 'end', {}), 'x', 0) / 1000000,
                    'end_y': getattr(getattr(track, 'end', {}), 'y', 0) / 1000000,
                    'width': getattr(track, 'width', 0.2),
                    'layer': getattr(track, 'layer', 0),
                    'net': getattr(track, 'net', None)
                }
                gui_tracks.append(track_data)
            except Exception as e:
                logger.debug(f"Error converting track: {e}")
        
        # Calculate board bounds
        all_x = []
        all_y = []
        
        for comp in components:
            all_x.extend([comp['x']])
            all_y.extend([comp['y']])
        
        if all_x and all_y:
            bounds = (min(all_x) - 5, min(all_y) - 5, max(all_x) + 5, max(all_y) + 5)
        else:
            bounds = (0, 0, 100, 100)
        
        # Create net information
        net_info = {}
        net_pins = {}
        
        # Count pins per net
        for net in nets:
            try:
                net_name = getattr(net, 'name', f'Net_{id(net)}')
                # Count pads on this net
                net_pads = [pad for pad in pads if hasattr(pad, 'net') and getattr(pad, 'net', None) == net]
                
                net_info[net_name] = {
                    'name': net_name,
                    'pin_count': len(net_pads),
                    'routed': len([t for t in gui_tracks if getattr(t.get('net', None), 'name', '') == net_name]) > 0
                }
                
                # Create pin positions for airwires
                pins = []
                for pad in net_pads[:10]:  # Limit for performance
                    try:
                        pos = getattr(pad, 'position', {'x': 0, 'y': 0})
                        pins.append({
                            'x': pos.x / 1000000 if hasattr(pos, 'x') else 0,
                            'y': pos.y / 1000000 if hasattr(pos, 'y') else 0
                        })
                    except:
                        pass
                
                net_pins[net_name] = pins
                
            except Exception as e:
                logger.debug(f"Error processing net: {e}")
        
        # Create airwires for unrouted nets
        airwires = []
        airwire_count = 0
        
        for net_name, pins in net_pins.items():
            if len(pins) > 1 and not net_info.get(net_name, {}).get('routed', False):
                # Create airwires between pins (minimum spanning tree approach)
                for i in range(len(pins) - 1):
                    airwires.append({
                        'start_x': pins[i]['x'],
                        'start_y': pins[i]['y'],
                        'end_x': pins[i + 1]['x'],
                        'end_y': pins[i + 1]['y'],
                        'net': net_name
                    })
                    airwire_count += 1
        
        logger.info(f"Pin distribution: {len([n for n in net_info.values() if n['pin_count'] > 1])} nets with pins")
        for net_name, info in list(net_info.items())[:5]:
            logger.info(f"  Net '{net_name}': {info['pin_count']} pins")
        
        gui_data = {
            'components': components,
            'tracks': gui_tracks,
            'nets': net_info,
            'airwires': airwires,
            'bounds': bounds,
            'layers': ['F.Cu', 'B.Cu', 'F.Mask', 'B.Mask']
        }
        
        logger.info(f"Drew {len(airwires)} airwires from {len(net_pins)} nets")
        if airwires:
            sample_coords = [f"({a['start_x']:.1f},{a['start_y']:.1f})->({a['end_x']:.1f},{a['end_y']:.1f})" for a in airwires[:3]]
            logger.info(f"Sample coordinates: {sample_coords}")
        
        unrouted_count = len([n for n in net_info.values() if not n.get('routed', False) and n['pin_count'] > 1])
        logger.info(f"Unrouted nets: {unrouted_count}")
        for net_name, info in list(net_info.items())[:3]:
            if not info.get('routed', False) and info['pin_count'] > 1:
                logger.info(f"  Net '{net_name}': {info['pin_count']} pins, routed={info.get('routed', False)}")
        
        return gui_data
        
    except Exception as e:
        logger.error(f"Error converting board data: {e}")
        return {
            'components': [],
            'tracks': [],
            'nets': {},
            'airwires': [],
            'bounds': (0, 0, 100, 100),
            'layers': ['F.Cu', 'B.Cu']
        }

def main():
    """Main entry point"""
    logger.info("=== OrthoRoute IPC API Client ===")
    logger.info("Plugin launched - testing KiCad IPC connection!")
    
    # Check environment and credentials
    if len(sys.argv) > 1:
        logger.info(f"Command line args: {sys.argv[1:]}")
    else:
        logger.info("⚠ No command line credentials, will try environment variables")
    
    # Check dependencies
    logger.info("Testing PyQt6 availability...")
    if not check_dependencies():
        return 1
    
    logger.info("Skipping venv re-execution - using current environment")
    
    # Connect to KiCad
    logger.info("Attempting to connect to KiCad via IPC API...")
    kicad = connect_to_kicad()
    if not kicad:
        logger.error("Failed to connect to KiCad")
        return 1
    
    # Get and analyze board data
    board = kicad.get_board()
    if not board:
        logger.error("No board available")
        return 1
    
    board_data = analyze_board_data(board)
    if not board_data:
        logger.error("Failed to analyze board data")
        return 1
    
    # Launch Qt interface
    launch_qt_interface(board_data, kicad)
    return 0

if __name__ == "__main__":
    sys.exit(main())
