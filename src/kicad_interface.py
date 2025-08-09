#!/usr/bin/env python3
"""
KiCad IPC Interface - Handles communication with KiCad via IPC API (kipy)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
import sys
import time

logger = logging.getLogger(__name__)

@dataclass
class BoardData:
    """Container for board data extracted from KiCad"""
    filename: str
    width: float  # mm
    height: float  # mm
    layers: int
    nets: List[Dict]
    components: List[Dict]
    tracks: List[Dict]
    vias: List[Dict]
    pads: List[Dict]
    bounds: Tuple[float, float, float, float]  # min_x, min_y, max_x, max_y


def _ipc_retry(func, desc: str, max_retries: int = 3, sleep_s: float = 0.5):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            msg = str(e)
            last_err = e
            logger.warning(f"IPC '{desc}' failed (attempt {attempt}/{max_retries}): {msg}")
            if "Timed out" in msg or "AS_BUSY" in msg or "busy" in msg.lower():
                time.sleep(sleep_s)
                continue
            break
    if last_err:
        raise last_err


class KiCadInterface:
    """Interface to KiCad via IPC API (kicad-python -> kipy)"""

    def __init__(self):
        self.client = None
        self.board = None
        self.connected = False

    def connect(self) -> bool:
        """Connect to KiCad via IPC API"""
        try:
            # Ensure kipy is importable from user site (common when KiCad launches process)
            try:
                from kipy import KiCad  # type: ignore
            except ImportError:
                import site
                user_site = site.getusersitepackages()
                if user_site and user_site not in sys.path:
                    sys.path.insert(0, user_site)
                from kipy import KiCad  # retry

            # Gather credentials if provided by KiCad runtime
            api_socket = os.environ.get('KICAD_API_SOCKET')
            api_token = os.environ.get('KICAD_API_TOKEN')
            timeout_ms = 25000
            if api_socket or api_token:
                self.client = KiCad(socket_path=api_socket, kicad_token=api_token, timeout_ms=timeout_ms)
            else:
                self.client = KiCad(timeout_ms=timeout_ms)

            # Get board to confirm connection - try different methods
            try:
                # Method 1: Try get_board directly
                self.board = _ipc_retry(self.client.get_board, "get_board", max_retries=2, sleep_s=0.5)
            except Exception as e1:
                logger.warning(f"get_board failed: {e1}")
                try:
                    # Method 2: Try getting open documents first
                    docs = self.client.get_open_documents()
                    if docs and len(docs) > 0:
                        # Use first open document
                        self.board = docs[0]
                        logger.info(f"✅ Retrieved board from open documents: {getattr(self.board, 'name', 'Unknown')}")
                    else:
                        raise Exception("No open documents found")
                except Exception as e2:
                    logger.warning(f"get_open_documents failed: {e2}")
                    # Method 3: Try direct board access
                    self.board = self.client.board
                    if not self.board:
                        raise Exception("No board available through any method")
                    
            self.connected = True
            logger.info("✅ Connected to KiCad IPC API and retrieved board")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to connect to KiCad: {e}")
            self.connected = False
            return False

    def get_board_data(self) -> Dict:
        """Extract comprehensive board data from KiCad"""
        if not self.connected or not self.board:
            logger.error("Not connected to KiCad")
            return self._get_fallback_board_data()

        board = self.board

        # File/name
        try:
            filename = getattr(board, 'name', None) or getattr(board, 'filename', 'Untitled')
        except Exception:
            filename = 'Untitled'

        # Components
        components = []
        try:
            fps = _ipc_retry(board.get_footprints, "get_footprints", max_retries=3, sleep_s=0.7)
            for i, fp in enumerate(fps):
                try:
                    pos = getattr(fp, 'position', None)
                    x = float(getattr(pos, 'x', 0.0)) / 1000000.0 if pos is not None else 0.0  # Convert nm to mm
                    y = float(getattr(pos, 'y', 0.0)) / 1000000.0 if pos is not None else 0.0  # Convert nm to mm
                    ref = None
                    try:
                        ref = getattr(getattr(getattr(fp, 'reference_field', None), 'text', None), 'value', None)
                    except Exception:
                        pass
                    val = None
                    try:
                        val = getattr(getattr(getattr(fp, 'value_field', None), 'text', None), 'value', None)
                    except Exception:
                        pass
                    rot = getattr(getattr(fp, 'orientation', None), 'degrees', 0.0)
                    layer = getattr(fp, 'layer', 'F.Cu')
                    components.append({
                        'reference': ref or f'U{i}',
                        'value': val or '',
                        'x': x,
                        'y': y,
                        'rotation': float(rot) if isinstance(rot, (int, float)) else 0.0,
                        'layer': layer
                    })
                except Exception as e:
                    logger.warning(f"Footprint parse error #{i}: {e}")
        except Exception as e:
            logger.error(f"Error getting footprints: {e}")

        # Pads (used to derive net pins) - Enhanced with full geometric data for accurate rendering
        pads = []
        try:
            all_pads = _ipc_retry(board.get_pads, "get_pads", max_retries=3, sleep_s=0.7)
            for i, p in enumerate(all_pads):
                try:
                    # Basic position and net data
                    pos = getattr(p, 'position', None)
                    x = float(getattr(pos, 'x', 0.0)) / 1000000.0 if pos is not None else 0.0  # Convert nm to mm
                    y = float(getattr(pos, 'y', 0.0)) / 1000000.0 if pos is not None else 0.0  # Convert nm to mm
                    net = getattr(getattr(p, 'net', None), 'name', None)
                    num = getattr(p, 'number', None)
                    
                    # Enhanced geometric data from padstack for accurate rendering
                    padstack = getattr(p, 'padstack', None)
                    size_x = 1.3  # Default
                    size_y = 1.3  # Default  
                    drill_diameter = 0.0
                    shape = 1  # Default to rectangle
                    layers = []
                    
                    if padstack:
                        # Get drill diameter from padstack
                        drill = getattr(padstack, 'drill', None)
                        if drill:
                            drill_dia = getattr(drill, 'diameter', None)
                            if drill_dia and hasattr(drill_dia, 'x'):
                                drill_diameter = float(getattr(drill_dia, 'x', 0.0)) / 1000000.0  # Convert nm to mm
                        
                        # Get pad size and shape from copper layers
                        copper_layers = getattr(padstack, 'copper_layers', [])
                        if copper_layers and len(copper_layers) > 0:
                            first_layer = copper_layers[0]
                            
                            # Size
                            size = getattr(first_layer, 'size', None)
                            if size:
                                size_x = float(getattr(size, 'x', 1300000.0)) / 1000000.0  # Convert nm to mm
                                size_y = float(getattr(size, 'y', 1300000.0)) / 1000000.0  # Convert nm to mm
                            
                            # Shape (PSS_CIRCLE=0, PSS_RECTANGLE=1, PSS_OVAL=2, etc.)
                            shape_val = getattr(first_layer, 'shape', 1)
                            if hasattr(shape_val, 'value'):
                                shape = shape_val.value
                            else:
                                shape = int(shape_val)
                        
                        # Get layers from padstack
                        padstack_layers = getattr(padstack, 'layers', [])
                        if padstack_layers:
                            for layer in padstack_layers:
                                layers.append(str(layer))
                    
                    # Fallback: Try direct pad attributes if padstack fails
                    if size_x == 1.3 and size_y == 1.3:  # Still default values
                        size = getattr(p, 'size', None)
                        if size:
                            size_x = float(getattr(size, 'x', 1300000.0)) / 1000000.0
                            size_y = float(getattr(size, 'y', 1300000.0)) / 1000000.0
                    
                    # === COMPREHENSIVE PAD DATA EXTRACTION ===
                    # Extract all pad information from padstack (the authoritative source)
                    pad_data = {
                        'net': net, 
                        'number': num, 
                        'x': x, 
                        'y': y,
                    }
                    
                    # Get comprehensive information from padstack
                    padstack = getattr(p, 'padstack', None)
                    if padstack and hasattr(padstack, 'copper_layers') and len(padstack.copper_layers) > 0:
                        copper_layer = padstack.copper_layers[0]  # Primary copper layer
                        
                        # Shape information (1=circle, 2=square, 3=oval/rectangle)
                        shape = getattr(copper_layer, 'shape', 1)
                        pad_data['shape'] = shape
                        pad_data['shape_name'] = {1: 'circle', 2: 'square', 3: 'oval'}.get(shape, 'circle')
                        
                        # Size information from padstack
                        copper_size = getattr(copper_layer, 'size', None)
                        if copper_size:
                            pad_data['size_x'] = float(getattr(copper_size, 'x', 1300000.0)) / 1000000.0
                            pad_data['size_y'] = float(getattr(copper_size, 'y', 1300000.0)) / 1000000.0
                        else:
                            pad_data['size_x'] = 1.3  # Default fallback
                            pad_data['size_y'] = 1.3
                        
                        # Pad type information
                        pad_data['padstack_type'] = getattr(padstack, 'type', None)
                        pad_data['padstack_mode'] = getattr(padstack, 'mode', None)
                        
                        # Check for different shapes on different layers (multi-layer padstack)
                        if len(padstack.copper_layers) > 1:
                            pad_data['multilayer'] = True
                            pad_data['layer_shapes'] = []
                            for i, layer in enumerate(padstack.copper_layers):
                                layer_info = {
                                    'layer_index': i,
                                    'shape': getattr(layer, 'shape', 1),
                                    'size_x': float(getattr(getattr(layer, 'size', None), 'x', 1300000.0)) / 1000000.0 if hasattr(layer, 'size') else pad_data['size_x'],
                                    'size_y': float(getattr(getattr(layer, 'size', None), 'y', 1300000.0)) / 1000000.0 if hasattr(layer, 'size') else pad_data['size_y'],
                                }
                                pad_data['layer_shapes'].append(layer_info)
                        else:
                            pad_data['multilayer'] = False
                    else:
                        # Fallback to legacy method if no padstack
                        pad_data.update({
                            'shape': shape,  # From previous logic
                            'shape_name': {1: 'circle', 2: 'square', 3: 'oval'}.get(shape, 'circle'),
                            'size_x': size_x,
                            'size_y': size_y,
                            'padstack_type': None,
                            'padstack_mode': None,
                            'multilayer': False,
                        })
                    
                    # Drill information
                    pad_data['drill_diameter'] = drill_diameter
                    
                    # Layer information
                    pad_data['layers'] = layers
                    
                    # Pad classification attempt
                    if drill_diameter > 0:
                        if pad_data.get('padstack_type') == 1:
                            pad_data['pad_type'] = 'through-hole'  # THT
                        else:
                            pad_data['pad_type'] = 'through-hole'  # Default for drilled pads
                    else:
                        pad_data['pad_type'] = 'smd'  # Surface mount if no drill
                    
                    # Special case detection
                    if pad_data['size_x'] > 5.0 or pad_data['size_y'] > 5.0:
                        pad_data['pad_type'] = 'mechanical'  # Large pads likely mechanical/mounting
                    
                    if drill_diameter > 2.0 and (pad_data['size_x'] < 0.1 or pad_data['size_y'] < 0.1):
                        pad_data['pad_type'] = 'npth'  # Non-plated through hole
                    
                    # Debug log first few pads to verify data extraction
                    if i < 3:
                        logger.info(f"Pad {i}: pos=({x:.3f},{y:.3f})mm, size=({size_x:.3f},{size_y:.3f})mm, drill={drill_diameter:.3f}mm, shape={shape}, layers={layers}")
                    
                    pads.append(pad_data)
                except Exception as e:
                    logger.warning(f"Pad parse error #{i}: {e}")
        except Exception as e:
            logger.error(f"Error getting pads: {e}")

        # Tracks - with proper coordinate conversion
        tracks = []
        try:
            trs = _ipc_retry(board.get_tracks, "get_tracks", max_retries=3, sleep_s=0.7)
            for i, tr in enumerate(trs):
                try:
                    start = getattr(tr, 'start', None)
                    end = getattr(tr, 'end', None)
                    s = (float(getattr(start, 'x', 0.0)) / 1000000.0, float(getattr(start, 'y', 0.0)) / 1000000.0) if start else (0.0, 0.0)  # Convert nm to mm
                    e = (float(getattr(end, 'x', 0.0)) / 1000000.0, float(getattr(end, 'y', 0.0)) / 1000000.0) if end else (0.0, 0.0)  # Convert nm to mm
                    tracks.append({'start': {'x': s[0], 'y': s[1]}, 'end': {'x': e[0], 'y': e[1]}})
                except Exception as e:
                    logger.warning(f"Track parse error #{i}: {e}")
        except Exception as e:
            logger.error(f"Error getting tracks: {e}")

        # Vias - with proper coordinate conversion  
        vias = []
        try:
            vs = _ipc_retry(board.get_vias, "get_vias", max_retries=3, sleep_s=0.7)
            for i, v in enumerate(vs):
                try:
                    pos = getattr(v, 'position', None)
                    x = float(getattr(pos, 'x', 0.0)) / 1000000.0 if pos is not None else 0.0  # Convert nm to mm
                    y = float(getattr(pos, 'y', 0.0)) / 1000000.0 if pos is not None else 0.0  # Convert nm to mm
                    
                    # Get via size
                    size = float(getattr(v, 'size', 0.6)) / 1000000.0 if hasattr(v, 'size') else 0.6  # Convert nm to mm, default 0.6mm
                    drill = float(getattr(v, 'drill', 0.3)) / 1000000.0 if hasattr(v, 'drill') else 0.3  # Convert nm to mm, default 0.3mm
                    
                    vias.append({
                        'x': x, 
                        'y': y, 
                        'size': size,
                        'drill': drill
                    })
                except Exception as e:
                    logger.warning(f"Via parse error #{i}: {e}")
        except Exception as e:
            logger.error(f"Error getting vias: {e}")

        # Get copper zones/planes for plane-aware routing AND visualization
        copper_zones = []
        zones_for_visualization = []
        try:
            zones = _ipc_retry(board.get_zones, "get_zones", max_retries=2, sleep_s=0.5)
            for zone in zones:
                try:
                    zone_net = getattr(zone, 'net', None)
                    zone_layer = getattr(zone, 'layer', 'F.Cu')
                    zone_net_name = zone_net.name if zone_net and hasattr(zone_net, 'name') else None
                    
                    # Basic zone info for routing
                    if zone_net_name:
                        copper_zones.append({
                            'net': zone_net_name,
                            'layer': zone_layer,
                            'filled': getattr(zone, 'is_filled', True)
                        })
                        logger.debug(f"Found copper zone: {zone_net_name} on {zone_layer}")
                    
                    # Detailed zone info for visualization
                    zone_outline_points = []
                    
                    # Get outline points
                    outline = getattr(zone, 'outline', None)
                    if outline and hasattr(outline, 'outline'):
                        for point in outline.outline:
                            if hasattr(point, 'location'):
                                loc = point.location
                                if hasattr(loc, 'x') and hasattr(loc, 'y'):
                                    zone_outline_points.append({
                                        'x': float(loc.x) / 1000000.0,  # Convert nm to mm
                                        'y': float(loc.y) / 1000000.0   # Convert nm to mm
                                    })
                    
                    # Extract filled polygons - this is where copper pour data is stored
                    # Based on debug output: filled_polygons is a dict {3: [PolygonWithHoles(...), ...]}
                    zone_filled_polygons = {}
                    
                    try:
                        filled_polygons = getattr(zone, 'filled_polygons', {})
                        logger.debug(f"Zone {zone_net_name}: filled_polygons type = {type(filled_polygons)}")
                        
                        if isinstance(filled_polygons, dict):
                            for layer_id, poly_list in filled_polygons.items():
                                logger.debug(f"Processing layer {layer_id} with {len(poly_list) if isinstance(poly_list, list) else 'non-list'} polygons")
                                layer_polys = []
                                
                                if isinstance(poly_list, list):
                                    for poly_idx, poly in enumerate(poly_list):
                                        logger.debug(f"Processing polygon {poly_idx}, type: {type(poly)}")
                                        
                                        # Handle PolygonWithHoles object
                                        outline = getattr(poly, 'outline', None)
                                        if outline:
                                            nodes = getattr(outline, 'nodes', [])
                                            poly_points = []
                                            
                                            logger.debug(f"Found outline with {len(nodes)} nodes")
                                            
                                            for node in nodes:
                                                point = getattr(node, 'point', None)
                                                if point:
                                                    x_nm = float(getattr(point, 'x', 0.0))
                                                    y_nm = float(getattr(point, 'y', 0.0))
                                                    poly_points.append({
                                                        'x': x_nm / 1000000.0,  # Convert nm to mm
                                                        'y': y_nm / 1000000.0   # Convert nm to mm
                                                    })
                                            
                                            if poly_points:
                                                logger.debug(f"Created polygon with {len(poly_points)} points")
                                                
                                                # Handle holes for thermal relief
                                                holes = []
                                                poly_holes = getattr(poly, 'holes', [])
                                                for hole in poly_holes:
                                                    hole_nodes = getattr(hole, 'nodes', [])
                                                    hole_points = []
                                                    for node in hole_nodes:
                                                        point = getattr(node, 'point', None)
                                                        if point:
                                                            x_nm = float(getattr(point, 'x', 0.0))
                                                            y_nm = float(getattr(point, 'y', 0.0))
                                                            hole_points.append({
                                                                'x': x_nm / 1000000.0,
                                                                'y': y_nm / 1000000.0
                                                            })
                                                    if hole_points:
                                                        holes.append(hole_points)
                                                
                                                layer_polys.append({
                                                    'outline': poly_points,
                                                    'holes': holes
                                                })
                                
                                if layer_polys:
                                    zone_filled_polygons[layer_id] = layer_polys
                                    logger.info(f"Successfully extracted {len(layer_polys)} polygons for layer {layer_id}")
                    
                    except Exception as e:
                        logger.error(f"Zone filled polygon extraction failed: {e}")
                        logger.exception("Full traceback:")
                    
                    # Log zone extraction results
                    if zone_filled_polygons:
                        total_polys = sum(len(polys) for polys in zone_filled_polygons.values())
                        total_points = sum(len(poly['outline']) for layer_polys in zone_filled_polygons.values() for poly in layer_polys)
                        logger.info(f"Zone '{zone_net_name}': {len(zone_filled_polygons)} layers, {total_polys} polygon(s), {total_points} total points")
                    else:
                        logger.debug(f"Zone '{zone_net_name}': No filled polygons found")
                    
                    # Add zone for visualization if it has geometry
                    if zone_outline_points or zone_filled_polygons:
                        zones_for_visualization.append({
                            'net': zone_net_name,
                            'layer': zone_layer,
                            'outline_points': zone_outline_points,
                            'filled_polygons': zone_filled_polygons,
                            'filled': getattr(zone, 'filled', True)  # Changed from is_filled to filled
                        })
                        logger.info(f"Added zone '{zone_net_name}' to visualization list")
                        
                except Exception as e:
                    logger.debug(f"Zone parse error: {e}")
            
            logger.info(f"Found {len(copper_zones)} copper zones/planes, {len(zones_for_visualization)} zones for visualization")
        except Exception as e:
            logger.debug(f"No zones found or error: {e}")

        # Nets (with pins derived from pads)
        nets = []
        try:
            board_nets = _ipc_retry(board.get_nets, "get_nets", max_retries=3, sleep_s=0.7)
            # Group pads by net - coordinates already converted to mm
            pins_by_net: Dict[str, List[Dict]] = {}
            for pad in pads:
                n = pad.get('net')
                if not n or n == "":
                    continue
                pins_by_net.setdefault(n, []).append({'x': pad['x'], 'y': pad['y'], 'layer': 0, 'pad_name': pad.get('number')})
            
            logger.info(f"Found {len(pins_by_net)} nets with pads")
            
            # Create set of nets that have copper planes (should be skipped)
            plane_nets = set(zone['net'] for zone in copper_zones if zone.get('filled', True))
            if plane_nets:
                logger.info(f"🏭 Found nets with copper planes: {', '.join(sorted(plane_nets))}")
            
            # Store all nets for debugging
            all_nets_debug = []
            routable_nets = []
            
            for i, net in enumerate(board_nets):
                try:
                    name = getattr(net, 'name', f'Net_{i}')
                    net_pins = pins_by_net.get(name, [])
                    
                    # Add to debug list
                    all_nets_debug.append({
                        'name': name,
                        'pins': len(net_pins),
                        'has_plane': name in plane_nets,
                        'routable': len(net_pins) >= 2 and name not in plane_nets
                    })
                    
                    # Skip nets with copper planes (already connected via plane)
                    if name in plane_nets:
                        logger.info(f"⚡ Skipping '{name}' - connected via copper plane")
                        continue
                    
                    # Only include nets with 2+ pins for routing
                    if len(net_pins) >= 2:
                        routable_nets.append({
                            'id': i,
                            'name': name,
                            'pins': net_pins,
                            'routed': False,  # Always mark as unrouted initially
                            'priority': 1
                        })
                        logger.debug(f"Added net '{name}' with {len(net_pins)} pins for routing")
                    else:
                        logger.debug(f"Skipped net '{name}' - only {len(net_pins)} pins")
                        
                except Exception as e:
                    logger.warning(f"Net parse error #{i}: {e}")
            
            nets = routable_nets
            logger.info(f"✅ Created {len(nets)} routable nets (excluding {len(plane_nets)} plane-connected nets)")
            
        except Exception as e:
            logger.error(f"Error getting nets: {e}")

        # Compute bounds from geometry (use pads, tracks, vias - all in mm)
        all_x = []
        all_y = []
        
        # Collect pad coordinates (already in mm)
        for p in pads:
            all_x.extend([p['x'] - p['size_x']/2, p['x'] + p['size_x']/2])
            all_y.extend([p['y'] - p['size_y']/2, p['y'] + p['size_y']/2])
            
        # Collect track coordinates (already in mm)
        for track in tracks:
            all_x.extend([track['start_x'], track['end_x']])
            all_y.extend([track['start_y'], track['end_y']])
            
        # Collect via coordinates (already in mm)
        for via in vias:
            all_x.extend([via['x'] - via['size']/2, via['x'] + via['size']/2])
            all_y.extend([via['y'] - via['size']/2, via['y'] + via['size']/2])
            
        # Set bounds with defaults if no geometry found
        min_x = min(all_x) if all_x else 0.0
        min_y = min(all_y) if all_y else 0.0
        max_x = max(all_x) if all_x else 100.0
        max_y = max(all_y) if all_y else 80.0
        
        # Add small margin
        margin = 5.0
        bounds = (min_x - margin, min_y - margin, max_x + margin, max_y + margin)

        board_data = {
            'filename': filename,
            'width': bounds[2] - bounds[0],
            'height': bounds[3] - bounds[1],
            'layers': self._get_layer_count(),
            'nets': nets,
            'components': components,
            'tracks': tracks,
            'vias': vias,
            'pads': pads,
            'bounds': bounds,
            'copper_zones': copper_zones,  # Include copper zone information
            'zones': zones_for_visualization,  # Include zones for visualization with filled polygons
            'all_nets_debug': all_nets_debug,  # For debugging net filtering
            'unrouted_count': len([n for n in nets if not n.get('routed', False)]),
            'routed_count': len([n for n in nets if n.get('routed', False)])
        }

        logger.info(f"Extracted board data: {len(nets)} routable nets, {len(components)} components, {len(tracks)} tracks, {len(copper_zones)} zones")
        return board_data

    def _get_layer_count(self) -> int:
        try:
            return getattr(self.board, 'layer_count', 2)
        except Exception:
            return 2

    # The following are stubs; real creation via IPC will be added later
    def create_track(self, start_x: float, start_y: float, end_x: float, end_y: float,
                     layer: str, width: float, net_name: str) -> bool:
        """Create a straight track on the board via IPC.
        Units: mm for coordinates and width. Layer by name (e.g., 'F.Cu').
        """
        if not self.connected or not self.board:
            logger.warning("create_track called while not connected; ignoring")
            return False
        try:
            # IPC API uses mm coordinates: provide as list of tuples
            coords = [(float(start_x), float(start_y)), (float(end_x), float(end_y))]
            
            # Log the track creation attempt
            logger.debug(f"🔧 Creating track: {net_name} from ({start_x:.3f}, {start_y:.3f}) to ({end_x:.3f}, {end_y:.3f}) on {layer}, width={width:.3f}mm")
            
            # Create track via IPC API
            tr = self.board.add_track(coords, layer=layer or 'F.Cu', width=float(width) or 0.2, net=net_name or None)
            
            if tr:
                logger.info(f"✅ Created track {net_name} {coords} {layer} w={width}")
                return True
            else:
                logger.warning(f"❌ Track creation returned None for {net_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ IPC create_track failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def create_via(self, x: float, y: float, size: float, drill: float,
                   from_layer: str, to_layer: str, net_name: str) -> bool:
        """Create a via via IPC; size/drill in mm, layer pair by names.
        """
        if not self.connected or not self.board:
            logger.warning("create_via called while not connected; ignoring")
            return False
        try:
            layer_pair = (from_layer or 'F.Cu', to_layer or 'B.Cu')
            
            # Log the via creation attempt
            logger.debug(f"🔧 Creating via: {net_name} at ({x:.3f}, {y:.3f}), size={size:.3f}, drill={drill:.3f}, layers={layer_pair}")
            
            # Create via via IPC API
            via = self.board.add_via((float(x), float(y)), layer_pair, size=float(size) or 0.4, drill=float(drill) or 0.2, net=net_name or None)
            
            if via:
                logger.info(f"✅ Created via {net_name} at ({x},{y}) size={size} drill={drill} layers={layer_pair}")
                return True
            else:
                logger.warning(f"❌ Via creation returned None for {net_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ IPC create_via failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def refresh_board(self):
        try:
            # Some IPC implementations require an explicit refresh/commit; attempt if available
            if hasattr(self.board, 'refresh'):
                self.board.refresh()
            elif hasattr(self.client, 'refresh_board'):
                self.client.refresh_board()
        except Exception as e:
            logger.error(f"Error refreshing board: {e}")

    def get_pad_polygons(self) -> List[Dict]:
        """
        Get exact pad polygon shapes using get_pad_shapes_as_polygons().
        Returns list of pad dictionaries with precise polygon geometry.
        """
        if not self.connected or not self.board:
            return []
        
        try:
            # Get all pads
            all_pads = _ipc_retry(self.board.get_pads, "get_pads", max_retries=3, sleep_s=0.7)
            
            # Get exact polygon shapes for all pads on front copper layer
            import kipy
            front_cu_layer = kipy.board_types.BoardLayer.BL_F_Cu
            pad_shapes = self.board.get_pad_shapes_as_polygons(all_pads, front_cu_layer)
            
            pad_data = []
            
            for pad, polygon_shape in zip(all_pads, pad_shapes):
                try:
                    # Get basic pad info
                    pad_pos = pad.position
                    x_mm = pad_pos.x / 1e6  # Convert nanometers to mm
                    y_mm = pad_pos.y / 1e6
                    
                    net_name = pad.net.name if pad.net else "No Net"
                    
                    # Extract drill diameter information
                    drill_diameter = 0.0
                    if hasattr(pad, 'padstack') and pad.padstack:
                        padstack = pad.padstack
                        if hasattr(padstack, 'drill') and padstack.drill:
                            drill = padstack.drill
                            if hasattr(drill, 'diameter'):
                                drill_raw = drill.diameter
                                # Handle Vector2 drill diameter
                                if hasattr(drill_raw, 'x') and hasattr(drill_raw, 'y'):
                                    drill_x = float(drill_raw.x) / 1e6  # Convert nm to mm
                                    drill_y = float(drill_raw.y) / 1e6
                                    drill_diameter = max(drill_x, drill_y)  # Use larger dimension
                                elif isinstance(drill_raw, (int, float)):
                                    drill_diameter = float(drill_raw) / 1e6
                    
                    pad_info = {
                        'x': x_mm,
                        'y': y_mm,
                        'net': net_name,
                        'pad_number': getattr(pad, 'number', ''),
                        'position': (x_mm, y_mm),
                        'drill_diameter': drill_diameter
                    }
                    
                    # Extract exact polygon geometry
                    if polygon_shape is not None and hasattr(polygon_shape, 'outline'):
                        outline = polygon_shape.outline
                        
                        if hasattr(outline, 'nodes'):
                            # Extract all polygon points
                            polygon_points = []
                            for node in outline.nodes:
                                if hasattr(node, 'point'):
                                    point = node.point
                                    point_x_mm = point.x / 1e6
                                    point_y_mm = point.y / 1e6
                                    polygon_points.append((point_x_mm, point_y_mm))
                            
                            pad_info['shape'] = 'polygon'
                            pad_info['polygon_points'] = polygon_points
                            pad_info['point_count'] = len(polygon_points)
                            
                            # Extract holes if present
                            holes = []
                            if hasattr(polygon_shape, 'holes'):
                                for hole in polygon_shape.holes:
                                    if hasattr(hole, 'nodes'):
                                        hole_points = []
                                        for node in hole.nodes:
                                            if hasattr(node, 'point'):
                                                point = node.point
                                                hole_x_mm = point.x / 1e6
                                                hole_y_mm = point.y / 1e6
                                                hole_points.append((hole_x_mm, hole_y_mm))
                                        holes.append(hole_points)
                            
                            pad_info['holes'] = holes
                        
                        else:
                            # Fallback: no nodes available
                            pad_info['shape'] = 'unknown'
                            pad_info['polygon_points'] = []
                            pad_info['holes'] = []
                    
                    else:
                        # No polygon shape on this layer
                        pad_info['shape'] = 'none'
                        pad_info['polygon_points'] = []
                        pad_info['holes'] = []
                    
                    pad_data.append(pad_info)
                    
                except Exception as e:
                    logger.warning(f"Error processing pad polygon: {e}")
                    continue
            
            logger.info(f"Extracted {len(pad_data)} pad polygons")
            
            # Log complexity summary
            shape_counts = {}
            for pad in pad_data:
                point_count = pad.get('point_count', 0)
                if point_count in shape_counts:
                    shape_counts[point_count] += 1
                else:
                    shape_counts[point_count] = 1
            
            logger.info(f"Pad polygon complexity: {shape_counts}")
            
            return pad_data
            
        except Exception as e:
            logger.error(f"Error extracting pad polygons: {e}")
            return []

    def _get_fallback_board_data(self) -> Dict:
        """Fallback mock data if IPC connection fails"""
        return {
            'filename': 'Mock_Board.kicad_pcb',
            'width': 100.0,
            'height': 80.0,
            'layers': 4,
            'nets': [],
            'components': [],
            'tracks': [],
            'vias': [],
            'pads': [],
            'bounds': (0.0, 0.0, 100.0, 80.0),
            'unrouted_count': 0,
            'routed_count': 0
        }
