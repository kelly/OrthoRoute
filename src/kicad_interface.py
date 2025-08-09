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
                    x = float(getattr(pos, 'x', 0.0)) if pos is not None else 0.0
                    y = float(getattr(pos, 'y', 0.0)) if pos is not None else 0.0
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

        # Pads (used to derive net pins)
        pads = []
        try:
            all_pads = _ipc_retry(board.get_pads, "get_pads", max_retries=3, sleep_s=0.7)
            for i, p in enumerate(all_pads):
                try:
                    pos = getattr(p, 'position', None)
                    x = float(getattr(pos, 'x', 0.0)) if pos is not None else 0.0
                    y = float(getattr(pos, 'y', 0.0)) if pos is not None else 0.0
                    net = getattr(getattr(p, 'net', None), 'name', None)
                    num = getattr(p, 'number', None)
                    pads.append({'net': net, 'number': num, 'x': x, 'y': y})
                except Exception as e:
                    logger.warning(f"Pad parse error #{i}: {e}")
        except Exception as e:
            logger.error(f"Error getting pads: {e}")

        # Tracks
        tracks = []
        try:
            trs = _ipc_retry(board.get_tracks, "get_tracks", max_retries=3, sleep_s=0.7)
            for i, tr in enumerate(trs):
                try:
                    start = getattr(tr, 'start', None)
                    end = getattr(tr, 'end', None)
                    s = (float(getattr(start, 'x', 0.0)), float(getattr(start, 'y', 0.0))) if start else (0.0, 0.0)
                    e = (float(getattr(end, 'x', 0.0)), float(getattr(end, 'y', 0.0))) if end else (0.0, 0.0)
                    tracks.append({'start': {'x': s[0], 'y': s[1]}, 'end': {'x': e[0], 'y': e[1]}})
                except Exception as e:
                    logger.warning(f"Track parse error #{i}: {e}")
        except Exception as e:
            logger.error(f"Error getting tracks: {e}")

        # Vias
        vias = []
        try:
            vs = _ipc_retry(board.get_vias, "get_vias", max_retries=3, sleep_s=0.7)
            for i, v in enumerate(vs):
                try:
                    pos = getattr(v, 'position', None)
                    x = float(getattr(pos, 'x', 0.0)) if pos is not None else 0.0
                    y = float(getattr(pos, 'y', 0.0)) if pos is not None else 0.0
                    vias.append({'x': x, 'y': y})
                except Exception as e:
                    logger.warning(f"Via parse error #{i}: {e}")
        except Exception as e:
            logger.error(f"Error getting vias: {e}")

        # Get copper zones/planes for plane-aware routing
        copper_zones = []
        try:
            zones = _ipc_retry(board.get_zones, "get_zones", max_retries=2, sleep_s=0.5)
            for zone in zones:
                try:
                    zone_net = getattr(zone, 'net', None)
                    zone_layer = getattr(zone, 'layer', 'F.Cu')
                    if zone_net and hasattr(zone_net, 'name'):
                        zone_net_name = zone_net.name
                        copper_zones.append({
                            'net': zone_net_name,
                            'layer': zone_layer,
                            'filled': getattr(zone, 'is_filled', True)
                        })
                        logger.debug(f"Found copper zone: {zone_net_name} on {zone_layer}")
                except Exception as e:
                    logger.debug(f"Zone parse error: {e}")
            
            logger.info(f"Found {len(copper_zones)} copper zones/planes")
        except Exception as e:
            logger.debug(f"No zones found or error: {e}")

        # Nets (with pins derived from pads)
        nets = []
        try:
            board_nets = _ipc_retry(board.get_nets, "get_nets", max_retries=3, sleep_s=0.7)
            # Group pads by net
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

        # Compute bounds from geometry
        min_x = min((p['x'] for p in pads), default=0.0)
        min_y = min((p['y'] for p in pads), default=0.0)
        max_x = max((p['x'] for p in pads), default=100.0)
        max_y = max((p['y'] for p in pads), default=80.0)
        # Include component centers
        if components:
            min_x = min(min_x, min(c['x'] for c in components))
            min_y = min(min_y, min(c['y'] for c in components))
            max_x = max(max_x, max(c['x'] for c in components))
            max_y = max(max_y, max(c['y'] for c in components))
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
