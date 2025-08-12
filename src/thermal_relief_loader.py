#!/usr/bin/env python3
"""
KiCad Thermal Relief Data Loader
Extracts complete board data including thermal reliefs, exact pad shapes, and components
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def load_kicad_thermal_relief_data(kicad_interface) -> Optional[Dict[str, Any]]:
    """
    Load complete board data from KiCad including thermal reliefs and exact pad shapes.
    
    Args:
        kicad_interface: Connected KiCad interface object
        
    Returns:
        Dictionary containing complete board data with thermal reliefs, or None if failed
    """
    logger.info("Loading thermal relief data from KiCad...")
    
    try:
        board = kicad_interface.board
        if not board:
            logger.error("Could not get board from KiCad interface")
            return None

        # Get basic board data
        zones = board.get_zones()
        pads = board.get_pads()
        vias = board.get_vias()
        tracks = board.get_tracks()
        footprints = board.get_footprints()
        
        logger.info(f"Found {len(zones)} zones, {len(pads)} pads, {len(vias)} vias")
        logger.info(f"Found {len(tracks)} tracks, {len(footprints)} footprints")
        
        # Get board filename using the dedicated method
        board_filename = "Unknown"
        try:
            # First try the dedicated filename method if available
            if hasattr(kicad_interface, 'get_board_filename'):
                board_filename = kicad_interface.get_board_filename()
            
            # Fallback to direct board access if needed
            if board_filename == "Unknown":
                # Try multiple methods to get the filename from the board
                if hasattr(board, 'filename') and board.filename:
                    board_filename = board.filename
                elif hasattr(board, 'GetFileName') and board.GetFileName():
                    board_filename = board.GetFileName()
                elif hasattr(board, 'get_filename') and board.get_filename():
                    board_filename = board.get_filename()
                # Try accessing the underlying board object if available
                elif hasattr(board, '_board') and hasattr(board._board, 'GetFileName'):
                    board_filename = board._board.GetFileName()
                # Try calling the KiCad native API if accessible
                elif hasattr(board, 'board') and hasattr(board.board, 'GetFileName'):
                    board_filename = board.board.GetFileName()
                # Final attempt: check if board has direct access to pcbnew board
                elif hasattr(board, 'GetBoard') and hasattr(board.GetBoard(), 'GetFileName'):
                    pcb_board = board.GetBoard()
                    board_filename = pcb_board.GetFileName()
            
            # Clean up the filename if we got one
            if board_filename and board_filename != "Unknown":
                import os
                board_filename = os.path.basename(board_filename)
                logger.info(f"Board filename: {board_filename}")
            else:
                logger.warning("Could not retrieve board filename")
                
        except Exception as e:
            logger.warning(f"Error getting board filename: {e}")
            board_filename = "Unknown"
        
        # Build the data structure that the GUI expects
        board_data = {
            'board_name': board_filename if board_filename != "Unknown" else 'PCB Board',
            'filename': board_filename,
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

        # Process pads with actual polygon shapes
        logger.info(f"Starting to process {len(pads)} pads...")
        
        # Get pad shapes as polygons for both front and back copper
        try:
            front_pad_shapes = board.get_pad_shapes_as_polygons(pads, 3)  # Front copper (layer 3)
            back_pad_shapes = board.get_pad_shapes_as_polygons(pads, 34)  # Back copper (layer 34)
            logger.info(f"Got pad polygon shapes: {len(front_pad_shapes)} front, {len(back_pad_shapes)} back")
        except Exception as e:
            logger.error(f"Error getting pad polygon shapes: {e}")
            front_pad_shapes = [None] * len(pads)
            back_pad_shapes = [None] * len(pads)
        
        for i, pad in enumerate(pads):
            try:
                pos = pad.position
                padstack = pad.padstack
                
                # Get size from first copper layer as fallback
                size_x = size_y = 1.0  # Default values
                drill_diameter = 0.0
                
                if padstack and padstack.copper_layers:
                    copper_layer = padstack.copper_layers[0]
                    size = copper_layer.size
                    size_x = size.x / 1000000.0  # Convert to mm
                    size_y = size.y / 1000000.0
                
                # Get drill diameter
                drill_diameter = 0.0
                drill = padstack.drill
                if drill and drill.diameter:
                    drill_diameter = drill.diameter.x / 1000000.0
                
                # Determine actual layers this pad exists on by examining the pad itself
                actual_pad_layers = set()
                
                # Check if pad has layer information directly
                try:
                    # PRIORITY: For through-hole pads, always assign to both layers
                    if drill_diameter > 0:
                        actual_pad_layers.update(['F.Cu', 'B.Cu'])
                        if i < 3:  # Debug logging for first few pads
                            logger.info(f"  Pad {i} is through-hole (drill={drill_diameter:.2f}mm) -> both layers")
                    else:
                        # For SMD pads, determine actual layer from pad/padstack data
                        
                        # Method 1: Check if pad has layer set property
                        if hasattr(pad, 'layers') and pad.layers:
                            for layer_id in pad.layers:
                                if layer_id == 3:  # F.Cu
                                    actual_pad_layers.add('F.Cu')
                                elif layer_id == 34:  # B.Cu
                                    actual_pad_layers.add('B.Cu')
                        
                        # Method 2: Check padstack copper layers
                        elif padstack and padstack.copper_layers:
                            for copper_layer in padstack.copper_layers:
                                # Check if this copper layer corresponds to F.Cu or B.Cu
                                layer_id = None
                                if hasattr(copper_layer, 'layer'):
                                    layer_id = copper_layer.layer
                                elif hasattr(copper_layer, 'layer_id'):
                                    layer_id = copper_layer.layer_id
                                # Also check if copper_layer itself is the layer id
                                elif isinstance(copper_layer, int):
                                    layer_id = copper_layer
                                
                                if layer_id == 3:  # F.Cu
                                    actual_pad_layers.add('F.Cu')
                                elif layer_id == 34:  # B.Cu
                                    actual_pad_layers.add('B.Cu')
                                
                                if i < 3:  # Additional debug
                                    logger.info(f"    copper layer analysis: {type(copper_layer)}, layer_id={layer_id}")
                        
                        # Method 3: Fallback for SMD pads - default to F.Cu
                        else:
                            actual_pad_layers.add('F.Cu')
                    
                    if i < 3:  # Debug logging for first few pads
                        layer_debug = f"layers={list(actual_pad_layers)}" if actual_pad_layers else "no layers detected"
                        logger.info(f"  Pad {i} layer detection: {layer_debug}")
                        
                        # Additional debug information
                        if hasattr(pad, 'layers'):
                            logger.info(f"    pad.layers: {list(pad.layers) if pad.layers else 'None'}")
                        if padstack and padstack.copper_layers:
                            layer_info = []
                            for cl in padstack.copper_layers:
                                layer_id = getattr(cl, 'layer', None) or getattr(cl, 'layer_id', None)
                                layer_info.append(f"layer_{layer_id}")
                            logger.info(f"    padstack copper layers: {layer_info}")
                    
                except Exception as e:
                    logger.error(f"Error determining pad layers for pad {i}: {e}")
                    # Fallback to previous behavior
                    if drill_diameter > 0:
                        actual_pad_layers.update(['F.Cu', 'B.Cu'])
                    else:
                        actual_pad_layers.add('F.Cu')
                
                # Now only get polygon shapes for layers where the pad actually exists
                polygons = {}
                
                # Process front copper polygon - only if pad is actually on F.Cu
                if ('F.Cu' in actual_pad_layers and i < len(front_pad_shapes) and 
                    front_pad_shapes[i] is not None and hasattr(front_pad_shapes[i], 'outline') and 
                    len(front_pad_shapes[i].outline) > 0):
                    front_polygon = front_pad_shapes[i]
                    front_outline = []
                    for poly_node in front_polygon.outline:
                        point = poly_node.point  # Get Vector2 from PolyLineNode
                        front_outline.append({
                            'x': point.x / 1000000.0,  # Convert to mm
                            'y': point.y / 1000000.0
                        })
                    
                    front_holes = []
                    for hole in front_polygon.holes:
                        hole_points = []
                        for poly_node in hole:
                            point = poly_node.point  # Get Vector2 from PolyLineNode
                            hole_points.append({
                                'x': point.x / 1000000.0,
                                'y': point.y / 1000000.0
                            })
                        front_holes.append(hole_points)
                    
                    # Calculate polygon area to check if it's meaningful
                    if len(front_outline) >= 3:
                        # Simple area calculation using shoelace formula
                        area = 0
                        for j in range(len(front_outline)):
                            j1 = (j + 1) % len(front_outline)
                            area += front_outline[j]['x'] * front_outline[j1]['y']
                            area -= front_outline[j1]['x'] * front_outline[j]['y']
                        area = abs(area) / 2.0
                        
                        # Only add if we have actual polygon data with meaningful area (> 0.001 mm²)
                        if area > 0.001:
                            polygons['F.Cu'] = {
                                'outline': front_outline,
                                'holes': front_holes
                            }
                            
                            if i < 3:
                                logger.info(f"  Front copper polygon: {len(front_outline)} outline points, {len(front_holes)} holes, area: {area:.6f} mm²")
                        else:
                            if i < 3:
                                logger.info(f"  Front copper polygon: SKIPPED (area too small: {area:.6f} mm²)")
                
                # Process back copper polygon - only if pad is actually on B.Cu
                if ('B.Cu' in actual_pad_layers and i < len(back_pad_shapes) and 
                    back_pad_shapes[i] is not None and hasattr(back_pad_shapes[i], 'outline') and 
                    len(back_pad_shapes[i].outline) > 0):
                    back_polygon = back_pad_shapes[i]
                    back_outline = []
                    for poly_node in back_polygon.outline:
                        point = poly_node.point  # Get Vector2 from PolyLineNode
                        back_outline.append({
                            'x': point.x / 1000000.0,  # Convert to mm
                            'y': point.y / 1000000.0
                        })
                    
                    back_holes = []
                    for hole in back_polygon.holes:
                        hole_points = []
                        for poly_node in hole:
                            point = poly_node.point  # Get Vector2 from PolyLineNode
                            hole_points.append({
                                'x': point.x / 1000000.0,
                                'y': point.y / 1000000.0
                            })
                        back_holes.append(hole_points)
                    
                    # Calculate polygon area to check if it's meaningful
                    if len(back_outline) >= 3:
                        # Simple area calculation using shoelace formula
                        area = 0
                        for j in range(len(back_outline)):
                            j1 = (j + 1) % len(back_outline)
                            area += back_outline[j]['x'] * back_outline[j1]['y']
                            area -= back_outline[j1]['x'] * back_outline[j]['y']
                        area = abs(area) / 2.0
                        
                        # Only add if we have actual polygon data with meaningful area (> 0.001 mm²)
                        if area > 0.001:
                            polygons['B.Cu'] = {
                                'outline': back_outline,
                                'holes': back_holes
                            }
                            
                            if i < 3:
                                logger.info(f"  Back copper polygon: {len(back_outline)} outline points, {len(back_holes)} holes, area: {area:.6f} mm²")
                        else:
                            if i < 3:
                                logger.info(f"  Back copper polygon: SKIPPED (area too small: {area:.6f} mm²)")
                
                pad_data = {
                    'x': pos.x / 1000000.0,  # Convert to mm
                    'y': pos.y / 1000000.0,
                    'size_x': size_x,
                    'size_y': size_y,
                    'shape': int(pad.pad_type),
                    'footprint_layer': 'F.Cu',
                    'drill_diameter': drill_diameter,
                    'number': str(pad.number),
                    'net': pad.net if pad.net else None,  # Store the complete net object
                    'net_name': str(pad.net.name) if pad.net else 'unconnected',  # Keep name for display
                    'polygons': polygons  # Add exact polygon shapes from KiCad
                }
                board_data['pads'].append(pad_data)
                if i < 3:  # Log first few for debugging
                    polygon_info = f", polygons: {list(polygons.keys())}" if polygons else ""
                    drill_info = f", drill: {drill_diameter:.2f}mm" if drill_diameter > 0 else " (SMD)"
                    pad_type = "Through-hole" if drill_diameter > 0 else "Surface-mount"
                    logger.info(f"Processed pad {i}: pos=({pad_data['x']:.2f}, {pad_data['y']:.2f}), size=({size_x:.2f}x{size_y:.2f}){drill_info} [{pad_type}]{polygon_info}")
                    
            except Exception as e:
                logger.error(f"Error processing pad {i}: {e}")
        
        logger.info(f"Finished processing pads. Total in board_data: {len(board_data['pads'])}")
        
        # Process vias
        logger.info(f"Starting to process {len(vias)} vias...")
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
                if i < 3:  # Log first few for debugging
                    logger.info(f"Processed via {i}: {via_data}")
                    
            except Exception as e:
                logger.error(f"Error processing via {i}: {e}")
        
        logger.info(f"Finished processing vias. Total in board_data: {len(board_data['vias'])}")
        
        # Process tracks  
        logger.info(f"Starting to process {len(tracks)} tracks...")
        for i, track in enumerate(tracks):
            try:
                start = track.start
                end = track.end
                
                track_data = {
                    'start_x': start.x / 1000000.0,
                    'start_y': start.y / 1000000.0,
                    'end_x': end.x / 1000000.0,
                    'end_y': end.y / 1000000.0,
                    'width': track.width / 1000000.0,
                    'layer': track.layer
                }
                board_data['tracks'].append(track_data)
                if i < 3:  # Log first few for debugging
                    logger.info(f"Processed track {i}: {track_data}")
                    
            except Exception as e:
                logger.error(f"Error processing track {i}: {e}")
        
        logger.info(f"Finished processing tracks. Total in board_data: {len(board_data['tracks'])}")
        
        # Process components (footprints) - determine layers based on footprint type and pad analysis
        logger.info(f"Starting to process {len(footprints)} footprints...")
        for i, footprint in enumerate(footprints):
            try:
                pos = footprint.position
                
                # Get reference and value from fields
                reference = "Unknown"
                value = "Unknown"
                
                try:
                    ref_field = footprint.reference_field
                    if ref_field and ref_field.text:
                        reference = ref_field.text.value
                except:
                    pass
                    
                try:
                    val_field = footprint.value_field
                    if val_field and val_field.text:
                        value = val_field.text.value
                except:
                    pass
                
                # Determine footprint mount type and layers
                component_layers = set()
                mount_type = "Unknown"
                
                try:
                    # Method 1: Check footprint attributes for mount type
                    footprint_is_smd = False
                    footprint_is_tht = False
                    
                    # Check if footprint has attributes method
                    if hasattr(footprint, 'attributes') and footprint.attributes:
                        # Check for SMD attribute
                        if hasattr(footprint.attributes, 'smd') and footprint.attributes.smd:
                            footprint_is_smd = True
                            mount_type = "SMD"
                        # Check for through hole attribute  
                        elif hasattr(footprint.attributes, 'through_hole') and footprint.attributes.through_hole:
                            footprint_is_tht = True
                            mount_type = "Through-hole"
                    
                    # Method 2: Check by examining pads belonging to this footprint
                    footprint_pos = footprint.position
                    footprint_pads = []
                    has_drilled_pads = False
                    
                    # Find pads that belong to this footprint (within 10mm of footprint center)
                    for pad_idx, pad_data in enumerate(board_data['pads']):
                        distance = ((footprint_pos.x/1000000.0 - pad_data['x'])**2 + (footprint_pos.y/1000000.0 - pad_data['y'])**2)**0.5
                        if distance < 10.0:  # 10mm tolerance
                            footprint_pads.append(pad_data)
                            if pad_data.get('drill_diameter', 0) > 0:
                                has_drilled_pads = True
                    
                    # Method 3: Determine mount type from pad analysis if attributes didn't work
                    if mount_type == "Unknown":
                        if has_drilled_pads:
                            footprint_is_tht = True
                            mount_type = "Through-hole"
                        else:
                            footprint_is_smd = True  
                            mount_type = "SMD"
                    
                    # Determine layers based on mount type
                    if footprint_is_tht:
                        # Through-hole components appear on both front and back copper
                        component_layers.update(['F.Cu', 'B.Cu'])
                    elif footprint_is_smd:
                        # SMD components - need to determine which layer
                        # Check which layer the pads actually exist on
                        pad_layers = set()
                        for pad_data in footprint_pads:
                            if 'polygons' in pad_data:
                                pad_layers.update(pad_data['polygons'].keys())
                        
                        if pad_layers:
                            component_layers.update(pad_layers)
                        else:
                            # Default to F.Cu for SMD if no pad layer info
                            component_layers.add('F.Cu')
                    
                    # If no layers determined, default to F.Cu
                    if not component_layers:
                        component_layers.add('F.Cu')
                        
                    layer_info = list(component_layers)
                    
                    if i < 3:  # Debug logging for first few components
                        logger.info(f"  Component {i} ({reference}): mount_type={mount_type}, drilled_pads={has_drilled_pads}, layers={layer_info}")
                    
                except Exception as e:
                    logger.error(f"Error determining component layers for footprint {i}: {e}")
                    component_layers = {'F.Cu'}
                    layer_info = ['F.Cu']
                    mount_type = "Unknown"
                
                component_data = {
                    'reference': reference,
                    'value': value,
                    'x': pos.x / 1000000.0,
                    'y': pos.y / 1000000.0,
                    'rotation': footprint.orientation,
                    'layers': layer_info,  # Array of layers this component appears on
                    'mount_type': mount_type,  # SMD, Through-hole, or Unknown
                    'footprint': str(footprint.id) if hasattr(footprint, 'id') else 'unknown'
                }
                board_data['components'].append(component_data)
                if i < 3:  # Log first few for debugging
                    logger.info(f"Processed component {i}: {component_data}")
                    
            except Exception as e:
                logger.error(f"Error processing component {i}: {e}")
        
        logger.info(f"Finished processing components. Total in board_data: {len(board_data['components'])}")
        
        # Process zones with thermal reliefs - THIS IS THE KEY PART
        for i, zone in enumerate(zones):
            try:
                filled_polygons = zone.filled_polygons
                logger.info(f"Zone {i}: net={zone.net}, layers={zone.layers}")
                
                zone_data = {
                    'net': str(zone.net),
                    'layers': list(zone.layers),
                    'filled_polygons': {}
                }
                
                for layer_id, polygon_list in filled_polygons.items():
                    logger.info(f"  Layer {layer_id}: {len(polygon_list)} polygons")
                    layer_polygons = []
                    
                    for j, polygon in enumerate(polygon_list):
                        outline = polygon.outline
                        holes = polygon.holes
                        
                        logger.info(f"    Polygon {j}: {len(outline)} outline points, {len(holes)} holes")
                        
                        # Convert outline points to our format
                        outline_points = []
                        for point in outline:
                            # Handle PolyLineNode structure properly
                            if hasattr(point, 'point'):
                                actual_point = point.point  # Get Vector2 from PolyLineNode
                                outline_points.append({
                                    'x': actual_point.x / 1000000.0,  # Convert to mm
                                    'y': actual_point.y / 1000000.0
                                })
                            else:
                                # Direct point access
                                outline_points.append({
                                    'x': point.x / 1000000.0,  # Convert to mm
                                    'y': point.y / 1000000.0
                                })
                        
                        # Convert hole points to our format
                        hole_data = []
                        for hole in holes:
                            hole_points = []
                            for point in hole:
                                # Handle PolyLineNode structure properly
                                if hasattr(point, 'point'):
                                    actual_point = point.point  # Get Vector2 from PolyLineNode
                                    hole_points.append({
                                        'x': actual_point.x / 1000000.0,
                                        'y': actual_point.y / 1000000.0
                                    })
                                else:
                                    # Direct point access
                                    hole_points.append({
                                        'x': point.x / 1000000.0,
                                        'y': point.y / 1000000.0
                                    })
                            hole_data.append(hole_points)
                        
                        polygon_data = {
                            'outline': outline_points,
                            'holes': hole_data
                        }
                        
                        layer_polygons.append(polygon_data)
                        logger.info(f"     Added layer {layer_id} polygon with {len(outline_points)} points, {len(hole_data)} holes")
                    
                    zone_data['filled_polygons'][layer_id] = layer_polygons
                
                board_data['copper_pours'].append(zone_data)
                logger.info(f"  Added copper pour for zone {i} with layers: {list(zone.layers)}")
                
            except Exception as e:
                logger.error(f"Error processing zone {i}: {e}")
        
        # Calculate board bounds
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        for pad in board_data['pads']:
            x, y = pad.get('x', 0), pad.get('y', 0)
            min_x, max_x = min(min_x, x), max(x, max_x)
            min_y, max_y = min(min_y, y), max(y, max_y)
        
        for comp in board_data['components']:
            x, y = comp.get('x', 0), comp.get('y', 0)
            min_x, max_x = min(min_x, x), max(x, max_x)
            min_y, max_y = min(min_y, y), max(y, max_y)
        
        # Add some padding
        padding = 5.0
        board_data['bounds'] = (min_x - padding, min_y - padding, max_x + padding, max_y + padding)
        logger.info(f"Board bounds: {board_data['bounds']}")
        
        # Extract nets for airwires
        logger.info("Extracting nets for airwires...")
        try:
            nets = board.get_nets()
            logger.info(f"Found {len(nets)} nets in board")
            
            for net in nets:
                net_name = net.name
                net_code = net.code
                
                # Skip special nets
                if net_name in ['', 'unconnected', 'No Net']:
                    continue
                
                # Count pads connected to this net
                connected_pads = []
                for pad in board_data['pads']:
                    pad_net = pad.get('net')
                    if pad_net and hasattr(pad_net, 'code') and pad_net.code == net_code:
                        connected_pads.append(pad)
                    elif isinstance(pad_net, dict) and pad_net.get('code') == net_code:
                        connected_pads.append(pad)
                
                if len(connected_pads) >= 2:  # Only nets with 2+ pads need airwires
                    board_data['nets'][net_name] = {
                        'net_code': net_code,
                        'pin_count': len(connected_pads),
                        'routed': False  # Assume unrouted for now (could check if tracks exist)
                    }
                    logger.debug(f"  Net '{net_name}' (code {net_code}): {len(connected_pads)} pads")
            
            logger.info(f"Extracted {len(board_data['nets'])} nets with 2+ pads for airwires")
            
        except Exception as e:
            logger.error(f"Error extracting nets: {e}")
        
        # Log success summary
        components = len(board_data['components'])
        pads = len(board_data['pads'])
        vias = len(board_data['vias'])
        tracks = len(board_data['tracks'])
        copper_pours_count = len(board_data['copper_pours'])
        total_points = 0
        
        # Calculate total points safely
        for pour in board_data['copper_pours']:
            filled_polygons = pour.get('filled_polygons', {})
            for layer_polygons in filled_polygons.values():
                for polygon in layer_polygons:
                    outline = polygon.get('outline', [])
                    total_points += len(outline)
        
        logger.info(f"Successfully loaded KiCad thermal relief data:")
        logger.info(f"   {components} components, {pads} pads, {vias} vias")
        logger.info(f"   {tracks} tracks, {copper_pours_count} copper pours with {total_points} total outline points")
        logger.info(f"   Thermal reliefs are embedded in the complex polygon outlines!")
        
        return board_data
        
    except Exception as e:
        logger.error(f"Error loading thermal relief data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None