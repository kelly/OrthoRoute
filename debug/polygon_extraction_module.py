#!/usr/bin/env python3
"""
Polygon extraction function that can be integrated into orthoroute.py
"""

def extract_pad_polygon_data(board, pad, pad_index=None):
    """
    Extract polygon shapes from a KiCad pad using the correct API.
    This function returns the data in the format expected by orthoroute.py
    
    Args:
        board: KiCad board object with get_pad_shapes_as_polygons method
        pad: KiCad pad object
        pad_index: Optional index for logging (for debugging first few pads)
        
    Returns:
        dict: Dictionary with layer names as keys and polygon data as values
        Format: {'F.Cu': {'outline': [{'x': float, 'y': float}, ...], 'holes': [...]}, ...}
    """
    from kipy.board_types import BoardLayer
    import logging
    
    logger = logging.getLogger(__name__)
    pad_polygons = {}
    
    # Try both front and back copper layers using correct BoardLayer constants
    layers_to_check = [
        (BoardLayer.BL_F_Cu, 'F.Cu'),
        (BoardLayer.BL_B_Cu, 'B.Cu')
    ]
    
    for layer_const, layer_name in layers_to_check:
        try:
            # Use correct API: get_pad_shapes_as_polygons(pads: Sequence[Pad], layer: int = BoardLayer.BL_F_Cu)
            polygon_results = board.get_pad_shapes_as_polygons([pad], layer_const)
            
            if polygon_results and len(polygon_results) > 0:
                polygon_shape = polygon_results[0]  # Get first (and only) result
                
                if polygon_shape is not None:
                    polygon_points = []
                    holes = []
                    
                    if pad_index is not None and pad_index < 3:
                        logger.info(f"  Got polygon shape for {layer_name}: {type(polygon_shape)}")
                    
                    # Extract outline points from PolygonWithHoles
                    if hasattr(polygon_shape, 'outline'):
                        outline = polygon_shape.outline
                        
                        # Iterate through PolyLineNode objects in the outline
                        if hasattr(outline, '__iter__'):
                            try:
                                for point_node in outline:
                                    # Each point_node is a PolyLineNode with .point attribute
                                    if hasattr(point_node, 'point'):
                                        point = point_node.point  # This is a Vector2
                                        if hasattr(point, 'x') and hasattr(point, 'y'):
                                            # Convert from nanometers to millimeters
                                            polygon_points.append({
                                                'x': float(point.x) / 1000000,
                                                'y': float(point.y) / 1000000
                                            })
                                
                                if pad_index is not None and pad_index < 3:
                                    logger.info(f"  Extracted {len(polygon_points)} outline points for {layer_name}")
                                    
                            except Exception as e:
                                if pad_index is not None and pad_index < 3:
                                    logger.info(f"  Error extracting outline points for {layer_name}: {e}")
                    
                    # Extract holes if present
                    if hasattr(polygon_shape, 'holes'):
                        hole_list = polygon_shape.holes
                        if hasattr(hole_list, '__iter__'):
                            try:
                                for hole in hole_list:
                                    hole_points = []
                                    if hasattr(hole, '__iter__'):
                                        for point_node in hole:
                                            if hasattr(point_node, 'point'):
                                                point = point_node.point
                                                if hasattr(point, 'x') and hasattr(point, 'y'):
                                                    hole_points.append({
                                                        'x': float(point.x) / 1000000,
                                                        'y': float(point.y) / 1000000
                                                    })
                                    if hole_points:
                                        holes.append(hole_points)
                                        
                                if pad_index is not None and pad_index < 3:
                                    logger.info(f"  Extracted {len(holes)} holes for {layer_name}")
                                    
                            except Exception as e:
                                if pad_index is not None and pad_index < 3:
                                    logger.info(f"  Error extracting holes for {layer_name}: {e}")
                    
                    # Store polygon data if we got points
                    if polygon_points:
                        pad_polygons[layer_name] = {
                            'outline': polygon_points,
                            'holes': holes
                        }
                        
                        if pad_index is not None and pad_index < 3:
                            logger.info(f"  Successfully extracted exact {layer_name} polygon: {len(polygon_points)} points, {len(holes)} holes")
                    else:
                        if pad_index is not None and pad_index < 3:
                            logger.info(f"  No polygon points extracted for {layer_name}")
                else:
                    if pad_index is not None and pad_index < 3:
                        logger.info(f"  Polygon shape is None for {layer_name}")
            else:
                if pad_index is not None and pad_index < 3:
                    logger.info(f"  No polygon results for {layer_name}")
        
        except Exception as e:
            if pad_index is not None and pad_index < 3:
                logger.info(f"  Error getting polygon for {layer_name}: {e}")
    
    return pad_polygons


# Integration code for orthoroute.py
def integrate_polygon_extraction_code():
    """
    Returns the code snippet that should replace the broken polygon extraction in orthoroute.py
    """
    return '''
                    # Debug pad structure for first few pads
                    if i < 3:
                        logger.info(f"  Attempting polygon extraction for pad {i}...")
                    
                    # Check if board has the polygon extraction method
                    if hasattr(board, 'get_pad_shapes_as_polygons'):
                        if i < 3:
                            logger.info(f"  Board has get_pad_shapes_as_polygons method")
                        
                        # Extract polygons using corrected logic
                        pad_polygons = extract_pad_polygon_data(board, pad, i)
                        
                        if pad_polygons:
                            if i < 3:
                                logger.info(f"  Successfully extracted polygon data: {list(pad_polygons.keys())}")
                        else:
                            if i < 3:
                                logger.info(f"  No polygon data extracted")
                    else:
                        if i < 3:
                            logger.info(f"  Board does not have get_pad_shapes_as_polygons method")
'''

if __name__ == "__main__":
    print("Polygon extraction function ready for integration!")
    print("Use extract_pad_polygon_data(board, pad, pad_index) in orthoroute.py")
    print("\\nIntegration code:")
    print(integrate_polygon_extraction_code())
