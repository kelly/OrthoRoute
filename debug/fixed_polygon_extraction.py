#!/usr/bin/env python3
"""
Fixed polygon extraction logic for KiCad pads.
This demonstrates the correct way to extract polygon data from KiCad pads.
"""

def extract_pad_polygons(board, pad):
    """
    Extract polygon shapes from a KiCad pad using the correct API.
    
    Args:
        board: KiCad board object with get_pad_shapes_as_polygons method
        pad: KiCad pad object
        
    Returns:
        dict: Dictionary with layer names as keys and polygon data as values
        Format: {'F.Cu': {'outline': [(x, y), ...], 'holes': [...]}, ...}
    """
    from kipy.board_types import BoardLayer
    
    pad_polygons = {}
    
    # Try both front and back copper layers
    layers_to_check = [
        (BoardLayer.BL_F_Cu, 'F.Cu'),
        (BoardLayer.BL_B_Cu, 'B.Cu')
    ]
    
    for layer_const, layer_name in layers_to_check:
        try:
            # Use correct API signature from KiCad documentation
            polygon_results = board.get_pad_shapes_as_polygons([pad], layer_const)
            
            if polygon_results and len(polygon_results) > 0:
                polygon_shape = polygon_results[0]  # Get first (and only) result
                
                if polygon_shape is not None:
                    polygon_points = []
                    holes = []
                    
                    # Extract outline points from PolygonWithHoles
                    if hasattr(polygon_shape, 'outline'):
                        outline = polygon_shape.outline
                        
                        # Iterate through PolyLineNode objects
                        if hasattr(outline, '__iter__'):
                            try:
                                for point_node in outline:
                                    # Each point_node is a PolyLineNode with .point attribute
                                    if hasattr(point_node, 'point'):
                                        point = point_node.point  # This is a Vector2
                                        if hasattr(point, 'x') and hasattr(point, 'y'):
                                            # Convert from nanometers to millimeters
                                            polygon_points.append((
                                                float(point.x) / 1000000,
                                                float(point.y) / 1000000
                                            ))
                            except Exception as e:
                                print(f"Error extracting outline points: {e}")
                    
                    # Extract holes if present
                    if hasattr(polygon_shape, 'holes'):
                        hole_list = polygon_shape.holes
                        if hasattr(hole_list, '__iter__'):
                            for hole in hole_list:
                                hole_points = []
                                if hasattr(hole, '__iter__'):
                                    for point_node in hole:
                                        if hasattr(point_node, 'point'):
                                            point = point_node.point
                                            if hasattr(point, 'x') and hasattr(point, 'y'):
                                                hole_points.append((
                                                    float(point.x) / 1000000,
                                                    float(point.y) / 1000000
                                                ))
                                if hole_points:
                                    holes.append(hole_points)
                    
                    # Store polygon data if we got points
                    if polygon_points:
                        pad_polygons[layer_name] = {
                            'outline': polygon_points,
                            'holes': holes
                        }
                        
        except Exception as e:
            print(f"Error getting polygon for {layer_name}: {e}")
    
    return pad_polygons


def test_with_actual_extraction():
    """Test the polygon extraction with the corrected logic."""
    import logging
    from kipy import KiCad
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Connecting to KiCad...")
        kicad = KiCad(timeout_ms=15000)
        board = kicad.get_board()
        logger.info(f"✓ Connected to board: {board}")
        
        # Get pads
        pads = board.get_pads()
        logger.info(f"✓ Retrieved {len(pads)} pads")
        
        if len(pads) == 0:
            logger.error("No pads found!")
            return False
        
        # Test on first few pads
        for i, pad in enumerate(pads[:3]):
            logger.info(f"\\n=== Testing pad {i} ===")
            logger.info(f"Pad position: {pad.position}")
            logger.info(f"Pad number: {pad.number}")
            
            # Extract polygons using corrected logic
            polygons = extract_pad_polygons(board, pad)
            
            if polygons:
                logger.info(f"✓ Successfully extracted polygons for pad {i}:")
                for layer, data in polygons.items():
                    outline = data['outline']
                    holes = data['holes']
                    logger.info(f"  {layer}: {len(outline)} outline points, {len(holes)} holes")
                    logger.info(f"    Outline: {outline}")
                    if holes:
                        logger.info(f"    Holes: {holes}")
            else:
                logger.info(f"  No polygons extracted for pad {i}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== Testing Corrected Polygon Extraction ===")
    success = test_with_actual_extraction()
    if success:
        print("🎉 Polygon extraction test completed successfully!")
    else:
        print("❌ Polygon extraction test failed!")
