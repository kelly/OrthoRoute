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
    
    # Check all pads for different connection types
    pads = board.get_pads()
    logger.info(f"Found {len(pads)} pads")
    
    connection_types = {}
    layer_connection_types = {}
    
    for i, pad in enumerate(pads):
        if hasattr(pad, 'net') and pad.net and pad.net.name == 'GND':
            if hasattr(pad, 'padstack') and pad.padstack:
                padstack = pad.padstack
                if hasattr(padstack, 'zone_settings'):
                    zone_conn = padstack.zone_settings.zone_connection
                    connection_types[zone_conn] = connection_types.get(zone_conn, 0) + 1
                    
                # Check layer-specific connections
                if hasattr(padstack, 'copper_layers'):
                    for layer in padstack.copper_layers:
                        if hasattr(layer, 'zone_settings'):
                            layer_conn = layer.zone_settings.zone_connection
                            layer_key = f"Layer_{layer.layer}_{layer_conn}"
                            layer_connection_types[layer_key] = layer_connection_types.get(layer_key, 0) + 1
    
    logger.info(f"\nPad Zone Connection Types:")
    for conn_type, count in connection_types.items():
        logger.info(f"  Type {conn_type}: {count} pads")
        
    logger.info(f"\nLayer Zone Connection Types:")
    for layer_conn, count in layer_connection_types.items():
        logger.info(f"  {layer_conn}: {count} pads")
    
    # Let's try to change a pad to use thermal relief and refill
    logger.info(f"\n=== Trying to enable thermal relief on a pad ===")
    
    # Find a GND pad
    gnd_pad = None
    for pad in pads:
        if hasattr(pad, 'net') and pad.net and pad.net.name == 'GND':
            gnd_pad = pad
            break
    
    if gnd_pad:
        logger.info(f"Found GND pad: {gnd_pad.number} at {gnd_pad.position}")
        
        # Try to modify the pad's zone connection to thermal relief
        if hasattr(gnd_pad, 'padstack') and gnd_pad.padstack:
            padstack = gnd_pad.padstack
            if hasattr(padstack, 'zone_settings'):
                logger.info(f"Original zone connection: {padstack.zone_settings.zone_connection}")
                
                # Try to set to thermal relief (type 3)
                try:
                    padstack.zone_settings.zone_connection = 3
                    logger.info(f"Changed zone connection to: {padstack.zone_settings.zone_connection}")
                    
                    # Update the pad on the board
                    board.update_items([gnd_pad])
                    logger.info("Updated pad on board")
                    
                    # Force refill
                    logger.info("Forcing zone refill...")
                    board.refill_zones(block=True)
                    logger.info("Zone refill complete!")
                    
                    # Check if holes appeared
                    zones = board.get_zones()
                    for zone in zones:
                        filled_polygons = zone.filled_polygons
                        for layer_id, polygon_list in filled_polygons.items():
                            for j, polygon in enumerate(polygon_list):
                                holes_count = 0
                                if hasattr(polygon, 'holes') and polygon.holes:
                                    holes_count = len(polygon.holes)
                                logger.info(f"Zone layer {layer_id}, polygon {j}: {holes_count} holes")
                                
                except Exception as e:
                    logger.error(f"Error changing pad connection: {e}")
    
except Exception as e:
    logger.error(f"Error: {e}")
    import traceback
    traceback.print_exc()
