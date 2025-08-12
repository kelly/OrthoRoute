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
    logger.info(f"Board connected: {board}")
    
    zones = board.get_zones()
    logger.info(f"Found {len(zones)} zones")
    
    for i, zone in enumerate(zones):
        logger.info(f"\n=== Zone {i} Connection Settings ===")
        logger.info(f"Zone net: {zone.net}")
        logger.info(f"Zone layers: {zone.layers}")
        
        # Check zone connection settings
        if hasattr(zone, 'connection') and zone.connection:
            connection = zone.connection
            logger.info(f"Connection type: {connection.zone_connection}")
            
            if hasattr(connection, 'thermal_spokes'):
                thermal = connection.thermal_spokes
                logger.info(f"Thermal spoke width: {getattr(thermal, 'width', 'None')}")
                logger.info(f"Thermal spoke gap: {getattr(thermal, 'gap', 'None')}")
                logger.info(f"Thermal spoke angle: {getattr(thermal, 'angle', 'None')}")
        else:
            logger.info("No connection settings found")
            
        # Check zone fill settings
        logger.info(f"Fill mode: {getattr(zone, 'fill_mode', 'None')}")
        logger.info(f"Min thickness: {getattr(zone, 'min_thickness', 'None')}")
        logger.info(f"Clearance: {getattr(zone, 'clearance', 'None')}")
        
    # Check some pads and their zone connection settings
    pads = board.get_pads()
    logger.info(f"\n=== Checking Pad Zone Connections ===")
    logger.info(f"Found {len(pads)} pads")
    
    gnd_pads = []
    for pad in pads[:10]:  # Check first 10 pads
        if hasattr(pad, 'net') and pad.net and pad.net.name == 'GND':
            gnd_pads.append(pad)
            
    logger.info(f"Found {len(gnd_pads)} GND pads in first 10")
    
    for i, pad in enumerate(gnd_pads[:5]):  # Check first 5 GND pads
        logger.info(f"\nPad {i} (number: {pad.number}):")
        logger.info(f"  Net: {pad.net.name}")
        logger.info(f"  Position: {pad.position}")
        
        # Check padstack zone settings
        if hasattr(pad, 'padstack') and pad.padstack:
            padstack = pad.padstack
            if hasattr(padstack, 'zone_settings'):
                zone_settings = padstack.zone_settings
                logger.info(f"  Zone connection: {getattr(zone_settings, 'zone_connection', 'None')}")
                
                if hasattr(zone_settings, 'thermal_spokes'):
                    thermal = zone_settings.thermal_spokes
                    logger.info(f"  Thermal width: {getattr(thermal, 'width', 'None')}")
                    logger.info(f"  Thermal gap: {getattr(thermal, 'gap', 'None')}")
                    
            # Check individual copper layers
            if hasattr(padstack, 'copper_layers'):
                logger.info(f"  Copper layers: {len(padstack.copper_layers)}")
                for layer in padstack.copper_layers:
                    if hasattr(layer, 'zone_settings'):
                        layer_zone = layer.zone_settings
                        logger.info(f"    Layer {layer.layer}: zone_connection={getattr(layer_zone, 'zone_connection', 'None')}")
                        
except Exception as e:
    logger.error(f"Error: {e}")
    import traceback
    traceback.print_exc()
