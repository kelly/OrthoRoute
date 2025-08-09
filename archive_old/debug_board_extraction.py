#!/usr/bin/env python3
"""
Debug script to test board data extraction
"""

import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from kicad_interface import KiCadInterface

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_board_extraction():
    """Test what happens when we extract board data"""
    logger.info("🔍 Testing KiCad board data extraction...")
    
    # Create interface
    kicad = KiCadInterface()
    
    # Try to connect
    logger.info("Attempting to connect to KiCad...")
    try:
        success = kicad.connect()
        logger.info(f"Connection result: {success}")
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return
    
    if not kicad.connected:
        logger.error("❌ Not connected to KiCad - this is likely the problem!")
        logger.error("Make sure:")
        logger.error("  1. KiCad PCB Editor is open")
        logger.error("  2. A PCB file is loaded")
        logger.error("  3. The plugin is being run from within KiCad")
        return
    
    # Get board data
    logger.info("Extracting board data...")
    try:
        board_data = kicad.get_board_data()
        
        # Debug the extracted data
        logger.info(f"📊 BOARD DATA EXTRACTED:")
        logger.info(f"  Filename: {board_data.get('filename', 'Unknown')}")
        logger.info(f"  Dimensions: {board_data.get('width', 0):.1f} x {board_data.get('height', 0):.1f} mm")
        logger.info(f"  Layers: {board_data.get('layers', 0)}")
        logger.info(f"  Components: {len(board_data.get('components', []))}")
        logger.info(f"  Pads: {len(board_data.get('pads', []))}")
        logger.info(f"  Tracks: {len(board_data.get('tracks', []))}")
        logger.info(f"  Vias: {len(board_data.get('vias', []))}")
        
        # Most importantly - nets!
        nets = board_data.get('nets', [])
        logger.info(f"  🎯 TOTAL NETS: {len(nets)}")
        
        if len(nets) == 0:
            logger.error("❌ NO NETS FOUND - This is why routing completes instantly!")
            logger.error("Possible causes:")
            logger.error("  1. Board has no electrical connections defined")
            logger.error("  2. KiCad IPC API call failed")
            logger.error("  3. Nets exist but have no pads/pins")
        else:
            logger.info(f"✅ Found {len(nets)} nets:")
            for i, net in enumerate(nets[:5]):  # Show first 5
                pins = net.get('pins', [])
                routed = net.get('routed', False)
                logger.info(f"    Net {i+1}: '{net.get('name', 'Unknown')}' - {len(pins)} pins - {'ROUTED' if routed else 'UNROUTED'}")
                
            # Count unrouted nets
            unrouted = [n for n in nets if not n.get('routed', False)]
            logger.info(f"  🔄 UNROUTED NETS: {len(unrouted)} (these would be processed for routing)")
            
            if len(unrouted) == 0:
                logger.warning("⚠️  All nets are marked as ROUTED - this is why routing completes instantly!")
            
            # Check for nets with insufficient pins
            bad_nets = [n for n in unrouted if len(n.get('pins', [])) < 2]
            if bad_nets:
                logger.warning(f"⚠️  {len(bad_nets)} nets have less than 2 pins and can't be routed")
        
    except Exception as e:
        logger.error(f"❌ Board data extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_board_extraction()
