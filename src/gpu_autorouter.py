#!/usr/bin/env python3
"""
OrthoRoute GPU Autorouter - Main Entry Point
High-performance GPU-accelerated PCB autorouter using KiCad IPC API
"""

import os
import sys
import time
import logging
from pathlib import Path

# Setup logging
log_file = Path.home() / "Documents" / "orthoroute_debug.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for OrthoRoute GPU Autorouter"""
    logger.info("🚀 OrthoRoute GPU Autorouter starting...")
    
    try:
        # Import KiCad IPC API
        from kicad import Client as KiCad
        logger.info("✅ KiCad IPC API imported successfully")
        
        # Connect to KiCad
        kicad = KiCad()
        logger.info("✅ Connected to KiCad via IPC API")
        
        # Get the current board
        board = kicad.get_board()
        if not board:
            logger.error("❌ No board currently loaded in KiCad")
            return 1
            
        logger.info(f"✅ Board loaded: {board.filename}")
        
        # Import GPU routing engine
        from gpu_router import GPUAutorouter
        
        # Initialize GPU autorouter
        router = GPUAutorouter(board)
        
        # Run the routing
        result = router.route_board()
        
        if result.success:
            logger.info("✅ GPU autorouting completed successfully!")
            logger.info(f"Routed {result.tracks_created} tracks")
            logger.info(f"Success rate: {result.success_rate:.1%}")
        else:
            logger.error(f"❌ Autorouting failed: {result.error}")
            return 1
            
        return 0
        
    except ImportError as e:
        logger.error(f"❌ KiCad IPC API not available: {e}")
        logger.error("Make sure 'kicad-python' is installed and KiCad API server is enabled")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.error(f"Traceback:", exc_info=True)
        return 1

if __name__ == "__main__":
    main()
