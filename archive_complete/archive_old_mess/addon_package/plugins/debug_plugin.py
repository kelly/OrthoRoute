#!/usr/bin/env python3
"""
OrthoRoute Debug Plugin - Comprehensive logging to identify crash cause
"""

import os
import sys
import json
import tempfile
import time
import subprocess
import traceback
from pathlib import Path
import logging

# Set up detailed logging
log_file = Path(tempfile.gettempdir()) / "orthoroute_debug.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_file)),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print(f"🔍 Debug log will be written to: {log_file}")

# Try to import KiCad - with detailed error reporting
try:
    logger.info("Attempting to import KiCad IPC API...")
    from kipy import KiCad
    from kipy.board import Board
    from kipy.util.units import to_mm
    logger.info("✅ KiCad IPC API imported successfully")
    KIPY_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ KiCad IPC API import failed: {e}")
    try:
        logger.info("Attempting to import legacy SWIG API...")
        import pcbnew
        logger.info("✅ Legacy SWIG API imported successfully")
        KIPY_AVAILABLE = False
    except ImportError as e2:
        logger.error(f"❌ Both APIs failed: IPC={e}, SWIG={e2}")
        # Don't call sys.exit(1) - this kills KiCad entirely!
        KIPY_AVAILABLE = False

class OrthoRouteDebugPlugin:
    """Debug version of OrthoRoute plugin with comprehensive logging"""
    
    def __init__(self):
        logger.info("=== OrthoRoute Debug Plugin Initializing ===")
        self.kicad = None
        self.board = None
        self.work_dir = None
        
    def __enter__(self):
        logger.info("Context manager: Entering plugin context")
        try:
            if KIPY_AVAILABLE:
                logger.info("Connecting to KiCad via IPC API...")
                self.kicad = KiCad()
                logger.info("✅ KiCad IPC connection established")
            else:
                logger.info("Using legacy SWIG API...")
                self.kicad = None  # Will use pcbnew directly
                logger.info("✅ Legacy API ready")
            return self
        except Exception as e:
            logger.error(f"❌ Failed to connect to KiCad: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Context manager: Exiting plugin context")
        if exc_type:
            logger.error(f"Exception in context: {exc_type.__name__}: {exc_val}")
            logger.error(traceback.format_exc())
        
        try:
            self.cleanup()
            logger.info("✅ Cleanup completed successfully")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            logger.error(traceback.format_exc())
    
    def extract_board_data(self):
        """Extract minimal board data with extensive logging"""
        logger.info("=== Starting Board Data Extraction ===")
        
        try:
            if KIPY_AVAILABLE:
                logger.info("Using IPC API to get board...")
                board = self.kicad.get_board()
                self.board = board
                logger.info(f"✅ Board obtained: {board.title if hasattr(board, 'title') else 'Unknown'}")
            else:
                logger.info("Using SWIG API to get board...")
                board = pcbnew.GetBoard()
                self.board = board
                logger.info(f"✅ Board obtained: {board.GetFileName() if board else 'No board'}")
            
            # Extract very minimal data for testing
            board_data = {
                'timestamp': time.time(),
                'api_type': 'IPC' if KIPY_AVAILABLE else 'SWIG',
                'board_available': self.board is not None,
                'extraction_success': True
            }
            
            logger.info(f"✅ Board data extracted: {board_data}")
            return board_data
            
        except Exception as e:
            logger.error(f"❌ Board extraction failed: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def save_board_data(self, board_data):
        """Save board data to work directory"""
        logger.info("=== Saving Board Data ===")
        
        try:
            data_file = self.work_dir / "routing_request.json"
            logger.info(f"Saving to: {data_file}")
            
            with open(data_file, 'w') as f:
                json.dump(board_data, f, indent=2)
            
            logger.info(f"✅ Board data saved ({data_file.stat().st_size} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save board data: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def test_minimal_operation(self):
        """Test minimal operation without launching subprocess"""
        logger.info("=== Testing Minimal Operation ===")
        
        try:
            # Create work directory
            self.work_dir = Path(tempfile.mkdtemp(prefix='orthoroute_debug_'))
            logger.info(f"Work directory: {self.work_dir}")
            
            # Extract board data
            board_data = self.extract_board_data()
            if not board_data:
                logger.error("❌ Board data extraction failed")
                return False
            
            # Save data
            if not self.save_board_data(board_data):
                logger.error("❌ Board data save failed")
                return False
            
            # Create fake results to test import
            fake_results = {
                'nets': [
                    {
                        'name': 'TEST_NET',
                        'success': True,
                        'path': [
                            {'x': 1000000, 'y': 1000000, 'layer': 0},
                            {'x': 2000000, 'y': 2000000, 'layer': 0}
                        ]
                    }
                ]
            }
            
            result_file = self.work_dir / "routing_result.json"
            with open(result_file, 'w') as f:
                json.dump(fake_results, f, indent=2)
            logger.info(f"✅ Fake results created: {result_file}")
            
            # Test result import
            logger.info("Testing result import...")
            import_success = self.test_import_results()
            
            if import_success:
                logger.info("✅ Minimal test completed successfully!")
                return True
            else:
                logger.error("❌ Import test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Minimal test failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def test_import_results(self):
        """Test importing fake results"""
        logger.info("=== Testing Results Import ===")
        
        try:
            result_file = self.work_dir / "routing_result.json"
            
            if not result_file.exists():
                logger.error(f"❌ Results file not found: {result_file}")
                return False
            
            logger.info(f"Loading results from: {result_file}")
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Results loaded: {len(results.get('nets', []))} nets")
            
            # Just validate results without actually importing to board
            for net in results.get('nets', []):
                logger.info(f"  Net: {net.get('name')} - Success: {net.get('success')} - Path points: {len(net.get('path', []))}")
            
            logger.info("✅ Results validation completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Results import test failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("=== Cleanup ===")
        
        try:
            if self.work_dir and self.work_dir.exists():
                import shutil
                shutil.rmtree(self.work_dir)
                logger.info(f"✅ Work directory cleaned: {self.work_dir}")
            
            if KIPY_AVAILABLE and self.kicad:
                # Try to disconnect gracefully
                try:
                    # Note: This might not be the correct method name
                    if hasattr(self.kicad, 'disconnect'):
                        self.kicad.disconnect()
                        logger.info("✅ KiCad IPC disconnected")
                except:
                    logger.info("ℹ️ KiCad disconnect not needed or failed (normal)")
                    
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

def run_debug_test():
    """Run the debug test"""
    logger.info("🚀 Starting OrthoRoute Debug Test")
    
    try:
        with OrthoRouteDebugPlugin() as plugin:
            logger.info("Plugin context established")
            
            success = plugin.test_minimal_operation()
            
            if success:
                logger.info("🎉 DEBUG TEST PASSED - Plugin works without crashing!")
                print("✅ Debug test completed successfully!")
                print(f"📋 Full log available at: {log_file}")
                return True
            else:
                logger.error("❌ DEBUG TEST FAILED")
                print("❌ Debug test failed - check log for details")
                print(f"📋 Full log available at: {log_file}")
                return False
                
    except Exception as e:
        logger.error(f"💥 DEBUG TEST CRASHED: {e}")
        logger.error(traceback.format_exc())
        print(f"💥 Debug test crashed: {e}")
        print(f"📋 Full log available at: {log_file}")
        return False

if __name__ == "__main__":
    run_debug_test()
