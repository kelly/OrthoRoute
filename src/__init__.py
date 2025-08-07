#!/usr/bin/env python3
"""
OrthoRoute - KiCad GPU-Accelerated Autorouter Plugin
Modern IPC API implementation for KiCad 9.0+
"""

import os
import sys
import traceback
import logging
from pathlib import Path

# Set up logging
def setup_logging():
    log_dir = Path.home() / "Documents" / "kicad_logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "orthoroute.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Try KiCad API imports
try:
    # Modern IPC API (KiCad 9.0+)
    from kicad import Client as KiCadIPC
    IPC_AVAILABLE = True
    logger.info("✅ KiCad IPC API available")
except ImportError:
    IPC_AVAILABLE = False
    logger.warning("⚠️ KiCad IPC API not available")

try:
    # Legacy SWIG API (fallback)
    import pcbnew
    import wx
    SWIG_AVAILABLE = True
    logger.info("✅ KiCad SWIG API available")
except ImportError:
    SWIG_AVAILABLE = False
    logger.warning("⚠️ KiCad SWIG API not available")

# Import our routing engine
try:
    from .routing_engine import OrthoRouteEngine
    ENGINE_AVAILABLE = True
except ImportError:
    try:
        from routing_engine import OrthoRouteEngine
        ENGINE_AVAILABLE = True
    except ImportError:
        ENGINE_AVAILABLE = False
        logger.error("❌ OrthoRoute engine not available")

class OrthoRoutePlugin:
    """
    OrthoRoute GPU Autorouter Plugin
    Supports both IPC API (modern) and SWIG API (legacy) for maximum compatibility
    """
    
    def __init__(self):
        self.name = "OrthoRoute GPU Autorouter"
        self.description = "High-performance GPU-accelerated PCB autorouter"
        self.logger = logger
        
    def run(self):
        """Main plugin entry point"""
        self.logger.info("🚀 OrthoRoute GPU Autorouter starting...")
        
        try:
            # Check system requirements
            if not self._check_requirements():
                return
            
            # Get board using available API
            board = self._get_board()
            if not board:
                self._show_error("No board loaded. Please open a PCB file first.")
                return
            
            # Show configuration dialog
            config = self._show_config_dialog()
            if not config:
                return  # User cancelled
            
            # Initialize routing engine
            engine = OrthoRouteEngine()
            
            # Extract board data
            board_data = self._extract_board_data(board)
            if not board_data or not board_data.get('nets'):
                self._show_warning("No nets found to route!")
                return
            
            self.logger.info(f"Found {len(board_data['nets'])} nets to route")
            
            # Run routing
            results = engine.route(board_data, config)
            
            # Show results
            self._show_results(results)
            
        except Exception as e:
            error_msg = f"OrthoRoute failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self._show_error(error_msg)
    
    def _check_requirements(self):
        """Check if all requirements are available"""
        if not ENGINE_AVAILABLE:
            self._show_error("OrthoRoute engine not available. Please check installation.")
            return False
        
        if not IPC_AVAILABLE and not SWIG_AVAILABLE:
            self._show_error("No KiCad API available. Please check your KiCad installation.")
            return False
        
        return True
    
    def _get_board(self):
        """Get current board using available API"""
        if IPC_AVAILABLE:
            try:
                client = KiCadIPC()
                return client.get_board()
            except Exception as e:
                self.logger.warning(f"IPC API failed: {e}")
        
        if SWIG_AVAILABLE:
            try:
                return pcbnew.GetBoard()
            except Exception as e:
                self.logger.warning(f"SWIG API failed: {e}")
        
        return None
    
    def _extract_board_data(self, board):
        """Extract board data for routing"""
        # This would be implemented based on the API being used
        # For now, return placeholder data
        return {
            'bounds': {'width': 100, 'height': 80},
            'nets': [],  # Extract actual nets
            'obstacles': []  # Extract existing tracks/components
        }
    
    def _show_config_dialog(self):
        """Show routing configuration dialog"""
        if SWIG_AVAILABLE:
            return self._show_wx_config_dialog()
        else:
            # Fallback to simple config
            return {
                'grid_pitch': 0.1,
                'max_iterations': 5,
                'use_gpu': True
            }
    
    def _show_wx_config_dialog(self):
        """Show wxPython configuration dialog"""
        # Implement configuration dialog
        # For now, return default config
        return {
            'grid_pitch': 0.1,
            'max_iterations': 5,
            'use_gpu': True
        }
    
    def _show_results(self, results):
        """Show routing results"""
        if results.get('success'):
            message = f"Routing completed!\nNets routed: {results.get('nets_routed', 0)}"
            self._show_info(message)
        else:
            error = results.get('error', 'Unknown error')
            self._show_error(f"Routing failed: {error}")
    
    def _show_error(self, message):
        """Show error message"""
        self.logger.error(message)
        if SWIG_AVAILABLE:
            wx.MessageBox(message, "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
        else:
            print(f"ERROR: {message}")
    
    def _show_warning(self, message):
        """Show warning message"""
        self.logger.warning(message)
        if SWIG_AVAILABLE:
            wx.MessageBox(message, "OrthoRoute Warning", wx.OK | wx.ICON_WARNING)
        else:
            print(f"WARNING: {message}")
    
    def _show_info(self, message):
        """Show info message"""
        self.logger.info(message)
        if SWIG_AVAILABLE:
            wx.MessageBox(message, "OrthoRoute", wx.OK | wx.ICON_INFORMATION)
        else:
            print(f"INFO: {message}")

# Legacy SWIG plugin wrapper
if SWIG_AVAILABLE:
    class OrthoRouteSWIGPlugin(pcbnew.ActionPlugin):
        """SWIG API wrapper for legacy KiCad versions"""
        
        def defaults(self):
            self.name = "OrthoRoute GPU Autorouter"
            self.category = "Routing"
            self.description = "GPU-accelerated PCB autorouter"
            self.show_toolbar_button = True
            
            # Set icon if available
            icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
            if os.path.exists(icon_path):
                self.icon_file_name = icon_path
        
        def Run(self):
            """SWIG plugin entry point"""
            plugin = OrthoRoutePlugin()
            plugin.run()
    
    # Register SWIG plugin
    OrthoRouteSWIGPlugin().register()

# IPC API entry point
def main():
    """IPC API entry point"""
    plugin = OrthoRoutePlugin()
    plugin.run()

if __name__ == "__main__":
    main()
