"""
OrthoRoute KiCad Plugin - Full Version with Fixed Imports
"""

import os
import sys
import json
import tempfile

# Add current directory to path
plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

try:
    import pcbnew
    import wx
    
    # Import our modules with proper error handling
    try:
        # Try to import ui_dialogs (should work now with stderr fix)
        import ui_dialogs
        ui_dialogs_available = True
    except Exception as e:
        print(f"Warning: Could not import ui_dialogs: {e}")
        ui_dialogs_available = False
        # Create a minimal dialog class as fallback
        class MinimalConfigDialog:
            def __init__(self, parent):
                self.config = {'grid_pitch_mm': 0.1, 'enable_visualization': False}
            def ShowModal(self):
                return wx.ID_OK
            def get_config(self):
                return self.config
    
    try:
        import board_export
        board_export_available = True
    except Exception as e:
        print(f"Warning: Could not import board_export: {e}")
        board_export_available = False
    
    try:
        import route_import
        route_import_available = True
    except Exception as e:
        print(f"Warning: Could not import route_import: {e}")
        route_import_available = False
    
    class OrthoRouteKiCadPlugin(pcbnew.ActionPlugin):
        def defaults(self):
            self.name = "OrthoRoute GPU Autorouter"
            self.category = "Route"
            self.description = "GPU-accelerated PCB autorouter using CuPy"
            
            # Set icon if available
            try:
                icon_path = os.path.join(plugin_dir, "icon.png")
                if os.path.exists(icon_path):
                    self.icon_file_name = icon_path
                self.show_toolbar_button = True
            except:
                self.show_toolbar_button = True
        
        def Run(self):
            """Main plugin execution"""
            try:
                # Get the current board
                board = pcbnew.GetBoard()
                if not board:
                    wx.MessageBox("No board found. Please open a PCB file first.", 
                                "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
                    return
                
                # Show configuration dialog
                if ui_dialogs_available:
                    config_dialog = ui_dialogs.OrthoRouteConfigDialog(None)
                else:
                    config_dialog = MinimalConfigDialog(None)
                    
                if config_dialog.ShowModal() == wx.ID_OK:
                    config = config_dialog.get_config()
                    
                    # Show progress dialog
                    progress_dialog = wx.ProgressDialog(
                        "OrthoRoute GPU Autorouter",
                        "Initializing...",
                        100,
                        style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE
                    )
                    
                    try:
                        # Export board data
                        progress_dialog.Update(10, "Exporting board data...")
                        
                        if board_export_available:
                            board_data = board_export.export_board_to_orthoroute(board)
                        else:
                            # Fallback - create minimal board data
                            board_data = self._create_minimal_board_data(board)
                        
                        # Initialize routing engine
                        progress_dialog.Update(30, "Initializing GPU engine...")
                        
                        # Create a simple test to verify GPU engine availability
                        try:
                            # Try to import from the installed package first
                            try:
                                from orthoroute.gpu_engine import OrthoRouteEngine
                                print("DEBUG: Imported OrthoRouteEngine from installed package")
                            except ImportError:
                                # Fallback - try to import from local path
                                sys.path.insert(0, os.path.join(plugin_dir, '..', 'orthoroute'))
                                from gpu_engine import OrthoRouteEngine
                                print("DEBUG: Imported OrthoRouteEngine from local path")
                            
                            engine = OrthoRouteEngine()
                            
                            # Enable visualization if requested
                            if config.get('enable_visualization', False):
                                engine.enable_visualization({'enabled': True})
                            
                            progress_dialog.Update(50, "Starting routing...")
                            
                            # Route the board
                            result = engine.route(board_data, config)
                            
                            progress_dialog.Update(80, "Importing routes...")
                            
                            # Import results back to KiCad
                            if route_import_available and result.get('success', False):
                                route_import.import_routes_to_kicad(board, result)
                                success_msg = f"Routing completed successfully!\n\n"
                                success_msg += f"Routed: {result['stats']['successful_nets']}/{result['stats']['total_nets']} nets\n"
                                success_msg += f"Success rate: {result['stats']['success_rate']:.1f}%\n"
                                success_msg += f"Time: {result['stats']['total_time_seconds']:.1f} seconds"
                                wx.MessageBox(success_msg, "OrthoRoute Success", wx.OK | wx.ICON_INFORMATION)
                            else:
                                error_msg = result.get('error', 'Unknown routing error')
                                wx.MessageBox(f"Routing failed: {error_msg}", "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
                        
                        except ImportError as e:
                            # GPU engine not available - show informative message
                            wx.MessageBox(
                                "OrthoRoute GPU engine is not available.\n\n"
                                "This may be because:\n"
                                "• CuPy is not installed\n"
                                "• CUDA is not available\n"
                                "• OrthoRoute modules are not properly installed\n\n"
                                f"Error: {e}",
                                "OrthoRoute - GPU Engine Not Available",
                                wx.OK | wx.ICON_WARNING
                            )
                        
                    finally:
                        progress_dialog.Destroy()
                        
            except Exception as e:
                wx.MessageBox(f"Plugin error: {str(e)}", "OrthoRoute Error", wx.OK | wx.ICON_ERROR)
                
        def _create_minimal_board_data(self, board):
            """Create minimal board data when board_export is not available"""
            # Get board bounding box
            bbox = board.GetBoardEdgesBoundingBox()
            
            return {
                'bounds': {
                    'width_nm': int(bbox.GetWidth()),
                    'height_nm': int(bbox.GetHeight()),
                    'layers': board.GetCopperLayerCount()
                },
                'nets': [],  # Would need to extract nets from board
                'grid': {
                    'pitch_nm': 100000  # 0.1mm default
                }
            }
    
    # Register the plugin
    plugin_instance = OrthoRouteKiCadPlugin()
    plugin_instance.register()
    
    # Write success to debug file
    debug_file = os.path.join(plugin_dir, "full_plugin_debug.txt")
    with open(debug_file, "w") as f:
        f.write("Full OrthoRoute plugin loaded successfully!\n")
        f.write(f"Plugin name: {plugin_instance.name}\n")
        f.write(f"Plugin category: {plugin_instance.category}\n")
        f.write(f"Plugin directory: {plugin_dir}\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"ui_dialogs available: {ui_dialogs_available}\n")
        f.write(f"board_export available: {board_export_available}\n")
        f.write(f"route_import available: {route_import_available}\n")
    
    print("Full OrthoRoute plugin registered successfully!")
    
except Exception as e:
    # Write error to debug file
    debug_file = os.path.join(plugin_dir, "full_plugin_debug.txt")
    try:
        with open(debug_file, "w") as f:
            f.write(f"Error loading full plugin: {e}\n")
            import traceback
            f.write(f"Traceback: {traceback.format_exc()}\n")
    except:
        pass
    print(f"Error loading full plugin: {e}")
