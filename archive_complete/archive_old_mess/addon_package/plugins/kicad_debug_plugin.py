#!/usr/bin/env python3
"""
OrthoRoute KiCad Debug Plugin - For testing within KiCad
This plugin will run inside KiCad and provide comprehensive debugging.
"""

import os
import sys
import json
import tempfile
import time
import traceback
from pathlib import Path

# Import KiCad SWIG API (always available in KiCad)
import pcbnew
import wx

class OrthoRouteKiCadDebugPlugin(pcbnew.ActionPlugin):
    """Debug plugin that runs inside KiCad with comprehensive logging"""
    
    def defaults(self):
        self.name = "OrthoRoute Debug"
        self.category = "Debug"
        self.description = "Debug version of OrthoRoute with comprehensive logging"
        self.show_toolbar_button = True
        self.icon_file_name = os.path.join(os.path.dirname(__file__), "icon.png")

    def Run(self):
        """Main plugin execution with detailed debugging"""
        print("=" * 80)
        print("🔍 OrthoRoute Debug Plugin Starting...")
        print("=" * 80)
        
        # Set up logging to file
        log_file = Path(tempfile.gettempdir()) / "orthoroute_kicad_debug.log"
        print(f"📋 Debug log: {log_file}")
        
        try:
            with open(log_file, 'w') as log:
                log.write("=== OrthoRoute KiCad Debug Log ===\n")
                log.write(f"Timestamp: {time.asctime()}\n")
                log.write(f"Python version: {sys.version}\n")
                log.write(f"KiCad plugin directory: {os.path.dirname(__file__)}\n\n")
                
                # Test 1: Basic KiCad Access
                log.write("=== TEST 1: Basic KiCad Access ===\n")
                print("TEST 1: Basic KiCad Access")
                
                try:
                    board = pcbnew.GetBoard()
                    if board:
                        filename = board.GetFileName()
                        log.write(f"✅ Board obtained: {filename}\n")
                        print(f"✅ Board: {filename}")
                        
                        # Get basic board info
                        size = board.GetBoardEdgesBoundingBox()
                        layer_count = board.GetCopperLayerCount()
                        net_count = board.GetNetInfo().GetNetCount()
                        
                        log.write(f"  - Size: {size.GetWidth()//1000000}x{size.GetHeight()//1000000}mm\n")
                        log.write(f"  - Layers: {layer_count}\n")
                        log.write(f"  - Nets: {net_count}\n")
                        
                        print(f"  - Size: {size.GetWidth()//1000000}x{size.GetHeight()//1000000}mm")
                        print(f"  - Layers: {layer_count}")
                        print(f"  - Nets: {net_count}")
                        
                    else:
                        log.write("❌ No board available\n")
                        print("❌ No board available")
                        
                except Exception as e:
                    log.write(f"❌ Board access failed: {e}\n")
                    log.write(f"Traceback: {traceback.format_exc()}\n")
                    print(f"❌ Board access failed: {e}")
                
                # Test 2: Work Directory Creation
                log.write("\n=== TEST 2: Work Directory Creation ===\n")
                print("TEST 2: Work Directory Creation")
                
                try:
                    work_dir = Path(tempfile.mkdtemp(prefix='orthoroute_test_'))
                    log.write(f"✅ Work directory created: {work_dir}\n")
                    print(f"✅ Work directory: {work_dir}")
                    
                    # Test file creation
                    test_file = work_dir / "test.json"
                    test_data = {"test": True, "timestamp": time.time()}
                    
                    with open(test_file, 'w') as f:
                        json.dump(test_data, f, indent=2)
                    
                    log.write(f"✅ Test file created: {test_file} ({test_file.stat().st_size} bytes)\n")
                    print(f"✅ Test file created: {test_file.name}")
                    
                except Exception as e:
                    log.write(f"❌ Work directory test failed: {e}\n")
                    log.write(f"Traceback: {traceback.format_exc()}\n")
                    print(f"❌ Work directory test failed: {e}")
                
                # Test 3: Server Launcher Check
                log.write("\n=== TEST 3: Server Launcher Check ===\n")
                print("TEST 3: Server Launcher Check")
                
                try:
                    plugin_dir = Path(__file__).parent
                    server_launcher = plugin_dir / "server_launcher.py"
                    standalone_server = plugin_dir / "orthoroute_standalone_server.py"
                    
                    log.write(f"Plugin directory: {plugin_dir}\n")
                    log.write(f"Server launcher exists: {server_launcher.exists()}\n")
                    log.write(f"Standalone server exists: {standalone_server.exists()}\n")
                    
                    print(f"Plugin dir: {plugin_dir}")
                    print(f"Server launcher: {'✅' if server_launcher.exists() else '❌'}")
                    print(f"Standalone server: {'✅' if standalone_server.exists() else '❌'}")
                    
                    if server_launcher.exists():
                        size = server_launcher.stat().st_size
                        log.write(f"Server launcher size: {size} bytes\n")
                        print(f"Launcher size: {size} bytes")
                    
                    if standalone_server.exists():
                        size = standalone_server.stat().st_size
                        log.write(f"Standalone server size: {size} bytes\n")
                        print(f"Server size: {size} bytes")
                        
                except Exception as e:
                    log.write(f"❌ Server check failed: {e}\n")
                    log.write(f"Traceback: {traceback.format_exc()}\n")
                    print(f"❌ Server check failed: {e}")
                
                # Test 4: Minimal Board Data Extraction
                log.write("\n=== TEST 4: Minimal Board Data Extraction ===\n")
                print("TEST 4: Minimal Board Data Extraction")
                
                try:
                    board = pcbnew.GetBoard()
                    if board:
                        # Extract minimal data
                        board_data = {
                            'filename': board.GetFileName(),
                            'layer_count': board.GetCopperLayerCount(),
                            'net_count': board.GetNetInfo().GetNetCount(),
                            'timestamp': time.time()
                        }
                        
                        # Save to work directory
                        data_file = work_dir / "board_data.json"
                        with open(data_file, 'w') as f:
                            json.dump(board_data, f, indent=2)
                        
                        log.write(f"✅ Board data extracted and saved: {data_file}\n")
                        log.write(f"Data: {board_data}\n")
                        print(f"✅ Board data extracted: {len(board_data)} fields")
                        
                    else:
                        log.write("❌ No board for data extraction\n")
                        print("❌ No board for data extraction")
                        
                except Exception as e:
                    log.write(f"❌ Board data extraction failed: {e}\n")
                    log.write(f"Traceback: {traceback.format_exc()}\n")
                    print(f"❌ Board data extraction failed: {e}")
                
                # Test 5: Simulate Adding a Track
                log.write("\n=== TEST 5: Simulate Adding a Track ===\n")
                print("TEST 5: Simulate Adding a Track")
                
                try:
                    board = pcbnew.GetBoard()
                    if board:
                        # Create a simple track
                        track = pcbnew.PCB_TRACK(board)
                        track.SetStart(pcbnew.VECTOR2I(10000000, 10000000))  # 10mm, 10mm
                        track.SetEnd(pcbnew.VECTOR2I(20000000, 20000000))    # 20mm, 20mm
                        track.SetWidth(200000)  # 0.2mm
                        track.SetLayer(pcbnew.F_Cu)
                        
                        # Add to board
                        board.Add(track)
                        
                        log.write("✅ Test track added to board\n")
                        print("✅ Test track added")
                        
                        # Try to refresh display
                        pcbnew.Refresh()
                        log.write("✅ Display refreshed\n")
                        print("✅ Display refreshed")
                        
                        # Remove the test track to clean up
                        board.Remove(track)
                        log.write("✅ Test track removed (cleanup)\n")
                        print("✅ Test track cleaned up")
                        
                    else:
                        log.write("❌ No board for track test\n")
                        print("❌ No board for track test")
                        
                except Exception as e:
                    log.write(f"❌ Track test failed: {e}\n")
                    log.write(f"Traceback: {traceback.format_exc()}\n")
                    print(f"❌ Track test failed: {e}")
                
                # Cleanup
                log.write("\n=== CLEANUP ===\n")
                try:
                    if 'work_dir' in locals():
                        import shutil
                        shutil.rmtree(work_dir)
                        log.write(f"✅ Work directory cleaned: {work_dir}\n")
                        print("✅ Cleanup completed")
                except Exception as e:
                    log.write(f"❌ Cleanup failed: {e}\n")
                    print(f"❌ Cleanup failed: {e}")
                
                log.write("\n=== DEBUG TEST COMPLETED ===\n")
                
        except Exception as e:
            error_msg = f"💥 Debug plugin crashed: {e}\n{traceback.format_exc()}"
            print(error_msg)
            try:
                with open(log_file, 'a') as log:
                    log.write(f"\n💥 PLUGIN CRASH: {error_msg}\n")
            except:
                pass
            
            # Show error dialog
            wx.MessageBox(f"Debug plugin crashed:\n{e}\n\nCheck log file: {log_file}", 
                         "Debug Plugin Error", wx.OK | wx.ICON_ERROR)
            return
        
        # Show completion dialog
        wx.MessageBox(f"Debug test completed!\n\nCheck the log file for details:\n{log_file}", 
                     "Debug Test Complete", wx.OK | wx.ICON_INFORMATION)
        
        print("=" * 80)
        print("🎉 Debug test completed successfully!")
        print(f"📋 Full log: {log_file}")
        print("=" * 80)
