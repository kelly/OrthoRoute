#!/usr/bin/env python3
"""
OrthoRoute Headless Debug Script
===============================
Runs the OrthoRoute plugin without GUI to capture detailed debug output
"""

import sys
import os
import traceback
import datetime

def setup_paths():
    """Setup Python paths for KiCad and plugin access"""
    print("🔧 Setting up Python paths...")
    
    # Add plugin path
    plugin_path = os.path.join(os.path.dirname(__file__), "addon_package", "plugins")
    if plugin_path not in sys.path:
        sys.path.insert(0, plugin_path)
        print(f"✅ Added plugin path: {plugin_path}")
    
    # Add system Python paths for CuPy
    system_paths = [
        r"C:\Users\Benchoff\AppData\Roaming\Python\Python312\site-packages",
        r"C:\Python312\Lib\site-packages"
    ]
    
    for path in system_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            print(f"✅ Added system path: {path}")

def test_imports():
    """Test all critical imports"""
    print("\n🧪 Testing imports...")
    
    try:
        import pcbnew
        print("✅ pcbnew imported successfully")
        
        # Test board loading
        board_path = os.path.join(os.path.dirname(__file__), "development", "testing", "test_board.kicad_pcb")
        if os.path.exists(board_path):
            board = pcbnew.LoadBoard(board_path)
            print(f"✅ Test board loaded: {board_path}")
            return board
        else:
            print(f"❌ Test board not found: {board_path}")
            return None
            
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return None

def test_cupy():
    """Test CuPy/GPU availability"""
    print("\n🚀 Testing GPU availability...")
    
    try:
        import cupy as cp
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        gpu_name = props["name"].decode("utf-8")
        print(f"✅ CuPy available - GPU: {gpu_name}")
        return True, cp, gpu_name
    except Exception as e:
        print(f"⚠️  CuPy not available: {e}")
        return False, None, "CPU Fallback"

def test_plugin_loading():
    """Test plugin loading and initialization"""
    print("\n📦 Testing plugin loading...")
    
    try:
        # Import the plugin
        from __init__ import OrthoRouteKiCadPlugin
        plugin = OrthoRouteKiCadPlugin()
        print("✅ Plugin class instantiated successfully")
        
        # Test plugin defaults
        plugin.defaults()
        print(f"✅ Plugin name: {plugin.name}")
        print(f"✅ Plugin description: {plugin.description}")
        
        return plugin
        
    except Exception as e:
        print(f"❌ Plugin loading failed: {e}")
        traceback.print_exc()
        return None

def run_headless_routing(plugin, board):
    """Run the routing algorithm headlessly"""
    print("\n🎯 Running headless routing test...")
    
    if not plugin or not board:
        print("❌ Cannot run routing - missing plugin or board")
        return False
    
    try:
        # Create a debug output capture
        debug_output = []
        
        def capture_debug(msg):
            debug_output.append(msg)
            print(f"[PLUGIN] {msg}")
        
        # Test the GPU routing function directly
        print("🔍 Testing GPU routing function...")
        
        # Get routing configuration (use defaults)
        config = {
            'grid_pitch': 0.1,
            'max_iterations': 3,
            'via_cost': 10,
            'batch_size': 10,
            'congestion_threshold': 5
        }
        
        print(f"📋 Using config: {config}")
        
        # Call the routing function directly
        try:
            result = plugin._route_board_gpu(board, config, debug_dialog=None)
            print(f"🎯 Routing result: {result}")
            
            if result and result.get('success'):
                print("✅ Headless routing succeeded!")
                return True
            else:
                print("❌ Headless routing failed")
                return False
                
        except Exception as e:
            print(f"❌ Routing function error: {e}")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Headless routing setup failed: {e}")
        traceback.print_exc()
        return False

def analyze_board(board):
    """Analyze the board to understand what we're working with"""
    print("\n📊 Analyzing board...")
    
    if not board:
        print("❌ No board to analyze")
        return
    
    try:
        # Get board dimensions
        bbox = board.GetBoardEdgesBoundingBox()
        print(f"📏 Board size: {bbox.GetWidth()/1e6:.2f} x {bbox.GetHeight()/1e6:.2f} mm")
        
        # Count nets
        nets = board.GetNetInfo()
        net_count = nets.NetsByName().__len__()
        print(f"🔗 Total nets: {net_count}")
        
        # Analyze nets with pads
        routeable_nets = []
        for net_item in nets:
            if net_item is None:
                continue
            net_name = net_item.GetNetname()
            netcode = net_item.GetNetCode()
            
            if netcode <= 0 or not net_name or net_name in ["", "No Net"]:
                continue
            
            # Count pads for this net
            pad_count = 0
            for footprint in board.GetFootprints():
                for pad in footprint.Pads():
                    if pad.GetNetCode() == netcode:
                        pad_count += 1
            
            if pad_count >= 2:
                routeable_nets.append((net_name, netcode, pad_count))
                print(f"  🎯 Net '{net_name}' (code {netcode}): {pad_count} pads")
        
        print(f"📈 Routeable nets: {len(routeable_nets)}")
        
        # Count footprints and pads
        footprint_count = len(list(board.GetFootprints()))
        total_pads = sum(len(list(fp.Pads())) for fp in board.GetFootprints())
        print(f"📦 Footprints: {footprint_count}")
        print(f"🔗 Total pads: {total_pads}")
        
        return routeable_nets
        
    except Exception as e:
        print(f"❌ Board analysis failed: {e}")
        traceback.print_exc()
        return []

def main():
    """Main headless debug function"""
    print("=" * 60)
    print("🐛 ORTHOROUTE HEADLESS DEBUG SESSION")
    print("=" * 60)
    print(f"🕒 Started: {datetime.datetime.now()}")
    print()
    
    # Create debug log file
    log_file_path = os.path.join(os.path.expanduser("~"), "Desktop", f"OrthoRoute_Headless_Debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    try:
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            # Redirect stdout to both console and file
            class TeeOutput:
                def __init__(self, file1, file2):
                    self.file1 = file1
                    self.file2 = file2
                
                def write(self, data):
                    self.file1.write(data)
                    self.file2.write(data)
                    self.file1.flush()
                    self.file2.flush()
                
                def flush(self):
                    self.file1.flush()
                    self.file2.flush()
            
            original_stdout = sys.stdout
            sys.stdout = TeeOutput(original_stdout, log_file)
            
            try:
                # Step 1: Setup paths
                setup_paths()
                
                # Step 2: Test imports and load board
                board = test_imports()
                
                # Step 3: Test GPU
                gpu_available, cp, gpu_name = test_cupy()
                
                # Step 4: Analyze board
                if board:
                    routeable_nets = analyze_board(board)
                else:
                    routeable_nets = []
                
                # Step 5: Test plugin loading
                plugin = test_plugin_loading()
                
                # Step 6: Run headless routing
                if plugin and board and routeable_nets:
                    success = run_headless_routing(plugin, board)
                    
                    if success:
                        print("\n🎉 HEADLESS DEBUG SUCCESS!")
                        print("The plugin works correctly in headless mode.")
                        print("The issue is likely in the UI/dialog system.")
                    else:
                        print("\n❌ HEADLESS DEBUG FAILED!")
                        print("The plugin has fundamental issues beyond UI.")
                else:
                    print("\n⚠️  HEADLESS DEBUG INCOMPLETE!")
                    print("Missing required components for full test.")
                
            finally:
                sys.stdout = original_stdout
                
    except Exception as e:
        print(f"❌ Debug session failed: {e}")
        traceback.print_exc()
    
    print()
    print("=" * 60)
    print(f"📄 Debug log saved to: {log_file_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
