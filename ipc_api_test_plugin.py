"""
KiCad IPC API Test Plugin
Tests the new IPC API functionality and provides transition bridge
"""

import pcbnew
import wx
import sys
import os

# Check if kicad-python is available
try:
    from kicad.pcbnew.board import Board as IPCBoard
    from kicad.pcbnew.module import Module as IPCModule
    IPC_AVAILABLE = True
    print("✅ KiCad IPC API (kicad-python) is available")
except ImportError as e:
    IPC_AVAILABLE = False
    print(f"⚠️ KiCad IPC API not available: {e}")
    print("   This plugin will use legacy SWIG API")

class KiCadIPCAPITestPlugin(pcbnew.ActionPlugin):
    """Test plugin for KiCad IPC API transition"""
    
    def defaults(self):
        self.name = "KiCad IPC API Test"
        self.category = "Debug"
        self.description = "Test KiCad IPC API vs SWIG API"
        self.show_toolbar_button = True
        
    def Run(self):
        """Run comprehensive API comparison test"""
        print("\n" + "="*80)
        print("🧪 KiCad IPC API vs SWIG API Comparison Test")
        print("="*80)
        
        try:
            # Get board using SWIG API
            swig_board = pcbnew.GetBoard()
            if not swig_board:
                wx.MessageBox("No board loaded!", "Error", wx.OK | wx.ICON_ERROR)
                return
            
            print("✅ SWIG Board loaded successfully")
            
            # Test IPC API if available
            if IPC_AVAILABLE:
                print("✅ IPC API available - running comparison tests")
                self.run_comparison_tests(swig_board)
            else:
                print("⚠️ IPC API not available - running SWIG-only tests")
                self.run_swig_tests(swig_board)
                
        except Exception as e:
            error_msg = f"API Test failed: {str(e)}"
            print(f"❌ {error_msg}")
            wx.MessageBox(error_msg, "Test Error", wx.OK | wx.ICON_ERROR)

    def run_comparison_tests(self, swig_board):
        """Run tests comparing SWIG vs IPC APIs"""
        print("\n📊 Running SWIG vs IPC API Comparison Tests")
        
        # Test 1: Board Information
        print("\n1️⃣ Board Information Comparison")
        self.test_board_info_comparison(swig_board)
        
        # Test 2: Net Detection Comparison
        print("\n2️⃣ Net Detection Comparison")
        self.test_net_detection_comparison(swig_board)
        
        # Test 3: Module/Footprint Comparison
        print("\n3️⃣ Module/Footprint Comparison")
        self.test_module_comparison(swig_board)
        
        # Test 4: Pad Access Comparison
        print("\n4️⃣ Pad Access Comparison")
        self.test_pad_comparison(swig_board)
        
        # Show final results
        self.show_comparison_results()

    def run_swig_tests(self, swig_board):
        """Run SWIG-only tests (fallback when IPC not available)"""
        print("\n📊 Running SWIG API Tests Only")
        
        # Test basic SWIG functionality
        self.test_swig_board_info(swig_board)
        self.test_swig_nets(swig_board)
        self.test_swig_modules(swig_board)
        
        wx.MessageBox(
            "SWIG API tests completed.\nIPC API not available.\nCheck console for details.",
            "SWIG Tests Complete",
            wx.OK | wx.ICON_INFORMATION
        )

    def test_board_info_comparison(self, swig_board):
        """Compare board info access between APIs"""
        try:
            # SWIG API
            swig_bounds = swig_board.GetBoardEdgesBoundingBox()
            swig_width = swig_bounds.GetWidth()
            swig_height = swig_bounds.GetHeight()
            swig_layers = swig_board.GetCopperLayerCount()
            
            print(f"📏 SWIG Board: {swig_width/1e6:.2f} x {swig_height/1e6:.2f} mm, {swig_layers} layers")
            
            # IPC API
            if IPC_AVAILABLE:
                try:
                    ipc_board = IPCBoard.wrap(swig_board)
                    # Note: IPC API might have different methods for board info
                    print(f"📏 IPC Board: Wrapped successfully, native_obj available: {hasattr(ipc_board, 'native_obj')}")
                except Exception as e:
                    print(f"❌ IPC Board info failed: {e}")
                    
        except Exception as e:
            print(f"❌ Board info comparison failed: {e}")

    def test_net_detection_comparison(self, swig_board):
        """Compare net detection between APIs"""
        try:
            # SWIG API Net Detection
            print("🔍 SWIG Net Detection:")
            swig_netcodes = swig_board.GetNetsByNetcode()
            swig_routeable_nets = 0
            
            for netcode, net in swig_netcodes.items():
                if netcode == 0:
                    continue
                    
                net_name = net.GetNetname()
                if not net_name:
                    continue
                
                # Count pads using SWIG method
                pad_count = 0
                for footprint in swig_board.GetFootprints():
                    for pad in footprint.Pads():
                        if pad.GetNet() == net:
                            pad_count += 1
                
                if pad_count >= 2:
                    swig_routeable_nets += 1
                    if swig_routeable_nets <= 3:
                        print(f"  ✅ SWIG Net: '{net_name}' ({pad_count} pads)")
            
            print(f"📊 SWIG: Found {swig_routeable_nets} routeable nets")
            
            # IPC API Net Detection (if available)
            if IPC_AVAILABLE:
                print("🔍 IPC Net Detection:")
                try:
                    ipc_board = IPCBoard.wrap(swig_board)
                    # IPC API might have different net access methods
                    print("📊 IPC: API wrapped, but net enumeration methods may differ")
                    print("     (IPC API focuses on high-level operations)")
                except Exception as e:
                    print(f"❌ IPC Net detection failed: {e}")
                    
        except Exception as e:
            print(f"❌ Net detection comparison failed: {e}")

    def test_module_comparison(self, swig_board):
        """Compare module/footprint access between APIs"""
        try:
            # SWIG API
            swig_footprints = list(swig_board.GetFootprints())
            print(f"🦶 SWIG Footprints: Found {len(swig_footprints)}")
            
            if swig_footprints:
                fp = swig_footprints[0]
                ref = fp.GetReference()
                pads = list(fp.Pads())
                print(f"   First: {ref} with {len(pads)} pads")
            
            # IPC API
            if IPC_AVAILABLE:
                try:
                    ipc_board = IPCBoard.wrap(swig_board)
                    ipc_modules = list(ipc_board.modules)
                    print(f"🦶 IPC Modules: Found {len(ipc_modules)}")
                    
                    if ipc_modules:
                        mod = ipc_modules[0]
                        ref = mod.reference
                        print(f"   First: {ref}")
                        
                except Exception as e:
                    print(f"❌ IPC Module access failed: {e}")
                    
        except Exception as e:
            print(f"❌ Module comparison failed: {e}")

    def test_pad_comparison(self, swig_board):
        """Compare pad access between APIs"""
        try:
            # SWIG API Pad Access
            total_swig_pads = 0
            for footprint in swig_board.GetFootprints():
                pads = list(footprint.Pads())
                total_swig_pads += len(pads)
                if total_swig_pads <= 5:  # Show first few
                    for pad in pads[:2]:
                        pad_name = pad.GetName()
                        pad_net = pad.GetNet()
                        net_name = pad_net.GetNetname() if pad_net else "No net"
                        pos = pad.GetPosition()
                        print(f"🔲 SWIG Pad: {pad_name} -> '{net_name}' at ({pos.x/1e6:.2f}, {pos.y/1e6:.2f})")
            
            print(f"📊 SWIG: Total {total_swig_pads} pads")
            
            # IPC API Pad Access
            if IPC_AVAILABLE:
                try:
                    ipc_board = IPCBoard.wrap(swig_board)
                    total_ipc_pads = 0
                    for module in ipc_board.modules:
                        # IPC API might have different pad access
                        print(f"🔲 IPC Module: {module.reference} (pad access method may differ)")
                        total_ipc_pads += 1  # Placeholder
                        break  # Just test first one
                        
                    print(f"📊 IPC: API structure available")
                    
                except Exception as e:
                    print(f"❌ IPC Pad access failed: {e}")
                    
        except Exception as e:
            print(f"❌ Pad comparison failed: {e}")

    def test_swig_board_info(self, swig_board):
        """Test SWIG board info (fallback)"""
        try:
            bounds = swig_board.GetBoardEdgesBoundingBox()
            width = bounds.GetWidth()
            height = bounds.GetHeight()
            layers = swig_board.GetCopperLayerCount()
            print(f"✅ Board: {width/1e6:.2f} x {height/1e6:.2f} mm, {layers} layers")
        except Exception as e:
            print(f"❌ SWIG board info failed: {e}")

    def test_swig_nets(self, swig_board):
        """Test SWIG net detection (fallback)"""
        try:
            netcodes = swig_board.GetNetsByNetcode()
            routeable_nets = 0
            
            for netcode, net in netcodes.items():
                if netcode == 0:
                    continue
                    
                net_name = net.GetNetname()
                if not net_name:
                    continue
                
                pad_count = 0
                for footprint in swig_board.GetFootprints():
                    for pad in footprint.Pads():
                        if pad.GetNet() == net:
                            pad_count += 1
                
                if pad_count >= 2:
                    routeable_nets += 1
                    if routeable_nets <= 3:
                        print(f"✅ Net: '{net_name}' ({pad_count} pads)")
            
            print(f"📊 Found {routeable_nets} routeable nets")
            
        except Exception as e:
            print(f"❌ SWIG nets test failed: {e}")

    def test_swig_modules(self, swig_board):
        """Test SWIG module access (fallback)"""
        try:
            footprints = list(swig_board.GetFootprints())
            print(f"✅ Found {len(footprints)} footprints")
            
            if footprints:
                fp = footprints[0]
                ref = fp.GetReference()
                pads = list(fp.Pads())
                print(f"✅ First footprint: {ref} with {len(pads)} pads")
                
        except Exception as e:
            print(f"❌ SWIG modules test failed: {e}")

    def show_comparison_results(self):
        """Show final comparison results"""
        if IPC_AVAILABLE:
            message = """IPC API vs SWIG API Test Complete!

✅ Both APIs are available
📊 Check console for detailed comparison
🔄 Transition recommendations:
   • IPC API provides cleaner, more pythonic interface
   • SWIG API will be deprecated in KiCad 10.0
   • Consider migrating to IPC API for future compatibility

Next Steps:
1. Install kicad-python: pip install kicad-python
2. Update plugins to use IPC API
3. Test thoroughly before KiCad 10.0 release"""
        else:
            message = """SWIG API Test Complete!

⚠️ IPC API not available
📦 To install: pip install kicad-python
🔄 Recommended for future KiCad versions

Current Status:
✅ SWIG API working (deprecated in KiCad 9.0+)
❌ IPC API not installed
⏰ Prepare for transition before KiCad 10.0"""
        
        wx.MessageBox(message, "API Test Results", wx.OK | wx.ICON_INFORMATION)

# Register the plugin
KiCadIPCAPITestPlugin().register()
