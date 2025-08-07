#!/usr/bin/env python3
"""
Hybrid KiCad Plugin - Works with both SWIG and IPC APIs
This plugin attempts to register with whatever system is available
"""

import os
import sys

def run_ipc_plugin():
    """Run the plugin using IPC API"""
    print("🚀 Running OrthoRoute via IPC API...")
    
    try:
        from kipy import KiCad
        from kipy.util.units import to_mm
        
        # Get connection info
        api_socket = os.environ.get('KICAD_API_SOCKET')
        api_token = os.environ.get('KICAD_API_TOKEN')
        
        if not api_socket or not api_token:
            print("⚠️  No IPC API environment - plugin may be running standalone")
            return
        
        # Connect and show basic info
        kicad = KiCad()
        print("✅ Connected to KiCad via IPC API")
        
        # Try to get board info
        try:
            board = kicad.get_board()
            if board:
                bbox = board.get_bounding_box()
                width_mm = to_mm(bbox.width)
                height_mm = to_mm(bbox.height)
                print(f"📏 Board: {width_mm:.1f} × {height_mm:.1f} mm")
            else:
                print("📋 No active board found")
        except Exception as e:
            print(f"⚠️  Could not get board info: {e}")
        
        print("🎯 OrthoRoute IPC plugin completed successfully!")
        
    except ImportError as e:
        print(f"❌ IPC API not available: {e}")
        raise

def run_swig_plugin():
    """Run the plugin using SWIG API (legacy)"""
    print("🚀 Running OrthoRoute via SWIG API (Legacy)...")
    
    try:
        import pcbnew
        import wx
        
        # Get current board
        board = pcbnew.GetBoard()
        
        # Get board info
        bbox = board.GetBoundingBox()
        width_mm = pcbnew.ToMM(bbox.GetWidth())
        height_mm = pcbnew.ToMM(bbox.GetHeight())
        
        # Count footprints
        footprints = list(board.GetFootprints())
        footprint_count = len(footprints)
        
        # Show info dialog
        message = f"OrthoRoute GPU Autorouter\\n\\n"
        message += f"Board: {width_mm:.1f} × {height_mm:.1f} mm\\n"
        message += f"Footprints: {footprint_count}\\n\\n"
        message += f"This is the legacy SWIG version.\\n"
        message += f"For KiCad 9.0+, use the IPC version."
        
        wx.MessageBox(message, "OrthoRoute - Legacy Mode", wx.OK | wx.ICON_INFORMATION)
        
        print("🎯 OrthoRoute SWIG plugin completed successfully!")
        
    except ImportError as e:
        print(f"❌ SWIG API not available: {e}")
        raise

# Try IPC first, fall back to SWIG
def main():
    """Main entry point - try IPC, fall back to SWIG"""
    print("🔍 OrthoRoute Plugin - Detecting available APIs...")
    
    # Try IPC API first (modern)
    try:
        run_ipc_plugin()
        return 0
    except ImportError:
        print("⚠️  IPC API not available, trying SWIG...")
    except Exception as e:
        print(f"⚠️  IPC API failed: {e}, trying SWIG...")
    
    # Fall back to SWIG API (legacy)
    try:
        run_swig_plugin()
        return 0
    except ImportError:
        print("❌ Neither IPC nor SWIG API available")
        print("   For IPC: Install kicad-python (pip install kicad-python)")
        print("   For SWIG: Use KiCad 8.0 or earlier")
        return 1
    except Exception as e:
        print(f"❌ SWIG API failed: {e}")
        return 1

# SWIG plugin registration (if available)
try:
    import pcbnew
    import wx
    
    class OrthoRouteHybridPlugin(pcbnew.ActionPlugin):
        def defaults(self):
            self.name = "OrthoRoute GPU Autorouter (Hybrid)"
            self.category = "Routing"
            self.description = "GPU-accelerated autorouter (IPC/SWIG hybrid)"
            self.show_toolbar_button = True
            icon_path = os.path.join(os.path.dirname(__file__), "resources", "icon.png")
            if os.path.exists(icon_path):
                self.icon_file_name = icon_path

        def Run(self):
            main()

    # Register SWIG plugin
    OrthoRouteHybridPlugin().register()
    print("✅ OrthoRoute registered as SWIG ActionPlugin")
    
except ImportError:
    print("ℹ️  SWIG API not available - IPC-only mode")

# Entry point for direct execution or IPC plugin system
if __name__ == "__main__":
    try:
        exit_code = main()
        print(f"🏁 Plugin finished with exit code: {exit_code}")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
